import torch
from .predictor_modules import *
from typing import Optional
from torch import Tensor

def get_norm(norm: str):
    if norm == "bn":
        return nn.BatchNorm1d
    elif norm == "ln":
        return nn.LayerNorm
    else:
        raise NotImplementedError


def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError
def build_mlp(c_in, channels, norm=None, activation="relu"):
    layers = []
    num_layers = len(channels)

    if norm is not None:
        norm = get_norm(norm)

    activation = get_activation(activation)

    for k in range(num_layers):
        if k == num_layers - 1:
            layers.append(nn.Linear(c_in, channels[k], bias=True))
        else:
            if norm is None:
                layers.extend([nn.Linear(c_in, channels[k], bias=True), activation()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, channels[k], bias=False),
                        norm(channels[k]),
                        activation(),
                    ]
                )
            c_in = channels[k]

    return nn.Sequential(*layers)
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2 = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src
class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1,
        use_ego_history=False,
        state_dropout=0.75):
        super(Encoder, self).__init__()
        self._lane_len = 80
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=11)
        self.ego_encoder = AgentEncoder(agent_dim=7)
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

        if not use_ego_history:
            self.ego_state_encoder = StateAttentionEncoder(
                7, dim, state_dropout
            )
        self._use_ego_history = use_ego_history
        encoder_depth = 4
        drop_path = 0.2
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.norm = nn.LayerNorm(dim)

        drop_rate = 0.
        attn_drop_rate = 0.
        mlp_ratio = 4
        qkv_bias = True
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            for i in range(encoder_depth)])
        self.pruning_loc = [1, 2, 3]

    def forward(self, inputs, keep_ratio):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        current_ego = ego[:, -1, :].reshape(ego.shape[0],-1,ego.shape[2])

        # agent encoding
        if not self._use_ego_history:
            current_ego = current_ego[:, 0, :]
            # current_ego = current_ego[:,:3]
            encoded_ego = self.ego_state_encoder(current_ego)
        else:
            encoded_ego = self.ego_encoder(current_ego)

        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)

        # encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # # fusion encoder from plantf
        # for blk in self.encoder_blocks:
        #     input = blk(input, key_padding_mask=mask)
        # encoding = self.norm(input)

        # fusion encoder with token attention prune
        neighbors_future = inputs['neighbors_future']
        B = input.shape[0]
        N1 = input.shape[1]
        num_token_all = input.shape[1]  # N + 1
        num_agents = neighbors_future.shape[1]
        token_idx_agent = torch.arange(0, num_agents).long().unsqueeze(0).expand(B, -1).to(input.device)
        token_idx_map = torch.arange(0, num_token_all - num_agents - 1).long().unsqueeze(0).expand(B, -1).to(input.device)  # (B, N)      initial
        attn_mask = torch.ones(B, N1, N1).to(input.device)  # (B, N+1, N+1) 1
        ratio = attn_ratio = keep_ratio
        actors = actors[:, 1:]
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                input, token_idx_agent, token_idx_map, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                       token_idx_agent=token_idx_agent,
                                                                       token_idx_map=token_idx_map, token_prune=True,
                                                                       attn_prune=True, attn_mask=attn_mask,
                                                                       key_padding_mask=mask,
                                                                       attn_keep_ratio=attn_ratio)
                ego_mask = mask[:, 0:1]
                map_mask = batched_index_select(mask[:, token_idx_agent.shape[1] + 1:], 1, token_idx_map)
                actors = batched_index_select(actors, 1, token_idx_agent)
                actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)
                mask = torch.cat([ego_mask, actors_mask, map_mask], dim=1)
                neighbors_future = batched_index_select(neighbors_future, 1, token_idx_agent)
            else:
                input, token_idx_agent, token_idx_map, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                       token_idx_agent=token_idx_agent,
                                                                       token_idx_map=token_idx_map, token_prune=False,
                                                                       attn_prune=False, attn_mask=attn_mask,
                                                                       key_padding_mask=mask,
                                                                       attn_keep_ratio=attn_ratio)

        encoding = self.norm(input)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'route_lanes': inputs['route_lanes'],
            'neighbors_future': neighbors_future
        }

        return encoder_outputs

class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1),
                                          nn.Linear(256, self._future_len * 2))
        self.route_encoder = VectorMapEncoder(3, 80)
        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=512,
            num_modes=6,
            future_steps=self._future_len,
            out_channels=3,
        )
        self.plan_predicter = GMMTraj_Predictor()
        self.agent_predictor = build_mlp(512, [512 * 2, self._future_len * 2], norm="ln")
        self.norm = nn.LayerNorm(512)

    def forward(self, encoder_outputs, route_lanes, initial_state):
        bs, A, T, _ = encoder_outputs['actors'].shape
        encoding = encoder_outputs['encoding']
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([encoding, route_encoding], dim=-1)
        # env_route_encoding = self.norm(env_route_encoding)

        # control = self.plan_decoder(env_route_encoding)
        # control = control.reshape(-1, self._future_len, 2)
        # plan = self.dynamics_layer(control, initial_state)

        # env_route_encoding = torch.max(env_route_encoding, dim=1)[0]  # max pooling over modalities
        ## use trajectory decode from plantf
        # trajectory, probability = self.trajectory_decoder(env_route_encoding)

        ## use trajectory decode from GMMpreodictor
        trajectory, probability = self.trajectory_decoder(env_route_encoding[:, 0])
        # predictions[..., :2] += initial_state[:, None, None, :2]
        prediction = self.agent_predictor(env_route_encoding[:, :A]).view(bs, -1, self._future_len, 2)
        # best_mode = probability.argmax(dim=-1)
        # plan = trajectory[torch.arange(env_encoding.shape[0]), best_mode]

        # return plan
        return trajectory, probability, prediction


class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        # self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward(self, inputs, keep_ratio):
        encoder_outputs = self.encoder(inputs, keep_ratio)
        route_lanes = encoder_outputs['route_lanes']
        initial_state = encoder_outputs['actors'][:, 0, -1]
        ego_plan, probability, prediction = self.planner(encoder_outputs, route_lanes, initial_state)

        return ego_plan, probability, prediction, encoder_outputs['neighbors_future']