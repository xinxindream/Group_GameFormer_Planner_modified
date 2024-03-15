import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output

class StateAttentionEncoder(nn.Module):
    def __init__(self, state_channel, dim, state_dropout=0.5) -> None:
        super().__init__()

        self.state_channel = state_channel
        self.state_dropout = state_dropout
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_channel, dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x):
        x_embed = []
        for i, linear in enumerate(self.linears):
            x_embed.append(linear(x[:, i, None]))
        x_embed = torch.stack(x_embed, dim=1)
        pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)
        x_embed += pos_embed

        if self.training and self.state_dropout > 0:
            visible_tokens = torch.zeros(
                (x_embed.shape[0], 3), device=x.device, dtype=torch.bool
            )
            dropout_tokens = (
                torch.rand((x_embed.shape[0], self.state_channel - 3), device=x.device)
                < self.state_dropout
            )
            key_padding_mask = torch.concat([visible_tokens, dropout_tokens], dim=1)
        else:
            key_padding_mask = None

        query = self.query.repeat(x_embed.shape[0], 1, 1)

        x_state = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )[0]

        return x_state[:, 0]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., token_num=196):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent

        eps = 1e-6
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (B, N+1, C) -> (B, N, 3C) -> (B, N+1, 3, H, C/H) -> (3, B, H, N+1, C/H)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)                 # (B, H, N+1, C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale       #  (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        # Key pruning (attention level) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        attn = attn.to(torch.float32).exp_() * attn_mask.unsqueeze(1).to(torch.float32)     # (B, H, N+1, N+1)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)          # (B, H, N+1, N+1)
        # attn = attn.softmax(dim=-1)                                           # (B, H, N+1, N+1)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # save attention map
        cls_attn = attn[:, :, 0, 1:].sum(1) / self.num_heads                      # (B, H, N) -> (B, N)
        patch_attn = attn[:, :, 1:, 1:].sum(1) / self.num_heads                   # (B, H, N, N) -> (B, N, N)
        return x, cls_attn, patch_attn

# Sparse Token >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def batched_index_select(input, dim, index):
    # input:(B, C, HW). index(B, N)
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)      # (B,C, N)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, token_keep_ratio=1.0, attn_keep_ratio=1.0, token_idx_agent=None, token_idx_map=None, token_prune=False, attn_prune=False,
                attn_mask=None):

        x_att, cls_attn, patch_attn = self.attn(self.norm1(x), attn_mask)
        # x: (B, N+1, C)
        # cls_attn: (B, N)      [cls] token, sum is 1
        # patch_attn: (B, N, N)     for each image patch
        x = x + self.drop_path(x_att)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Token Prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if token_prune:
            # print('keep_ratio',keep_ratio)
            x_cls_token = x[:, 0:1]  # (B, 1, C)
            B, N = cls_attn.shape  # N = 196
            agent_N = token_idx_agent.shape[1]
            map_N = token_idx_map.shape[1]
            cls_attn_agent = cls_attn[:,:agent_N]
            cls_attn_map = cls_attn[:, agent_N:]
            num_keep_node_agent = math.ceil(agent_N * token_keep_ratio)  # 196 r
            num_keep_node_map = math.ceil(map_N * token_keep_ratio)  # 196 r
            # attentive token
            token_idx_agent = cls_attn_agent.topk(num_keep_node_agent, dim=1)[1]
            token_idx_map = cls_attn_map.topk(num_keep_node_map, dim=1)[1] # (B, rN)        without gradient
            x_attentive_agent = batched_index_select(x[:, 1:agent_N+1], 1, token_idx_agent)
            x_attentive_map = batched_index_select(x[:, agent_N+1:], 1, token_idx_map)# (B, N, C) -> (B, rN, C)

            x = torch.cat([x_cls_token, x_attentive_agent, x_attentive_map], dim=1)
            # x = torch.cat([x_cls_token, x_attentive], dim=1)  # (B, 1+rN, C)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attention Prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if attn_prune:
            # get attention map of pruned token
            patch_attn_agent = patch_attn[:,:agent_N,:agent_N]
            patch_attn_prune_agent = batched_index_select(patch_attn_agent, 1, token_idx_agent)  # (B, N, N) -> (B, rN, N)
            patch_attn_prune_agent = batched_index_select(patch_attn_prune_agent, 2, token_idx_agent)  # (B, rN, N) -> (B, rN, rN)
            #
            B, rN1, _ = x.shape
            rN = rN1 - 1
            num_keep_attn_agent = math.ceil(agent_N * attn_keep_ratio)  # rN * ra
            top_val, _ = patch_attn_prune_agent.topk(dim=2, k=num_keep_attn_agent)  # (B, rN, rN * ra)

            attn_mask_p_agent = (patch_attn_prune_agent >= top_val[:, :, -1].unsqueeze(-1).expand(-1, -1,
                                                                                      num_keep_attn_agent)) + 0  # （B, rN, rN） without gradient     0/1 mask

            patch_attn_map= patch_attn[:, agent_N:, agent_N:]
            patch_attn_prune_map = batched_index_select(patch_attn_map, 1,
                                                          token_idx_map)  # (B, N, N) -> (B, rN, N)
            patch_attn_prune_map = batched_index_select(patch_attn_prune_map, 2,
                                                          token_idx_map)  # (B, rN, N) -> (B, rN, rN)
            #
            B, rN1, _ = x.shape
            rN = rN1 - 1
            num_keep_attn_map = math.ceil(map_N * attn_keep_ratio)  # rN * ra
            top_val, _ = patch_attn_prune_map.topk(dim=2, k=num_keep_attn_map)  # (B, rN, rN * ra)

            attn_mask_p_map = (patch_attn_prune_map >= top_val[:, :, -1].unsqueeze(-1).expand(-1, -1,
                                                                                                  num_keep_attn_map)) + 0  # （B, rN, rN） without gradient     0/1 mask

            # TODO: may add some random here

            attn_mask = torch.ones(B, rN1, rN1).to(x.device)  # (B, rN+1, rN+1)
            attn_mask[:, 1:num_keep_attn_agent + 1, 1:num_keep_attn_agent + 1] = attn_mask_p_agent
            attn_mask[:, num_keep_attn_agent + 1:, num_keep_attn_agent + 1:] = attn_mask_p_map

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # x = x + self.drop_path(self.attn(self.norm1(x)))          # old form
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, token_idx_agent, token_idx_map, attn_mask

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, attn_mask=None):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         if attn_mask is not None:
#             attn = attn * attn_mask.unsqueeze(1).unsqueeze(1)
#
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         # save attention map
#         cls_attn = attn[:, :, 0, 1:].sum(1) / self.num_heads                      # (B, H, N) -> (B, N)
#         patch_attn = attn[:, :, 1:, 1:].sum(1) / self.num_heads                   # (B, H, N, N) -> (B, N, N)
#
#         return x, cls_attn, patch_attn

class TrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, future_steps, out_channels) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        loc = self.loc(x).view(-1, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(x).squeeze(-1)

        return loc, pi


class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256))

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-6)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        output = torch.max(trajs, dim=-2).values

        return output


class GMMPredictor(nn.Module):
    def __init__(self, modalities=6):
        super(GMMPredictor, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 1))
    
    def forward(self, input):
        B, N, M, _ = input.shape
        traj = self.gaussian(input).view(B, N, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return traj, score

class GMMTraj_Predictor(nn.Module):
    def __init__(self, modalities=6):
        super(GMMTraj_Predictor, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        # self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1),
        #                               nn.Linear(512, self._future_len * 4))
        self.gaussian = nn.Sequential(nn.ELU(), nn.Dropout(0.1),
                                      nn.Linear(512, self._future_len * 3))
        self.score = nn.Sequential(nn.Linear(512, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, input):
        B, M, _ = input.shape
        traj = self.gaussian(input).view(B, M, self._future_len, 3)
        score = self.score(input).squeeze(-1)

        return traj, score

class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class CrossTransformer_timemask(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer_timemask, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, attn_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class InitialPredictionDecoder(nn.Module):
    def __init__(self, modalities, neighbors, dim=256):
        super(InitialPredictionDecoder, self).__init__()
        self._modalities = modalities
        self._agents = neighbors + 1
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(self._agents, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor()
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(self._agents).long())

    def forward(self, current_states, encoding, mask):
        N = self._agents
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent)
        query = encoding[:, :N, None] + multi_modal_query[None, :, :] + agent_query[:, None, :]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask) for i in range(N)], dim=1)
        predictions, scores = self.predictor(query_content)
        predictions[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, predictions, scores


    # time_mask = torch.tril(torch.ones(80, 80)).to(query.device)
    # time_mask = time_mask.unsqueeze(0).expand(query.shape[2], -1, -1).clone()
    # # time_mask = time_mask.expand(query.shape[0], query.shape[1], -1, -1)
    # query_contents = []
    # for i in range(N):
    #     # combined_mask = time_mask.clone()
    #     # combined_mask[:, i:, :, :] = 1
    #     # query_i = query[:, i].reshape(query.shape[0], query.shape[2] * query.shape[3], query.shape[4])
    #     query_i = query[:, i]
    #     # for j in range(query.shape[2]):
    #     #     query_j = query_i[:,j]
    #     #     query_content_j = self.query_encoder(query_j, query_j, query_j, time_mask)
    #
    #     query_content_i, _ = self.query_encoder(query_i, encoding, encoding, time_mask)
    #     query_content_i = query_content_i.view(query.shape[0], query.shape[2], query.shape[3], query.shape[4])
    #     query_contents.append(query_content_i)
    # query_content = torch.stack(query_contents, dim=1)


class InteractionDecoder(nn.Module):
    def __init__(self, modalities, future_encoder):
        super(InteractionDecoder, self).__init__()
        self.modalities = modalities
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor()

    def forward(self, current_states, actors, scores, last_content, encoding, mask):
        N = actors.shape[1]
        
        # using future encoder to encode the future trajectories
        multi_futures = self.future_encoder(actors[..., :2], current_states[:, :N])
        
        # using scores to weight the encoded futures
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)    
        
        # using self-attention to encode the interaction
        interaction = self.interaction_encoder(futures, mask[:, :N])
        
        # append the interaction encoding to the common content
        encoding = torch.cat([interaction, encoding], dim=1)

        # mask out the corresponding agents
        mask = torch.cat([mask[:, :N], mask], dim=1)
        mask = mask.unsqueeze(1).expand(-1, N, -1).clone()
        for i in range(N):
            mask[:, i, i] = 1

        # using cross-attention to decode the future trajectories
        query = last_content + multi_futures
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask[:, i]) for i in range(N)], dim=1)
        trajectories, scores = self.decoder(query_content)
        
        # add the current states to the trajectories
        trajectories[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, trajectories, scores