import torch
from .predictor_modules import *

class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1,
        use_ego_history=True,
        state_dropout=0.75):
        super(Encoder, self).__init__()
        self._lane_len = 50
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
            self.ego_encoder = StateAttentionEncoder(
                7, dim, state_dropout
            )

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        current_ego = ego[:, -1, :].reshape(ego.shape[0],-1,ego.shape[2])
        # agent encoding
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

        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'route_lanes': inputs['route_lanes']
        }

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()

        # initial level
        self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)

        # level-k reasoning
        self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

    def forward(self, encoder_outputs):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1]
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']

        # level 0 decode
        last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask)
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        env_encoding = last_content[:, 0]

        
        # # level k reasoning
        # for k in range(1, self.levels+1):
        #     interaction_decoder = self.interaction_stage[k-1]
        #     last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask)
        #     decoder_outputs[f'level_{k}_interactions'] = last_level
        #     decoder_outputs[f'level_{k}_scores'] = last_score
        #
        # env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding


class Refined_Decoder(nn.Module):
    def __init__(self, dim=256, future_len=80, levels=3):
        super(Refined_Decoder, self).__init__()
        self._future_len = future_len
        attention_mask = torch.ones((future_len, future_len)).bool()
        causal_mask = ~torch.tril(attention_mask, diagonal=0)  # diagonal=0, keep the diagonal
        self.register_buffer("causal_mask", causal_mask)
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, dim))
        self.ego_plan_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, dim))
        self.embed_timestep = nn.Embedding(future_len, dim)
        self.query_encoder_timemask = CrossTransformer_timemask()
        self.query_encoder = CrossTransformer()
        self.interaction_encoder = SelfTransformer()
        self.predictor = GMMPredictor()
        self.route_fusion = CrossTransformer()
        self.route_encoder = VectorMapEncoder(3, 50)
        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=512,
            num_modes=6,
            future_steps=self._future_len,
            out_channels=3,
        )

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-6)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1)  # (x, y, heading, vx, vy)

        return trajs

    def forward(self, ego_plan, decoder_outputs, encoder_outputs, env_encoding):
        current_states = encoder_outputs['actors'][:, :, -1]
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']
        route_lanes = encoder_outputs['route_lanes']

        k = 0
        N = decoder_outputs[f'level_{k}_interactions'].shape[1]
        trajs = self.state_process(decoder_outputs[f'level_{k}_interactions'][..., :2], current_states[:, :N])
        trajs_embeddings = self.mlp(trajs.detach())
        ego_plan_embeddings = self.ego_plan_mlp(ego_plan.detach())
        # time_mask = torch.tril(torch.ones(self._future_len, self._future_len)).to(ego_plan.device)
        timestep = torch.arange(self._future_len - self._future_len, self._future_len, device=ego_plan.device)
        time_embeddings_ = self.embed_timestep(timestep)
        time_embeddings_.unsqueeze(0).expand(ego_plan_embeddings.shape[1], -1, -1).clone()
        ego_plan_embeddings = ego_plan_embeddings + time_embeddings_[None]
        causal_mask = self.causal_mask[:self._future_len, :self._future_len]

        query_contents = []
        # using casual time mask cross-attention to decode the future multi ego-plan and mutlti agent trajectories
        for i in range(N):
            query_i = trajs_embeddings[:, i]
            # query_content_i = self.query_encoder(query_i, ego_plan_embeddings, ego_plan_embeddings, mask)
            # query_contents.append(query_content_i)
            query_i_contents = []
            for j in range(trajs_embeddings.shape[2]):
                query_j = query_i[:,j]
                ego_plan_j = ego_plan_embeddings[:,j]
                query_content_j = self.query_encoder_timemask(query_j, ego_plan_j, ego_plan_j, causal_mask)
                query_i_contents.append(query_content_j)
            query_content_i = torch.stack(query_i_contents, dim=1)
            query_contents.append(query_content_i)
        query_content = torch.stack(query_contents, dim=1)
        query = torch.max(query_content, dim=-2).values
        futures = (torch.max(trajs_embeddings, dim=-2).values * decoder_outputs['level_0_scores'].softmax(-1).unsqueeze(-1)).mean(dim=2)

        # using self-attention to encode the interaction
        interaction = self.interaction_encoder(futures, mask[:, :N])

        # append the interaction encoding to the common content
        encoding = torch.cat([interaction, encoding], dim=1)
        # mask out the corresponding agents
        refined_mask = mask.clone()
        refined_mask = torch.cat([refined_mask[:, :N], refined_mask], dim=1)
        refined_mask = refined_mask.unsqueeze(1).expand(-1, N, -1).clone()
        for i in range(N):
            refined_mask[:, i, i] = 1
        refined_query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, refined_mask[:, i]) for i in range(N)],dim=1)
        # refined_query_content = self.interaction_encoder(refined_query_content, mask[:, :N])
        trajectories, scores = self.predictor(refined_query_content)
        # add the current states to the trajectories
        trajectories[..., :2] += current_states[:, :N, None, None, :2]
        env_encoding = refined_query_content[:, 0]
        decoder_outputs['level_1_interactions'] = trajectories
        decoder_outputs['level_1_scores'] = scores
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        env_route_encoding = torch.max(env_route_encoding, dim=1)[0]  # max pooling over modalities
        ego_refined_plan, ego_refined_scores = self.trajectory_decoder(env_route_encoding)

        return decoder_outputs, ego_refined_plan, ego_refined_scores


class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))
        self.route_encoder = VectorMapEncoder(3, 50)
        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=512,
            num_modes=6,
            future_steps=self._future_len,
            out_channels=3,
        )

    def dynamics_layer(self, controls, initial_state):       
        dt = 0.1 # discrete time period [s]
        max_a = 5 # vehicle's accleration limits [m/s^2]
        max_d = 0.5 # vehicle's steering limits [rad]
        
        vel_init = torch.hypot(initial_state[..., 3], initial_state[..., 4])
        vel = vel_init[:, None] + torch.cumsum(controls[..., 0].clamp(-max_a, max_a) * dt, dim=-1)
        vel = torch.clamp(vel, min=0)

        yaw_rate = controls[..., 1].clamp(-max_d, max_d) * vel
        yaw = initial_state[:, None, 2] + torch.cumsum(yaw_rate * dt, dim=-1)
        yaw = torch.fmod(yaw, 2*torch.pi)

        vel_x = vel * torch.cos(yaw)
        vel_y = vel * torch.sin(yaw)

        x = initial_state[:, None, 0] + torch.cumsum(vel_x * dt, dim=-1)
        y = initial_state[:, None, 1] + torch.cumsum(vel_y * dt, dim=-1)

        return torch.stack((x, y, yaw), dim=-1)

    def forward(self, env_encoding, route_lanes, initial_state):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        env_route_encoding = torch.max(env_route_encoding, dim=1)[0] # max pooling over modalities
        # control = self.plan_decoder(env_route_encoding)
        # control = control.reshape(-1, self._future_len, 2)
        # plan = self.dynamics_layer(control, initial_state)
        trajectory, probability = self.trajectory_decoder(env_route_encoding)
        # best_mode = probability.argmax(dim=-1)
        # plan = trajectory[torch.arange(env_encoding.shape[0]), best_mode]

        # return plan
        return trajectory, probability
    
class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()
        self.refined_decoder = Refined_Decoder()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        initial_state = encoder_outputs['actors'][:, 0, -1]
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan, probability = self.planner(env_encoding, route_lanes, initial_state)
        decoder_outputs, ego_plan, probability = self.refined_decoder(ego_plan, decoder_outputs, encoder_outputs, env_encoding)

        return decoder_outputs, ego_plan, probability