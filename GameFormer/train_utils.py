import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DrivingData(Dataset):
    def __init__(self, data_dir, n_neighbors):
        self.data_list = glob.glob(data_dir)
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        # 增加了列切片控制
        neighbors = data['neighbor_agents_past'][:self._n_neighbors]
        route_lanes = data['route_lanes'] 
        map_lanes = data['lanes']
        map_crosswalks = data['crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt


def imitation_loss(gmm, scores, ground_truth):
    B, N = gmm.shape[0], gmm.shape[1]
    distance = torch.norm(gmm[:, :, :, :, :2] - ground_truth[:, :, None, :, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)

    mu = gmm[..., :2]
    best_mode_mu = mu[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_mu = best_mode_mu.squeeze(2)
    dx = ground_truth[..., 0] - best_mode_mu[..., 0]
    dy = ground_truth[..., 1] - best_mode_mu[..., 1]

    cov = gmm[..., 2:]
    best_mode_cov = cov[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_cov = best_mode_cov.squeeze(2)
    log_std_x = torch.clamp(best_mode_cov[..., 0], -2, 2)
    log_std_y = torch.clamp(best_mode_cov[..., 1], -2, 2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    # KL divergence penalty for uncertainty
    # Assuming a target distribution with a mean of 0 and a standard deviation of 1 (unit Gaussian)
    target_std = torch.ones_like(std_x)
    kl_div_x = torch.log(target_std / std_x) + (std_x.pow(2) + dx.pow(2)) / (2 * target_std.pow(2)) - 0.5
    kl_div_y = torch.log(target_std / std_y) + (std_y.pow(2) + dy.pow(2)) / (2 * target_std.pow(2)) - 0.5
    kl_divergence = kl_div_x + kl_div_y
    kl_divergence = 0.5 * torch.mean(kl_divergence)

    gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
    gmm_loss = torch.mean(gmm_loss)
    # # GMM loss calculation with KL divergence
    # gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx / std_x) + torch.square(dy / std_y))
    # gmm_loss = torch.mean(gmm_loss) + kl_divergence

    # # environmnet uncertainty loss
    uncertainty = calculate_scene_complexity_and_uncertainty(gmm)
    uncertainty_loss = torch.mean(uncertainty)

    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, label_smoothing=0.2, reduction='none')
    score_loss = score_loss * torch.ne(ground_truth[:, :, 0, 0], 0)
    score_loss = torch.mean(score_loss)
    
    loss = gmm_loss + score_loss
    # loss = gmm_loss + score_loss + uncertainty_loss

    return loss, best_mode_mu, best_mode


def level_k_loss(outputs, ego_future, neighbors_future, neighbors_future_valid):
    loss: torch.tensor = 0
    levels = len(outputs.keys()) // 2 
    gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)

    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions']
        scores = outputs[f'level_{k}_scores']
        predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
        plan = trajectories[:, :1]
        trajectories = torch.cat([plan, predictions], dim=1)
        il_loss, future, best_mode = imitation_loss(trajectories, scores, gt_future)
        loss += il_loss 

    return loss, future

def planning_loss(plan, ego_future, ego_multi_plan, k_trajectories, neighbors_future_valid):
    loss = F.smooth_l1_loss(plan, ego_future)
    loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1])

    neighbors_valid = neighbors_future_valid[:, :, 0, 0]
    inter_loss = interaction_loss(ego_multi_plan, k_trajectories, neighbors_valid)
    loss += 0.5 * inter_loss

    return loss

# def interaction_loss(ego_plan, last_trajectories, neighbors_valid):
#     B, N, M, T, _ = last_trajectories.shape
#     neighbors_to_ego = []
#     neighbors_to_neighbors = []
#     neighbors_mask = neighbors_valid.logical_not()
#     mask = torch.zeros(B, N - 1, N - 1).to(neighbors_mask.device)
#     mask = torch.masked_fill(mask, neighbors_mask[:, :, None], 1)
#     mask = torch.masked_fill(mask, neighbors_mask[:, None, :], 1)
#     mask = torch.masked_fill(mask, torch.eye(N - 1)[None, :, :].bool().to(neighbors_mask.device), 1)
#     mask = mask.unsqueeze(-1).unsqueeze(-1) * torch.ones(1, 1, M, M).to(neighbors_mask.device)
#     # mask = mask.permute(0, 1, 3, 2, 4).reshape(B, (N - 1) * M, (N - 1) * M)
#
#     for t in range(T):
#         # AV-agents last level
#         ego_p = ego_plan[:, :, t, :2]
#         # ego_p = last_trajectories[:, 0, :, t, :2]
#         last_neighbors_p = last_trajectories[:, 1:, :, t, :2]
#         last_neighbors_p = last_neighbors_p.reshape(B, -1, 2)
#         dist_to_ego = torch.cdist(ego_p, last_neighbors_p)
#         n_mask = neighbors_mask.unsqueeze(-1).expand(-1, -1, M).reshape(B, 1, -1)
#         dist_to_ego = torch.masked_fill(dist_to_ego, n_mask, 1000)
#         neighbors_to_ego.append(torch.min(dist_to_ego, dim=-1).values)
#
#         # # agents-agents last level
#         # neighbors_p = trajectories[:, 1:, :, t, :2].reshape(B, -1, 2)
#         # dist_neighbors = torch.cdist(neighbors_p, last_neighbors_p)
#         # dist_neighbors = torch.masked_fill(dist_neighbors, mask.bool(), 1000)
#         # neighbors_to_neighbors.append(torch.min(dist_neighbors, dim=-1).values)
#
#     neighbors_to_ego = torch.stack(neighbors_to_ego, dim=-1)
#     PF_to_ego = 1.0 / (neighbors_to_ego + 1)
#     PF_to_ego = PF_to_ego * (neighbors_to_ego < 3)  # safety threshold
#     PF_to_ego = PF_to_ego.sum(-1).sum(-1).mean()
#
#     # neighbors_to_neighbors = torch.stack(neighbors_to_neighbors, dim=-1)
#     # PF_to_neighbors = 1.0 / (neighbors_to_neighbors + 1)
#     # PF_to_neighbors = PF_to_neighbors * (neighbors_to_neighbors < 3)  # safety threshold
#     # PF_to_neighbors = PF_to_neighbors.sum(-1).mean(-1).mean()
#     PF_to_neighbors = 0
#
#     return PF_to_ego + PF_to_neighbors

def interaction_loss(ego_plan, last_trajectories, neighbors_future_valid):
    neighbors_valid = neighbors_future_valid[:, :, 0, 0]                           
    ego_plan = ego_plan.unsqueeze(1)
    last_trajectories = last_trajectories.unsqueeze(2)
    B, N, M, T, _ = last_trajectories.shape
    neighbors_to_ego = []
    neighbors_to_neighbors = []
    neighbors_mask = neighbors_valid.logical_not()
    # mask = torch.zeros(B, N - 1, N - 1).to(neighbors_mask.device)
    # mask = torch.masked_fill(mask, neighbors_mask[:, :, None], 1)
    # mask = torch.masked_fill(mask, neighbors_mask[:, None, :], 1)
    # mask = torch.masked_fill(mask, torch.eye(N - 1)[None, :, :].bool().to(neighbors_mask.device), 1)
    # mask = mask.unsqueeze(-1).unsqueeze(-1) * torch.ones(1, 1, M, M).to(neighbors_mask.device)
    # mask = mask.permute(0, 1, 3, 2, 4).reshape(B, (N - 1) * M, (N - 1) * M)

    for t in range(T):
        # AV-agents last level
        ego_p = ego_plan[:, :, t, :2]
        # ego_p = last_trajectories[:, 0, :, t, :2]
        last_neighbors_p = last_trajectories[:, :, :, t, :2]
        last_neighbors_p = last_neighbors_p.reshape(B, -1, 2)
        dist_to_ego = torch.cdist(ego_p, last_neighbors_p)
        n_mask = neighbors_mask.unsqueeze(-1).expand(-1, -1, M).reshape(B, 1, -1)
        dist_to_ego = torch.masked_fill(dist_to_ego, n_mask, 1000)
        neighbors_to_ego.append(torch.min(dist_to_ego, dim=-1).values)

        # # agents-agents last level
        # neighbors_p = trajectories[:, 1:, :, t, :2].reshape(B, -1, 2)
        # dist_neighbors = torch.cdist(neighbors_p, last_neighbors_p)
        # dist_neighbors = torch.masked_fill(dist_neighbors, mask.bool(), 1000)
        # neighbors_to_neighbors.append(torch.min(dist_neighbors, dim=-1).values)

    neighbors_to_ego = torch.stack(neighbors_to_ego, dim=-1)
    PF_to_ego = 1.0 / (neighbors_to_ego + 1)
    PF_to_ego = PF_to_ego * (neighbors_to_ego < 3)  # safety threshold
    PF_to_ego = PF_to_ego.sum(-1).sum(-1).mean()

    # neighbors_to_neighbors = torch.stack(neighbors_to_neighbors, dim=-1)
    # PF_to_neighbors = 1.0 / (neighbors_to_neighbors + 1)
    # PF_to_neighbors = PF_to_neighbors * (neighbors_to_neighbors < 3)  # safety threshold
    # PF_to_neighbors = PF_to_neighbors.sum(-1).mean(-1).mean()
    PF_to_neighbors = 0

    return PF_to_ego + PF_to_neighbors
def calculate_scene_complexity_and_uncertainty(input_tensor):
    """
    Calculate scene complexity and uncertainty based on input tensor.

    :param input_tensor: Tensor of shape [B, N, M, 80, 4], containing m_x, m_y, log_sig_x, log_sig_y.
    :return: Scene complexity and uncertainty measures.
    """
    B, N, M, T, _ = input_tensor.shape

    # 分离出坐标和对数标准差
    m_x, m_y, log_sig_x, log_sig_y = input_tensor.split(1, dim=-1)  # 每个的形状都是 [B, N, M, 80, 1]

    # 计算场景复杂度：考虑所有预测坐标的标准差
    m_x_std = torch.std(m_x, dim=[1, 2])  # [B, 80]
    m_y_std = torch.std(m_y, dim=[1, 2])  # [B, 80]
    scene_complexity = (m_x_std + m_y_std).mean(dim=-1) / 2  # 对所有时间步取平均

    # 计算不确定性：标准差的大小反映了不确定性
    std_x = torch.exp(log_sig_x).mean(dim=[1, 2])  # [B, 80]
    std_y = torch.exp(log_sig_y).mean(dim=[1, 2])  # [B, 80]
    uncertainty = (std_x + std_y).mean(dim=-1) / 2  # 对所有时间步取平均

    uncertainty = scene_complexity + 2 * uncertainty
    # uncertainty = torch.log1p(uncertainty)
    # uncertainty = torch.sigmoid(uncertainty)

    return uncertainty

def motion_metrics(plan_trajectory, prediction_trajectories, ego_future, neighbors_future, neighbors_future_valid):
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)
    heading_error = torch.abs(torch.fmod(plan_trajectory[:, :, 2] - ego_future[:, :, 2] + np.pi, 2 * np.pi) - np.pi)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    plannerAHE = torch.mean(heading_error)
    plannerFHE = torch.mean(heading_error[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), plannerAHE.item(), plannerFHE.item(), predictorADE.item(), predictorFDE.item()
