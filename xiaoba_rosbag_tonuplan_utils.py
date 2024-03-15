import torch
import math
import cv2
from GameFormer.data_utils import map_process, pad_agent_states_with_zeros, global_velocity_to_local, convert_absolute_quantities_to_relative, filter_agents_tensor
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import Point2D, StateVector2D
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import *


def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input tensor data of the ego past.
    :param past_time_stamps: The input tensor data of the past timestamps.
    :param past_time_stamps: The input tensor data of other agents in the past.
    :return: ego_agent_array, other_agents_array.
    """
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    """
    Model input feature representing the present and past states of the ego and agents, including:
    ego: <np.ndarray: num_frames, 7>
        The num_frames includes both present and past frames.
        The last dimension is the ego pose (x, y, heading) velocities (vx, vy) acceleration (ax, ay) at time t.
    agents: <np.ndarray: num_frames, num_agents, 8>
        Agent features indexed by agent feature type.
        The num_frames includes both present and past frames.
        The num_agents is padded to fit the largest number of agents across all frames.
        The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) and size (length, width) at time t.
    """

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(
                convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))

        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)

        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    '''
    Post-process the agents tensor to select a fixed number of agents closest to the ego vehicle.
    agents: <np.ndarray: num_agents, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle 
            and added to the feature of the agent
        The num_agents is padded or trimmed to fit the predefined number of agents across.
        The num_frames includes both present and past frames.
    '''
    agents = np.zeros(shape=(num_agents, agents_tensor.shape[0], agents_tensor.shape[-1] + 3), dtype=np.float32)

    # sort agents according to distance to ego
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # indices = [i for i in range(len(indices))]

    # fill agent features into the array
    for i, j in enumerate(indices):
        agents[i, :, :agents_tensor.shape[-1]] = agents_tensor[:, j, :agents_tensor.shape[-1]].numpy()
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_tensor.shape[-1]:] = [1, 0, 0]
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_tensor.shape[-1]:] = [0, 1, 0]
        else:
            agents[i, :, agents_tensor.shape[-1]:] = [0, 0, 1]

    return ego_tensor.numpy().astype(np.float32), agents, indices

def agent_future_process(current_ego_state, future_tracked_objects, num_agents, agent_index):
    '''
        functions:
            1. 处理交通参与者未来状态，并将其转换到以自车为中心的局部坐标系中的状态
            2. 具体转换（绝对坐标系坐标 --> 相对坐标系坐标）
        params:
            -
        returns:
            - agent_futures : shape(num_agents, 时间点数-1, 3) => 3表示x，y，heading
    '''
    # 构建自车为中心的相对坐标系状态张量，7个特征
    anchor_ego_state = torch.tensor([current_ego_state.position.x, current_ego_state.position.y,
                                      current_ego_state.euler_angles.z,
                                      current_ego_state.linear_velocity.x,
                                      current_ego_state.linear_velocity.y,
                                      current_ego_state.linear_acceleration.x,
                                      current_ego_state.linear_acceleration.y])

    # 一定规则过滤得到的agent未来状态
    agent_future = filter_agents_tensor(future_tracked_objects)
    
    # 局部坐标系状态
    local_coords_agent_states = []
    
    # 坐标转换
    for agent_state in agent_future:
        local_coords_agent_states.append \
            (convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
            
    # 填充agent坐标系状态
    padded_agent_states = pad_agent_states_with_zeros(local_coords_agent_states)

    # fill agent features into the array
    agent_futures = np.zeros(shape=(num_agents, padded_agent_states.shape[0]-1, 3), dtype=np.float32)
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[1:, j, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]].numpy()

    return agent_futures

def get_neighbor_agents_future(current_ego_state, agents_future_queue, agent_index, num_agents):
    future_tracked_objects_tensor_list = get_tracked_future_objects_to_tensor_list(agents_future_queue)
    agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, num_agents,
                                         agent_index)

    return agent_futures

def get_tracked_objects_to_tensor_list(agents_past_queue):
    # 创建一个空字典，用于保存每个物体的信息
    objects_list = []
    objects_type_list = []
    # 收集所有唯一的物体id
    unique_ids = {}
    sorted_set = set()
    for agents_past in agents_past_queue:
        for i, past_tracked_object in enumerate(agents_past.obstacle_list):
            if past_tracked_object.id not in unique_ids:
                unique_ids[past_tracked_object.id] = []
    sorted_set.update(unique_ids.keys())
    sorted_ids = sorted(sorted_set)

    for agents_past in agents_past_queue:
        # 创建一个物体数量 * 8 的张量
        obj_tensor = torch.zeros((len(agents_past.obstacle_list), 8), dtype=torch.float32)
        objects_type = []
        for i, past_tracked_object in enumerate(agents_past.obstacle_list):
            # obj_tensor[i, 0] = past_tracked_object.id
            obj_tensor[i, 0] = sorted_ids.index(past_tracked_object.id)
            if past_tracked_object.velocity > 0.5:
                obj_tensor[i, 1] = past_tracked_object.velocity*math.sin(past_tracked_object.vel_heading)
                obj_tensor[i, 2] = past_tracked_object.velocity*math.cos(past_tracked_object.vel_heading)
                obj_tensor[i, 3] = past_tracked_object.vel_heading
            else:
                obj_tensor[i, 1] = past_tracked_object.velocity * math.sin(past_tracked_object.heading) #可能考虑直接为0
                obj_tensor[i, 2] = past_tracked_object.velocity * math.cos(past_tracked_object.heading)
                obj_tensor[i, 3] = past_tracked_object.heading
            obj_tensor[i, 4] = past_tracked_object.width
            obj_tensor[i, 5] = past_tracked_object.length
            obj_tensor[i, 6] = past_tracked_object.position.x
            obj_tensor[i, 7] = past_tracked_object.position.y
            objects_type.append(TrackedObjectType.VEHICLE)
        # 按照 id 的大小对张量的行进行排序
        first_column = obj_tensor[:, 0]
        sorted_indices = torch.argsort(first_column)
        sorted_obj_tensor = obj_tensor[sorted_indices]
        objects_list.append(sorted_obj_tensor)
        objects_type_list.append(objects_type)
    return objects_list, objects_type_list

def get_tracked_future_objects_to_tensor_list(agents_past_queue):
    # 创建一个空字典，用于保存每个物体的信息
    objects_list = []
    objects_type_list = []
    # 收集所有唯一的物体id
    unique_ids = {}
    sorted_set = set()
    for agents_past in agents_past_queue:
        for i, past_tracked_object in enumerate(agents_past.obstacle_list):
            if past_tracked_object.id not in unique_ids:
                unique_ids[past_tracked_object.id] = []
    sorted_set.update(unique_ids.keys())
    sorted_ids = sorted(sorted_set)

    for agents_past in agents_past_queue:
        # 创建一个物体数量 * 8 的张量
        obj_tensor = torch.zeros((len(agents_past.obstacle_list), 8), dtype=torch.float32)
        objects_type = []
        for i, past_tracked_object in enumerate(agents_past.obstacle_list):
            obj_tensor[i, 0] = sorted_ids.index(past_tracked_object.id)
            if past_tracked_object.velocity > 0.5:
                obj_tensor[i, 1] = past_tracked_object.velocity*math.sin(past_tracked_object.vel_heading)
                obj_tensor[i, 2] = past_tracked_object.velocity*math.cos(past_tracked_object.vel_heading)
                obj_tensor[i, 3] = past_tracked_object.vel_heading
            else:
                obj_tensor[i, 1] = past_tracked_object.velocity * math.sin(past_tracked_object.heading) #可能考虑直接为0
                obj_tensor[i, 2] = past_tracked_object.velocity * math.cos(past_tracked_object.heading)
                obj_tensor[i, 3] = past_tracked_object.heading
            obj_tensor[i, 4] = past_tracked_object.width
            obj_tensor[i, 5] = past_tracked_object.length
            obj_tensor[i, 6] = past_tracked_object.position.x
            obj_tensor[i, 7] = past_tracked_object.position.y
            objects_type.append(TrackedObjectType.VEHICLE)
        # 按照 id 的大小对张量的行进行排序
        first_column = obj_tensor[:, 0]
        sorted_indices = torch.argsort(first_column)
        sorted_obj_tensor = obj_tensor[sorted_indices]
        objects_list.append(sorted_obj_tensor)
        objects_type_list.append(objects_type)
    return objects_list


def get_ego_past_to_tensor_list(_ego_past_queue):
    '''
    functions:
        队列变向量
        
    return:
        返回了包含7个特征的自车张量，分别是：位置信息的(x, y, z)， 矢量信息的（x轴速度，y轴速度，x轴加速度，y轴加速度）
    '''
    ego_tensor = torch.zeros((len(_ego_past_queue), 7), dtype=torch.float32)
    for i in range(len(_ego_past_queue)):
        ego_tensor[i, 0] = _ego_past_queue[i].position.x
        ego_tensor[i, 1] = _ego_past_queue[i].position.y
        ego_tensor[i, 2] = _ego_past_queue[i].euler_angles.z
        # ego_tensor[i, 3] = _ego_past_queue[i].linear_velocity * math.sin(_ego_past_queue[i].euler_angles.z)
        # ego_tensor[i, 4] = _ego_past_queue[i].linear_velocity * math.cos(_ego_past_queue[i].euler_angles.z)
        # ego_tensor[i, 5] = _ego_past_queue[i].linear_acceleration * math.sin(_ego_past_queue[i].euler_angles.z)
        # ego_tensor[i, 6] = _ego_past_queue[i].linear_acceleration * math.cos(_ego_past_queue[i].euler_angles.z)

        ego_tensor[i, 3] = _ego_past_queue[i].linear_velocity.x
        ego_tensor[i, 4] = _ego_past_queue[i].linear_velocity.y
        ego_tensor[i, 5] = _ego_past_queue[i].linear_acceleration.x
        ego_tensor[i, 6] = _ego_past_queue[i].linear_acceleration.y
    return ego_tensor

def get_ego_future_to_tensor_list(ego_state, _ego_past_queue):
    '''
        将自车（ego vehicle）的未来状态转换为与过去状态相对的位置列表
        自车的未来状态需要相对于其过去状态来描述，以便于模型更好地理解车辆的动态变化
    '''
    trajectory_relative_poses = convert_absolute_to_relative_poses(
        ego_state, [StateSE2(x=ego_past.position.x, y=ego_past.position.y,
                                 heading=ego_past.euler_angles.z) for ego_past in _ego_past_queue]
    )
    return trajectory_relative_poses
