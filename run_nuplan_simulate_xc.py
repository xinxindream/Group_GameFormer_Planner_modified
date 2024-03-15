import time
import argparse
import datetime
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings("ignore")
import torch

# import depth_estimator as depth_estimator

from tqdm import tqdm
from Planner.planner import Planner
from common_utils import *

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile

def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[:, 1] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[:, 0] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)

def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    if agent_type == 'ego':
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2].float()
    else:
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state

def polyline_process(polylines, avails, traffic_light=None):
    # dim = 3 if traffic_light is None else 7
    dim = 7
    new_polylines = torch.zeros((polylines.shape[0], polylines.shape[1], dim), dtype=torch.float32)
    traffic_light = torch.zeros((polylines.shape[1], 4), dtype=torch.float32)
    traffic_light[:, -1] = 1

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_heading = torch.atan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0])
            polyline_heading = torch.fmod(polyline_heading, 2*torch.pi)
            polyline_heading = torch.cat([polyline_heading, polyline_heading[-1].unsqueeze(0)], dim=0).unsqueeze(-1)
            polyline_heading = torch.cat([polyline, polyline_heading], dim=-1)

            polyline = global_state_se2_tensor_to_local(polyline_heading[:][:], polyline_heading[20][:], precision=torch.float64)
            # if traffic_light is None:
            #     new_polylines[i] = torch.cat([polyline, polyline_heading], dim=-1)
            # else:
            # new_polylines[i] = torch.cat([polyline, polyline_heading, traffic_light], dim=-1)
            new_polylines[i] = torch.cat([polyline, traffic_light], dim=-1)

    return new_polylines
def interpolate_points(coords: torch.Tensor, max_points: int, interpolation: str) -> torch.Tensor:
    """
    Interpolate points within map element to maintain fixed size.
    :param coords: Sequence of coordinate points representing map element. <torch.Tensor: num_points, 2>
    :param max_points: Desired size to interpolate to.
    :param interpolation: Torch interpolation mode. Available options: 'linear' and 'area'.
    :return: Coordinate points interpolated to max_points size.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}. Expected shape: (*, 2)")

    x_coords = coords[:, 0].unsqueeze(0).unsqueeze(0)
    y_coords = coords[:, 1].unsqueeze(0).unsqueeze(0)
    align_corners = True if interpolation == 'linear' else None
    x_coords = torch.nn.functional.interpolate(x_coords, max_points, mode=interpolation, align_corners=align_corners)
    y_coords = torch.nn.functional.interpolate(y_coords, max_points, mode=interpolation, align_corners=align_corners)
    coords = torch.stack((x_coords, y_coords), dim=-1).squeeze()

    return coords
def create_map_raster(lanes, crosswalks, route_lanes):
    lane = np.array(lanes)
    crosswalk = np.array(crosswalks)
    route_lane = np.array(route_lanes)

    plt.plot(lane[:, 0], lane[:, 1], 'c', linewidth=3) # plot centerline

    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4) # plot crosswalk

    plt.plot(route_lane[:, 0], route_lane[:, 1], 'g', linewidth=4) # plot route_lanes

def plot_scenario(data):
    # 读取图片
    img = plt.imread('/media/xingchen24/xingchen4T/datasets/navigation/deeplearning/kxdun/车道线标注返修交付_20231024/front_camera_images/03051.jpg')

    # Create map layers
    create_map_raster(data['left_white_solid_points'], data['right_white_solid_points'], data['center_white_solid_points'])

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_scenario_tensor(plan, lanes, neighbor_agents_past):

    plan = plan.cpu()
    plan = plan.detach().numpy()
    plan = plan[0,:,:]
    lanes = lanes.cpu()
    lanes = lanes.detach().numpy()
    lanes = lanes[0,:,:,:]
    plan = np.array(plan)
    lanes = np.array(lanes)
    neighbor_agents_past = neighbor_agents_past.cpu()
    neighbor_agents_past = neighbor_agents_past.detach().numpy()
    neighbor_agents_past = neighbor_agents_past[0,:,0,:]
    neighbor_agents_past = np.array(neighbor_agents_past)

    for i in range(neighbor_agents_past.shape[0]):
        if neighbor_agents_past[i, 0] != 0:
            x_center, y_center, heading = neighbor_agents_past[i, 0], neighbor_agents_past[i, 1], neighbor_agents_past[i, 2]
            agent_length, agent_width = neighbor_agents_past[i, 6], neighbor_agents_past[i, 7]
            agent_bottom_right = (x_center - agent_length / 2, y_center - agent_width / 2)

            rect = plt.Rectangle(agent_bottom_right, agent_length, agent_width, linewidth=2, color='m', alpha=0.6,
                                 zorder=3,
                                 transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center),
                                                                                   heading) + plt.gca().transData)
            plt.gca().add_patch(rect)

    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if lane[1][1] != 0:
            plt.plot(lane[:, 0], lane[:, 1], 'c', linewidth=1) # plot centerline

    plt.plot(100, 100)
    plt.plot(-100, -100)
    #
    # lanes = lanes[0, 0, :, :]
    # plt.plot(lanes[:, 0], lanes[:, 1], 'g', linewidth=2)
    #
    plt.plot(plan[:, 0], plan[:, 1], 'r', linewidth=1)

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        tensor_data[k] = v.float().unsqueeze(0).to(device)

    return tensor_data
def read_lanes_json(lanes_json_path):
    # _mat_frontcam2rightlidar, _, _RT_rightlidar2imu, _, _, _RT_imulink2baselink = calib_get.calib_mat()
    # 读取JSON文件
    with open(lanes_json_path, 'r') as f:
        data = json.load(f)

    max_elements = {'LANE': 40, 'ROUTE_LANES': 10,
                    'CROSSWALK': 5}  # maximum number of elements to extract per feature layer.
    max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30}
    align_corners = True
    interpolation = 'linear'
    num_agents = 20
    device = 'cuda'
    # max_points = 50

    left_white_solids = []
    right_white_solids = []
    # for lanes in data['shapes']:
    left_white_solid_points = data['shapes'][0]['points']
    right_white_solid_points = data['shapes'][1]['points']

    coords_tensor = torch.zeros((max_elements['LANE'], max_points['LANE'], 2), dtype=torch.float32)
    avails_tensor = torch.zeros((max_elements['LANE'], max_points['LANE']), dtype=torch.bool)

    center_white_solid_points = []
    for i in range(len(left_white_solid_points)):
        x = (left_white_solid_points[i][0] + right_white_solid_points[i][0]) / 2
        y = (left_white_solid_points[i][1] + right_white_solid_points[i][1]) / 2
        center_white_solid_points.append((x, y))

    center_white_solid_points_tensor = torch.tensor(center_white_solid_points[0:20][:])
    coords = interpolate_points(center_white_solid_points_tensor, max_points['LANE'], interpolation)
    coords_tensor[0] = coords
    # coords_tensor[1][:,-1] = coords_tensor[1][:,-1]+4
    # coords_tensor[2][:,-1] = coords_tensor[2][:,-1]-4
    avails_tensor[0] = True  # specify real vs zero-padded data
    vector_map_lanes_local = polyline_process(coords_tensor, avails_tensor)
    vector_map_lanes_local[0][:, 1] = 1
    vector_map_lanes_local[1] = vector_map_lanes_local[0]
    vector_map_lanes_local[2] = vector_map_lanes_local[0]

    vector_map_lanes_local[1][:, 1] = vector_map_lanes_local[0][:, 1] - 4
    vector_map_lanes_local[2][:, 1] = vector_map_lanes_local[0][:, 1] + 4
    # ego_poses = convert_absolute_quantities_to_relative(vector_map_lanes[0][:21][:], vector_map_lanes[0][20][:])
    ego_agent_past = vector_map_lanes_local[0][:21][:]
    ego_agent_past[:, -1] = 0.1
    ego_agent_past[:, -2] = 0.1
    ego_agent_past[:, -3] = 0
    ego_agent_past[:, -4] = 1
    vector_map_crosswalks = torch.zeros((max_elements['CROSSWALK'], max_points['CROSSWALK'], 3), dtype=torch.float32)
    vector_map_route_lanes = torch.zeros((max_elements['ROUTE_LANES'], max_points['ROUTE_LANES'], 3), dtype=torch.float32)
    neighbor_agents_past = torch.zeros((num_agents, 21, 11), dtype=torch.float32)
    obs_lane_i = 2
    obs_i = 24
    v_x = 0
    v_y = 0
    neighbor_agents_past[0,:,:] = torch.tensor([vector_map_lanes_local[obs_lane_i][obs_i][0],vector_map_lanes_local[obs_lane_i][obs_i][1],vector_map_lanes_local[obs_lane_i][obs_i][2],v_x,v_y,0,4.6,1.8,0,1,0])
    neighbor_agents_past[1, :, :] = torch.tensor(
        [vector_map_lanes_local[2][obs_i][0], vector_map_lanes_local[2][obs_i][1],
         vector_map_lanes_local[2][obs_i][2], v_x, v_y, 0, 4.6, 1.8, 0, 1, 0])
    vector_map_lanes_local0 = vector_map_lanes_local[0]
    vector_map_route_lanes[0] = vector_map_lanes_local[0][:, :3]
    vector_map_route_lanes[1] = vector_map_lanes_local[1][:, :3]
    vector_map_route_lanes[2] = vector_map_lanes_local[2][:, :3]
    vector_map_output = {'map_lanes': vector_map_lanes_local, 'map_crosswalks': vector_map_crosswalks,
                         'route_lanes': vector_map_route_lanes}
    data = {"ego_agent_past": ego_agent_past,
            "neighbor_agents_past": neighbor_agents_past}
    data.update(vector_map_output)
    data = convert_to_model_inputs(data, device)

    data_ori = {"left_white_solid_points": left_white_solid_points, "right_white_solid_points": right_white_solid_points, "center_white_solid_points": center_white_solid_points}
    return data, data_ori

def main(args):
    # parameters
    experiment_name = args.experiment_name
    job_name = 'gameformer_planner'
    experiment_time = datetime.datetime.now()
    experiment = f"{experiment_name}/{job_name}/{experiment_time}"
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"
    aggregator_metric_dir = "aggregator_metric"

    data, data_ori = read_lanes_json('/media/xingchen24/xingchen4T/datasets/navigation/deeplearning/kxdun/车道线标注返修交付_20231024/front_camera_images/03051.json')
    # initialize planner
    planner = Planner(args.model_path, args.device)
    planner._initialize_model()
    plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = planner._get_prediction(data)
    data_ori.update({"plan": plan})

    plot_scenario_tensor(plan, data['map_lanes'], data['neighbor_agents_past'])
    # plot_scenario(data_ori)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NuPlan test')
    parser.add_argument('--experiment_name', choices=['open_loop_boxes', 'closed_loop_nonreactive_agents',
                                                      'closed_loop_reactive_agents'], help='experiment name')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--map_path', type=str, help='path to nuplan maps')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model on')
    parser.add_argument('--scenarios_per_type', type=int, default=1, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', default=None, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')
    args = parser.parse_args()

    main(args)
