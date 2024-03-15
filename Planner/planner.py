import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from .planner_utils import *
from .observation import *
from GameFormer.predictor import GameFormer
from .state_lattice_path_planner import LatticePlanner

import osqp
from scipy import sparse

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class Planner(AbstractPlanner):
    def __init__(self, model_path, device=None):
        self._max_path_length = MAX_LEN # [m]
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._target_speed = 13.0 # [m/s]
        self._N_points = int(T/DT)
        self._model_path = model_path

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._device = device
    
    def name(self) -> str:
        return "GameFormer Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TrajectoryPlanner()
        self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length)

    def _initialize_model(self):
        # The parameters of the model should be the same as the one used in training
        self._model = GameFormer(encoder_layers=3, decoder_levels=1)
        
        # Load trained model
        self._model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        # print('model load')
        
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]
    
    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
            
        # In case the ego vehicle is not on the route, return None
        if closest_distance > 5:
            return None

        # Get reference path, handle exception
        try:
            ref_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        except:
            ref_path = None

        if ref_path is None:
            return None

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        
        return ref_path.astype(np.float32)

    # def _get_prediction(self, features):
    #     predictions, plan, probability = self._model(features)
    #
    #     best_mode = probability.argmax(dim=-1)
    #     output_trajectory = plan[torch.arange(plan.shape[0]), best_mode]
    #     plan = output_trajectory
    #
    #     K = len(predictions) // 2 - 1
    #     final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
    #     final_scores = predictions[f'level_{K}_scores']
    #     ego_current = features['ego_agent_past'][:, -1]
    #     neighbors_current = features['neighbor_agents_past'][:, :, -1]
    #
    #     return plan, final_predictions, final_scores, ego_current, neighbors_current

    def _get_prediction(self, features):
        final_multi_plan, probability, prediction = self._model(features)

        best_mode = probability.argmax(dim=-1)
        output_trajectory = final_multi_plan[torch.arange(final_multi_plan.shape[0]), best_mode]
        plan = output_trajectory

        final_predictions = None
        final_scores = None
        ego_current = None
        neighbors_current = None

        return plan, final_predictions, final_scores, ego_current, neighbors_current, final_multi_plan

    def quad_prog_smoother(self, init_path):
        init_path = init_path[0].cpu().numpy()

        # 相关参数
        w_cost_smooth = 50  # 平滑代价权重
        w_cost_length = 2  # 紧凑代价权重
        w_cost_ref = 10  # 偏移代价权重
        ref_max = 0.4  # 边界约束的上限

        size = init_path.shape[0]

        # A1
        A1 = np.zeros((2 * size - 4, 2 * size))
        for j in range(0, 2 * size - 5, 2):
            A1[j, j] = 1
            A1[j, j + 2] = -2
            A1[j, j + 4] = 1
            A1[j + 1, j + 1] = 1
            A1[j + 1, j + 3] = -2
            A1[j + 1, j + 5] = 1

        # A2
        A2 = np.zeros((2 * size - 2, 2 * size))
        for k in range(0, 2 * size - 3, 2):
            A2[k, k] = 1
            A2[k, k + 2] = -1
            A2[k + 1, k + 1] = 1
            A2[k + 1, k + 3] = -1

        # A3
        A3 = np.eye(2 * size)

        # f、lb、ub
        f = np.zeros((2 * size, 1))
        lb = np.zeros((2 * size, 1))
        ub = np.zeros((2 * size, 1))
        for i in range(size):
            f[2 * i] = init_path[i, 0]
            f[2 * i + 1] = init_path[i, 1]
            lb[2 * i] = f[2 * i] - ref_max
            ub[2 * i] = f[2 * i] + ref_max
            lb[2 * i + 1] = f[2 * i + 1] - ref_max
            ub[2 * i + 1] = f[2 * i + 1] + ref_max

        # H
        H = 2 * (w_cost_smooth * A1.T @ A1 + w_cost_length * A2.T @ A2 + w_cost_ref * A3)

        # 二次规划求解
        P = sparse.csc_matrix(H)
        q = -2 * w_cost_ref * f.ravel()

        A = sparse.csc_matrix(A3)
        l = lb.ravel()
        u = ub.ravel()

        # 创建 OSQP 对象
        prob = osqp.OSQP()
        prob.setup(P, q, A=A, l=l, u=u, verbose=False)

        # 求解
        res = prob.solve()

        if res.info.status != 'solved':
            return False

        QPSolution = res.x

        refined_path = np.zeros((size,3))

        # 将结果输出到 referenceline 中
        for index in range(size):
            refined_path[index, 0] = QPSolution[2 * index]
            refined_path[index, 1] = QPSolution[2 * index + 1]
            refined_path[index, 2] = math.atan2(refined_path[index, 1], refined_path[index, 0])

        return refined_path

    def _plan(self, ego_state, history, traffic_light_data, observation):
        # Construct input features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # Get reference path
        # ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        # Infer prediction model
        with torch.no_grad():
            plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)

        smooth_plan = self.quad_prog_smoother(plan)
        plan = plan[0].cpu().numpy()

        plt.plot(plan[:, 0], plan[:, 1]*10, 'r', linewidth=2)
        plt.plot(smooth_plan[:, 0], smooth_plan[:, 1]*10 + 40, 'b', linewidth=1)

        # plt.plot(100, 100)
        # plt.plot(-100, -100)

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()


        # plan = plan[0].cpu().numpy()

        # # Trajectory refinement
        # with torch.no_grad():
        #     plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
        #                                          predictions, plan, scores, ref_path, observation)
            
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)

        return trajectory
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        trajectory = self._plan(ego_state, history, traffic_light_data, observation)
        print(f'Iteration {iteration}: {time.time() - s:.3f} s')

        return trajectory
