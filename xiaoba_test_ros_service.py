import time

import rospy
import tf
from geometry_msgs.msg import Pose, Vector3, PoseStamped
from std_msgs.msg import Header, ColorRGBA
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler

from kxdun_perception_msgs.msg import PerceptionObstacle, PerceptionObstacleArray, PerceptionLaneArray, PerceptionLane
from kxdun_localization_msgs.msg import Localization
from kxdun_msgs.msg import CenterlineArray
from kxdun_control_msgs.msg import PlanningADCTrajectory, TrajectoryPoint

from kxdun_planning_srv.srv import PlannerService, PlannerServiceResponse

from visualization_msgs.msg import MarkerArray, Marker

from Planner.planner import Planner
from run_nuplan_ros_service_utils import *

import osqp
from scipy import sparse
from collections import deque
import threading
import copy

import argparse

class gameformer_planner(object):
    def __init__(self, model_path, device=None):
        self._model_path = model_path
        self._device = device
        self._num_agents = 20
        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK']  # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5}  # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 60, 'ROUTE_LANES': 60, 'CROSSWALK': 30}  # maximum number of points per feature to extract per feature layer.
        self._radius = 25  # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear'

        self.frame_id = 'map'

        self._perception_flag = False
        self._ego_flag = False
        self._lanes_flag = False
        self._routes_flag = False
        self._loc_connt = 0

        # self._agents_past_queue = []
        # self._ego_past_queue = []

        self._agents_past_queue = deque(maxlen = 22)
        self._ego_past_queue = deque(maxlen = 22)
        self._lock = threading.Lock()  # 创建一个线程锁

        self._map_lanes = None
        self._route_lanes = None
        self._added_obstacles = None
        self._last_data = None
        self._last_plan = None
        self._last_ego_state = None

        self._markers_pub = rospy.Publisher('/objects_markers', MarkerArray, queue_size=10)
        self._planner_path_pub = rospy.Publisher('/learn_based_planner_path_pub', Path, queue_size=10)
        self._planner_path_forcarla_pub = rospy.Publisher('/carla/ego_vehicle/trajectory',
                                                          PlanningADCTrajectory, queue_size=10)
        self._multi_path_pub  = []
        for plani in range(6):
            self._multi_path_pub.append(rospy.Publisher('/learn_based_planner_path_' + str(plani) + '_pub', Path, queue_size=10))

        self._planner = Planner(self._model_path, device)
        self._planner._initialize_model()

    def update_frame_data(self, static_scenario, dynamic_scenario):

        self._map_lanes = static_scenario
        self._route_lanes = static_scenario
        """
        添加新帧到队列
        """
        with self._lock:  # 获取锁
            self._agents_past_queue.append(dynamic_scenario)
            self._ego_past_queue.append(dynamic_scenario)

    def handle_planner_service(self, new_frame):
        # 从请求中获取输入
        static_scenario = new_frame.static_scenario
        dynamic_scenario = new_frame.dynamic_scenario
        self.update_frame_data(static_scenario, dynamic_scenario)

        # 使用深拷贝复制数据
        with self._lock:  # 获取锁
            agents_past_queue = copy.deepcopy(self._agents_past_queue)
            ego_past_queue = copy.deepcopy(self._ego_past_queue)

        if (len(self._agents_past_queue) == 22)&(len(self._ego_past_queue) == 22):
            # 如果队列中有 22 帧数据，则执行规划逻辑
            output_trajectory = self.execute_planning(agents_past_queue, ego_past_queue, self._map_lanes, self._route_lanes)
            # print('output_trajectory', output_trajectory)

            # 返回响应
            return PlannerServiceResponse(trajectory=output_trajectory)
        # else:
        #     # 如果数据不足 20 帧，则可能需要返回一个错误或等待更多数据
        #     return PlannerServiceResponse(error="Insufficient data")

    def execute_planning(self, agents_past_queue, ego_past_queue, map_lanes, route_lanes):
        """
        执行规划逻辑
        """
        # 使用 frame_data 进行规划
        # 这里是您的规划逻辑
        # 返回规划结果

        print('planing...')
        start = time.perf_counter()
        ego_agent_past = get_ego_past_to_tensor_list(ego_past_queue)
        past_tracked_objects_tensor_list, past_tracked_objects_types = get_tracked_objects_to_tensor_list(agents_past_queue)
        time_stamps_past = get_past_timestamps_to_tensor(ego_past_queue)
        ego_agent_past, neighbor_agents_past = agent_past_process(
            ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list, past_tracked_objects_types,
            self._num_agents
        )

        coords_map_lanes_polylines = get_map_lane_polylines(map_lanes)
        coords_route_lanes_polylines = get_route_lane_polylines(route_lanes)
        crosswalk: List[List[Point2D]] = []
        coords_crosswalk_polylines = MapObjectPolylines(crosswalk)
        coords: Dict[str, MapObjectPolylines] = {}
        # extract generic map objects
        coords[self._map_features[0]] = coords_map_lanes_polylines
        coords[self._map_features[1]] = coords_route_lanes_polylines
        coords[self._map_features[2]] = coords_crosswalk_polylines
        # traffic_light_encoding = np.zeros([len(self._map_lanes.centerlines),4], dtype=int)

        traffic_light_encoding = np.zeros([len(self._map_lanes.lane_net.lanes), 4], dtype=int)
        # print(len(self._map_lanes.centerlines))
        traffic_light_encoding[:, -1] = 1
        traffic_light_data_at_t: Dict[str, LaneSegmentTrafficLightData] = {}
        traffic_light_data: List[Dict[str, LaneSegmentTrafficLightData]] = []
        traffic_light_data_at_t[self._map_features[0]] = LaneSegmentTrafficLightData(
            list(map(tuple, traffic_light_encoding)))
        traffic_light_data = traffic_light_data_at_t

        # traffic_light_data.append(traffic_light_data_at_t)
        # print(traffic_light_data)
        ego_state = StateSE2(x=self._ego_past_queue[-1].vehicle_set.vehicles[0].state.vec_position.x, y=self._ego_past_queue[-1].vehicle_set.vehicles[0].state.vec_position.y,
                             heading=self._ego_past_queue[-1].vehicle_set.vehicles[0].state.angle)
        vector_map = map_process(ego_state, coords, traffic_light_data, self._map_features,
                                 self._max_elements, self._max_points, self._interpolation_method)

        data = {"ego_agent_past": ego_agent_past[1:],
                "neighbor_agents_past": neighbor_agents_past[:, 1:]}
        data.update(vector_map)
        data = convert_to_model_inputs(data, self._device)
        with torch.no_grad():
            plan, predictions, scores, ego_state_transformed, neighbors_state_transformed, final_multi_plan = self._planner._get_prediction(data)
        smooth_plan = plan[0].cpu().numpy()
        origin_pose = [ego_state.x, ego_state.y, ego_state.heading]
        absolute_poses = np.array(self.relative_to_absolute_poses(origin_pose, smooth_plan))
        headings = self.get_heading(absolute_poses)
        kappas = self.get_kappa(absolute_poses)

        adc_trajectory_pb = PlanningADCTrajectory()
        accumulated_trajectory_s = 0.0
        prev_trajectory_point = [0.0, 0.0]
        prev_trajectory_point[0] = absolute_poses[0, 0]
        prev_trajectory_point[1] = absolute_poses[0, 1]
        start_timestamp_ = rospy.Time.now().to_sec()
        adc_trajectory_pb.header.timestamp_sec = start_timestamp_
        adc_trajectory_pb.header.sequence_num = 0
        adc_trajectory_pb.header.frame_id = "map"
        adc_trajectory_pb.header.module_name = "planning"
        adc_trajectory_pb.header.status.error_code = 0
        adc_trajectory_pb.header.status.msg = ""
        adc_trajectory_pb.header.version = 1
        adc_trajectory_pb.complete_parking = 0  # 是否停车?
        adc_trajectory_pb.engage_advice.Advice = 3
        adc_trajectory_pb.engage_advice.reason = ""
        adc_trajectory_pb.estop.is_estop = 0  # 是否急停?
        adc_trajectory_pb.estop.reason = ""
        adc_trajectory_pb.gear = 1
        adc_trajectory_pb.is_replan = False  # 是否进行了重新规划?
        adc_trajectory_pb.trajectory_type = 1
        PathPointList: List[TrajectoryPoint] = []
        delt_time = 0.1
        points = absolute_poses[:, :2]
        points = np.insert(points, 0, [0, 0], axis=0)
        velocities = np.diff(points, axis=0) / delt_time
        velocities_magnitude = np.linalg.norm(velocities, axis=1)
        velocities_a = np.insert(velocities_magnitude, 0, [0], axis=0)
        accelerations = np.diff(velocities_a, axis=0) / delt_time
        for pi in range(absolute_poses.shape[0]):
            nPathPoint = TrajectoryPoint()
            nPathPoint.v = velocities_magnitude[pi]
            nPathPoint.a = accelerations[pi]
            nPathPoint.relative_time = start_timestamp_ + delt_time
            # nPathPoint.steer = state.steer
            nPathPoint.path_point.x = absolute_poses[pi, 0]
            nPathPoint.path_point.y = absolute_poses[pi, 1]
            nPathPoint.path_point.z = 0.0
            nPathPoint.path_point.theta = headings[pi]
            nPathPoint.path_point.kappa = kappas[pi]

            delta_x = nPathPoint.path_point.x - prev_trajectory_point[0]
            delta_y = nPathPoint.path_point.y - prev_trajectory_point[1]
            delta_as = math.hypot(delta_x, delta_y)
            accumulated_trajectory_s += delta_as
            prev_trajectory_point[0] = nPathPoint.path_point.x
            prev_trajectory_point[1] = nPathPoint.path_point.y
            nPathPoint.path_point.s = accumulated_trajectory_s
            PathPointList.append(nPathPoint)
        adc_trajectory_pb.trajectory_point = PathPointList
        # self._planner_path_forcarla_pub.publish(adc_trajectory_pb)
        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = "map"
        # path = Path(header=header)
        # for plani in range(absolute_poses.shape[0]):
        #     posestamp = PoseStamped(header=header)
        #     pose = Pose()
        #     pose.position.x = absolute_poses[plani, 0]
        #     pose.position.y = absolute_poses[plani, 1]
        #     pose.position.z = self._ego_past_queue[-1].position.z
        #     q = quaternion_from_euler(0, 0, absolute_poses[plani, 2])
        #     pose.orientation.x = q[0]
        #     pose.orientation.y = q[1]
        #     pose.orientation.z = q[2]
        #     pose.orientation.w = q[3]
        #     posestamp.pose = pose
        #     path.poses.append(posestamp)
        # self._planner_path_pub.publish(path)

        # final_multi_plan = final_multi_plan[0].cpu().numpy()
        # origin_pose = [ego_state.x, ego_state.y, ego_state.heading]
        # for multi_i in range(6):
        #     header = Header()
        #     header.stamp = rospy.Time.now()
        #     header.frame_id = "map"
        #     path = Path(header=header)
        #     absolute_poses = np.array(self.relative_to_absolute_poses(origin_pose, final_multi_plan[multi_i, :, :]))
        #     for plani in range(absolute_poses.shape[0]):
        #         posestamp = PoseStamped(header=header)
        #         pose = Pose()
        #         pose.position.x = absolute_poses[plani, 0]
        #         pose.position.y = absolute_poses[plani, 1]
        #         pose.position.z = 0
        #         q = quaternion_from_euler(0, 0, absolute_poses[plani, 2])
        #         pose.orientation.x = q[0]
        #         pose.orientation.y = q[1]
        #         pose.orientation.z = q[2]
        #         pose.orientation.w = q[3]
        #         posestamp.pose = pose
        #         path.poses.append(posestamp)
        #     self._multi_path_pub[multi_i].publish(path)
        end = time.perf_counter()
        print(f"运算时间： {end-start}")
        print('plan pub')

        return adc_trajectory_pb

    def run(self):
        rospy.init_node('planner_service_node')
        s = rospy.Service('PlannerService', PlannerService, self.handle_planner_service)
        rospy.spin()
        print('learn-based-planner service started')

    def get_heading(self, points):
        xy_points = points[:,:2]
        points_size = xy_points.shape[0]
        headings: List[float] = []
        dxs: List[float] = []
        dys: List[float] = []
        for i in range(points_size):
            x_delta = 0.0
            y_delta = 0.0
            if i == 0:
                x_delta = (xy_points[i + 1, 0] - xy_points[i, 0])
                y_delta = (xy_points[i + 1, 1] - xy_points[i, 1])
            elif i == points_size - 1:
                x_delta = (xy_points[i, 0] - xy_points[i - 1, 0])
                y_delta = (xy_points[i, 1] - xy_points[i - 1, 1])
            else:
                x_delta = 0.5 * (xy_points[i + 1, 0] - xy_points[i - 1, 0])
                y_delta = 0.5 * (xy_points[i + 1, 1] - xy_points[i - 1, 1])
            dxs.append(x_delta)
            dys.append(y_delta)

        # Heading calculation
        for i in range(points_size):
            headings.append(math.atan2(dys[i], dxs[i]))

        return headings

    def get_kappa(self, points):
        # Get linear interpolated s for dkappa calculation
        xy_points = points[:,:2]
        y_over_s_first_derivatives: List[float] = []
        x_over_s_first_derivatives: List[float] = []
        y_over_s_second_derivatives: List[float] = []
        x_over_s_second_derivatives: List[float] = []
        kappas: List[float] = []
        distance = 0.0
        accumulated_s: List[float] = []
        accumulated_s.append(distance)
        fx = xy_points[0, 0]
        fy = xy_points[0, 1]
        nx = 0.0
        ny = 0.0
        for i in range(1, xy_points.shape[0]):
            nx = xy_points[i, 0]
            ny = xy_points[i, 1]
            end_segment_s = math.sqrt((fx - nx) * (fx - nx) + (fy - ny) * (fy - ny))
            accumulated_s.append(end_segment_s + distance)
            distance += end_segment_s
            fx = nx
            fy = ny

        # Get finite difference approximated first derivative of y and x respective
        # to s for kappa calculation
        for i in range(xy_points.shape[0]):
            xds = 0.0
            yds = 0.0
            if i == 0:
                xds = (xy_points[i + 1, 0] - xy_points[i, 0]) / (accumulated_s[i + 1] - accumulated_s[i])
                yds = (xy_points[i + 1, 1] - xy_points[i, 1]) / (accumulated_s[i + 1] - accumulated_s[i])
            elif i == xy_points.shape[0] - 1:
                xds = (xy_points[i, 0] - xy_points[i - 1, 0]) / (accumulated_s[i] - accumulated_s[i - 1])
                yds = (xy_points[i, 1] - xy_points[i - 1, 1]) / (accumulated_s[i] - accumulated_s[i - 1])
            else:
                xds = (xy_points[i + 1, 0] - xy_points[i - 1, 0]) / (accumulated_s[i + 1] - accumulated_s[i - 1])
                yds = (xy_points[i + 1, 1] - xy_points[i - 1, 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1])
            x_over_s_first_derivatives.append(xds)
            y_over_s_first_derivatives.append(yds)

        # Get finite difference approximated second  derivative of y and x
        # respective to s for kappa calculation
        for i in range(xy_points.shape[0]):
            xdds = 0.0
            ydds = 0.0
            if i == 0:
                xdds = (x_over_s_first_derivatives[i + 1] - x_over_s_first_derivatives[i]) / (accumulated_s[i + 1] - accumulated_s[i])
                ydds = (y_over_s_first_derivatives[i + 1] - y_over_s_first_derivatives[i]) / (accumulated_s[i + 1] - accumulated_s[i])
            elif i == xy_points.shape[0] - 1:
                xdds = (x_over_s_first_derivatives[i] - x_over_s_first_derivatives[i - 1]) / (accumulated_s[i] - accumulated_s[i - 1])
                ydds = (y_over_s_first_derivatives[i] - y_over_s_first_derivatives[i - 1]) / (accumulated_s[i] - accumulated_s[i - 1])
            else:
                xdds = (x_over_s_first_derivatives[i + 1] - x_over_s_first_derivatives[i - 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1])
                ydds = (y_over_s_first_derivatives[i + 1] - y_over_s_first_derivatives[i - 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1])
            x_over_s_second_derivatives.append(xdds)
            y_over_s_second_derivatives.append(ydds)

        for i in range(xy_points.shape[0]):
            xds = x_over_s_first_derivatives[i]
            yds = y_over_s_first_derivatives[i]
            xdds = x_over_s_second_derivatives[i]
            ydds = y_over_s_second_derivatives[i]
            kappa = (xds * ydds - yds * xdds) /(math.sqrt(xds * xds + yds * yds) * (xds * xds + yds * yds) + 1e-6)
            kappas.append(kappa)

        return kappas

    def relative_to_absolute_poses(self,origin_pose, relative_poses):

        def matrix_from_pose(pose):
            """
                Converts a 2D pose to a 3x3 transformation matrix

                :param pose: 2D pose (x, y, yaw)
                :return: 3x3 transformation matrix
                """
            return np.array(
                [
                    [np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                    [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
                    [0, 0, 1],
                ]
            )

        def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]):
            """
            Converts a 3x3 transformation matrix to a 2D pose
            :param transform_matrix: 3x3 transformation matrix
            :return: 2D pose (x, y, yaw)
            """
            if transform_matrix.shape != (3, 3):
                raise RuntimeError(f"Expected a 3x3 transformation matrix, got {transform_matrix.shape}")

            heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

            return [transform_matrix[0, 2], transform_matrix[1, 2], heading]

        relative_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(relative_poses[i,:]) for i in range(relative_poses.shape[0])])
        origin_transform = matrix_from_pose(origin_pose)
        absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
        absolute_poses = [pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms]

        return absolute_poses

def main(args):
    node = gameformer_planner(args.model_path, args.device )
    node.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run test')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model on')
    args = parser.parse_args()

    main(args)

