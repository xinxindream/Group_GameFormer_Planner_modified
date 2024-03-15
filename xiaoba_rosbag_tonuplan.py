import numpy as np
import math
import argparse
import os
import sys
import pdb
import torch
from scipy.spatial.distance import cdist
from collections import deque

from xiaoba_rosbag_tonuplan_utils import *

os.environ['ROS_PACKAGE_PATH'] = '/home/xingchen24/catkin_ws/src:/opt/ros/noetic/share'
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rosbag
# from cv_bridge import CvBridge
# bridge = CvBridge()

class process_rosbag(object):
    def __init__(self, bag_path, save_path):
        self._bag_path = bag_path
        self._save_path = save_path
        self._num_agents = 20
        
        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK']  # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5}  # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 60, 'ROUTE_LANES': 60, 'CROSSWALK': 30}  # maximum number of points per feature to extract per feature layer.
        self._radius = 25  # [m] query radius scope relative to the current pose.
        self._sample_interval = 0.2
        self._interpolation_method = 'linear'
        
        self._past_time_horizon = 2  # [seconds]                    # 定义过去时间范围
        self._num_past_poses = 10 * self._past_time_horizon         # 定义过去姿态数量 = 20，帧率10，每秒10帧
        self._future_time_horizon = 5  # [seconds]                  # 定义未来时间范围
        self._num_future_poses = 10 * self._future_time_horizon     # 定义预测未来姿态数量 = 50
        self._drop_start_frames = 0                                 # 开始阶段丢弃帧数
        self._drop_end_frames = 1                                   # 结束阶段丢弃帧数 = 1 

        self._agents_past_queue = deque(maxlen=self._num_past_poses+1)          # 周围交通参与者历史姿态信息 => 21
        self._ego_past_queue = deque(maxlen=self._num_past_poses+1)             # 自车历史姿态信息 => 21            
        self._ego_past_time_queue = deque(maxlen=self._num_past_poses+1)        # 自车历史姿态信息对应时间戳  => 21
        self._agents_future_queue = deque(maxlen=self._num_future_poses+1)      # 周围交通参与者未来姿态信息  => 51                  
        self._ego_future_queue = deque(maxlen=self._num_future_poses)           # 自车未来姿态信息 => 50                                                                              

        # TODO：初始航道线的x，y坐标
        self._original_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/original_route_lane_data_x.npz"
        self._original_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/original_route_lane_data_y.npz"
        
        # 航道改变（变道？转弯？直线加速度改变？）后的x，y坐标？
        self._shift_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/shift_route_lane_data_x.npz"
        self._shift_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/shift_route_lane_data_y.npz"

    def save_to_disk(self, dir, data):
        save_agent_time = "{: .2f}".format(data['ego_time'])
        np.savez(f"{dir}/{data['datatime']}_{save_agent_time}.npz", **data)

    def distance_to_ego(self, obj_position, ego_position):
        return math.sqrt((obj_position.x - ego_position.x)**2 + (obj_position.y - ego_position.y)**2)

    def run(self):
        print(self._bag_path)
        # rosbag对象实例，可以访问bag包内容
        bag = rosbag.Bag(self._bag_path)

        original_route_lane_data_x = np.load(self._original_route_lane_data_x_path)
        original_route_lane_data_y = np.load(self._original_route_lane_data_y_path)
        shift_route_lane_data_x = np.load(self._shift_route_lane_data_x_path)
        shift_route_lane_data_y = np.load(self._shift_route_lane_data_y_path)

        # 获取坐标对（x, y），感觉可以直接axis=1
        # original_route_lane_data_x = [3,5,51,35,23,...]
        # original_route_lane_data_y = [3,5,51,35,23,...]
        original_route_lane_data_xy = np.stack((original_route_lane_data_x, original_route_lane_data_y)).transpose()
        shift_route_lane_data_xy = np.stack((shift_route_lane_data_x, shift_route_lane_data_y)).transpose()

        agents_list = []    # 保存周围交通参与者的信息（速度？加速度？姿态？）
        ego_poses_list = [] # 保存自车姿态信息，按时间来保存
        img_color_list = [] # 摄像机2D图像信息
        
        # TODO: 不太可以理解具体含义
        ego_time_list = []  # 交通参与者的时间？
        

        agent_topic = '/kxdun/perception/obstacles'
        ego_pose_topic = '/kxdun/ego_vehicle/localization'
        img_color_topic = '/zkhy_stereo/left/color'
        ego_pose_flag, agent_flag, img_flag = False, False, False
        last_ego_time = 0
        #读取消息并存储时间戳和信息
        # topic：主题
        # msg：消息
        # t: 时间戳
        for topic, msg, t in bag.read_messages(topics = [agent_topic, ego_pose_topic, img_color_topic]):
            if topic == agent_topic:
                agent_time = t.to_sec()
                agent_flag = True
                agent_msg = msg
            if topic == img_color_topic and agent_flag:
                img_time = t.to_sec()
                dt = img_time - agent_time
                if dt > 0.10: continue
                img_flag = True
                img_msg = msg
            if topic == ego_pose_topic and agent_flag:
                ego_pose_time = t.to_sec()
                dt = ego_pose_time - agent_time
                # 时间同步
                # dt > 0.1则跳出for循环，则表示要求时间差不能超过0.1，确保消息之间时间差接近，从而近似同步
                if dt > 0.10: continue
                ego_pose_flag = True
                ego_pose_msg = msg

            if ego_pose_flag and agent_flag and img_flag:  #and (abs(last_ego_time - ego_pose_time)>0.005)
                ego_pose_flag, agent_flag, img_flag = False, False, False
                last_ego_time = ego_pose_time
                ego_poses_list.append(ego_pose_msg)
                ego_time_list.append(agent_time)
                agents_list.append(agent_msg)
                img_color_list.append(img_msg)
        print('同步完成')

        # 模拟？除去前2s的交通参与者状态，预留2s的未来数据？
        for i in range(self._num_past_poses+self._drop_start_frames, len(agents_list) - self._num_future_poses - self._drop_end_frames):
            # 提取自车过去2s姿态信息以及对应时间戳
            for j in range(self._num_past_poses + 1):
                self._ego_past_queue.append(ego_poses_list[i - self._num_past_poses + j])
                self._ego_past_time_queue.append(ego_time_list[i - self._num_past_poses + j])
            # 提取自车未来2s姿态信息
            for j in range(self._num_future_poses):
                self._ego_future_queue.append(ego_poses_list[i + j + 1])
                
            for j in range(self._num_past_poses + 1):
                # 提取过去2s内，周围障碍物列表，包含障碍物位置信息
                obj_list = agents_list[i - self._num_past_poses + j].obstacle_list
                # 按照距离远近对物体进行重新排序，勾股定理
                obj_list.sort(key=lambda obj: self.distance_to_ego(obj.position, ego_poses_list[i].position),
                              reverse=False)
                # 更新障碍物列表，安装距离远近，近的下标小
                agents_list[i - self._num_past_poses + j].obstacle_list = obj_list
                # 最终获取交通参与者的历史姿态信息
                self._agents_past_queue.append(agents_list[i - self._num_past_poses + j])
                
            for j in range(self._num_future_poses + 1):
                # 按照距离远近对物体进行重新排序
                obj_list = agents_list[i + j + 1].obstacle_list
                obj_list.sort(key=lambda obj: self.distance_to_ego(obj.position, ego_poses_list[i].position),
                              reverse=False)
                agents_list[i + j + 1].obstacle_list = obj_list
                self._agents_future_queue.append(agents_list[i + j])

            ego_state = StateSE2(x=self._ego_past_queue[-1].position.x, y=self._ego_past_queue[-1].position.y,
                                 heading=self._ego_past_queue[-1].euler_angles.z)
            
            # cur_centroid = original_route_lane_data_xy[cur_ego_index]
            # 获取自车历史姿态信息的张量
            ego_agent_past = get_ego_past_to_tensor_list(self._ego_past_queue)
            
            # 获取交通参与者的历史姿态信息以及类型，但是按照了距离远近重新排序
            past_tracked_objects_tensor_list, neighbor_agents_types = get_tracked_objects_to_tensor_list(
                self._agents_past_queue)
            # time_stamps_past = get_past_timestamps_to_tensor(self._ego_past_queue)
            
            # 微秒时间戳
            time_stamps_past = torch.tensor([ego_time * 1e6 for ego_time in self._ego_past_time_queue], dtype=torch.int64)

            ego_agent_past, neighbor_agents_past, neighbor_indices = agent_past_process(ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list, neighbor_agents_types,self._num_agents)
            
            ego_agent_future = get_ego_future_to_tensor_list(ego_state, self._ego_future_queue)
            
            future_tracked_objects_tensor_list = get_tracked_future_objects_to_tensor_list(self._agents_future_queue)

            neighbor_agents_future = agent_future_process(self._ego_past_queue[-1], future_tracked_objects_tensor_list, self._num_agents, neighbor_indices)


            distances = cdist(original_route_lane_data_xy, np.array([ego_poses_list[i].position.x, ego_poses_list[i].position.y]).reshape(1, -1))
            cur_ego_index = np.argmin(distances)  # 找到距离最小的点的索引
            min_index = int(cur_ego_index - self._radius / self._sample_interval + 1)
            max_index = int(cur_ego_index + self._radius / self._sample_interval)

            if (min_index < 0) | (max_index > original_route_lane_data_xy.shape[0]-1): continue
            lanes_mid: List[List[Point2D]] = []
            baseline_path_polyline = [Point2D(original_route_lane_data_xy[min_index + node,0], original_route_lane_data_xy[min_index + node,1]) for node in range(int(self._radius / self._sample_interval * 2))]
            lanes_mid.append(baseline_path_polyline)
            baseline_path_polyline = [Point2D(shift_route_lane_data_xy[min_index + node, 0], shift_route_lane_data_xy[min_index + node, 1]) for node in range(int(self._radius / self._sample_interval * 2))]
            lanes_mid.append(baseline_path_polyline)
            coords_map_lanes_polylines = MapObjectPolylines(lanes_mid)
            coords_route_lanes_polylines = MapObjectPolylines(lanes_mid)
            crosswalk: List[List[Point2D]] = []
            coords_crosswalk_polylines = MapObjectPolylines(crosswalk)
            coords: Dict[str, MapObjectPolylines] = {}
            # extract generic map objects
            coords[self._map_features[0]] = coords_map_lanes_polylines
            coords[self._map_features[1]] = coords_route_lanes_polylines
            coords[self._map_features[2]] = coords_crosswalk_polylines
            traffic_light_encoding = np.zeros([2, 4], dtype=int)
            traffic_light_encoding[:, -1] = 1
            traffic_light_data_at_t: Dict[str, LaneSegmentTrafficLightData] = {}
            traffic_light_data: List[Dict[str, LaneSegmentTrafficLightData]] = []
            traffic_light_data_at_t[self._map_features[0]] = LaneSegmentTrafficLightData(
                list(map(tuple, traffic_light_encoding)))
            traffic_light_data = traffic_light_data_at_t
            vector_map = map_process(ego_state, coords, traffic_light_data, self._map_features,
                                     self._max_elements, self._max_points, self._interpolation_method)

            # gather data
            data = {"datatime": os.path.basename(self._bag_path)[:-10], "ego_time": ego_time_list[i], "ego_agent_past": ego_agent_past,
                    "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future}
            data.update(vector_map)
            # rgb = bridge.imgmsg_to_cv2(img_color_list[i], "8UC3")

            # # save to disk
            '''
            data:
                - datatime: 数据获取时间
                - ego_time: 
                - ego_agent_past: 自车历史姿态信息的张量
                - ego_agent_future: 自车未来状态相对于过去自身状态
                - neighbor_agents_past: 过去时刻，自车坐标系下agent状态
                - neighbor_agents_future: 未来时刻，自车坐标中心下的agent状态
            '''
            self.save_to_disk(self._save_path, data)
            # cv2.imwrite(f"{dir}/{data['datatime']}_{data['ego_time']}.jpg", rgb)


            # pdb.set_trace()
            # gather data
            # data = {"datatime": os.path.basename(self._bag_path), "ego_time": ego_time_list[i], "ego_agent_past": prev_ego_np,
            #         "ego_agent_future": future_ego_np,
            #         "neighbor_agents_past": prev_agents_np, "neighbor_agents_future": future_agents_np}
            # # get vector set map
            # vector_map = self.get_map(ego_poses_list[i])
            # data.update(vector_map)
            # # save to disk
            # self.save_to_disk(self._save_path, data)

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    node = process_rosbag(args.bag_path,args.save_dir)
    node.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run test')
    parser.add_argument('--bag_path', type=str, help='path to bag', default="/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.21/2023-12-21-11-38-53_with_det_part2.bag")
    parser.add_argument('--save_dir', type=str, help='path to save',
                        default="/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.21/2023-12-21-11-38-53_with_det_train/")
    args = parser.parse_args()

    main(args)

# lanes_np = np.zeros((self._max_elements['LANE'], self._max_points['LANE'], 7))
# lanes_np[:, :, -1] = 1  # traffic light unknown (0 0 0 1)
# crosswalks_np = np.zeros((self._max_elements['CROSSWALK'], self._max_points['CROSSWALK'], 3))
# route_lanes_np = np.zeros((self._max_elements['ROUTE_LANES'], self._max_points['ROUTE_LANES'], 3))
# prev_ego_np = np.zeros((self._num_past_poses + 1, 7))
# future_ego_np = np.zeros((self._num_future_poses, 7))
# prev_agents_np = np.zeros((self._num_agents, self._num_past_poses + 1, 11))
# future_agents_np = np.zeros((self._num_agents, self._num_future_poses, 11))
# for j in range(self._num_past_poses + 1):
#     prev_ego_np[self._num_past_poses - j, 0] = ego_poses_list[i - j].position.x
#     prev_ego_np[self._num_past_poses - j, 1] = ego_poses_list[i - j].position.y
#     prev_ego_np[self._num_past_poses - j, 2] = ego_poses_list[i - j].euler_angles.z
#     prev_ego_np[self._num_past_poses - j, 3] = ego_poses_list[i - j].linear_velocity.x
#     prev_ego_np[self._num_past_poses - j, 4] = ego_poses_list[i - j].linear_velocity.y
#     prev_ego_np[self._num_past_poses - j, 5] = ego_poses_list[i - j].linear_acceleration.x
#     prev_ego_np[self._num_past_poses - j, 6] = ego_poses_list[i - j].linear_acceleration.y
# for m in range(self._num_future_poses):
#     future_ego_np[m, 0] = ego_poses_list[i + m].position.x
#     future_ego_np[m, 1] = ego_poses_list[i + m].position.y
#     future_ego_np[m, 2] = ego_poses_list[i + m].euler_angles.z
#     future_ego_np[m, 3] = ego_poses_list[i + m].linear_velocity.x
#     future_ego_np[m, 4] = ego_poses_list[i + m].linear_velocity.y
#     future_ego_np[m, 5] = ego_poses_list[i + m].linear_acceleration.x
#     future_ego_np[m, 6] = ego_poses_list[i + m].linear_acceleration.y
#
# for n in range(self._num_past_poses + 1):
#     obj_positions = np.array([[obj.position.x, obj.position.y] for obj in agents_list[i - n].obstacle_list])
#     distances = cdist(obj_positions, np.array([0, 0]).reshape(1, -1))
#
#     obj_list = agents_list[i - n].obstacle_list
#     # obj_sorted_list = sorted(obj_list, key = self.distance_to_ego, reverse=False, lambda p: Point(p.position.x, p.position.y))
#     cur_ego = ego_poses_list[i].position
#     obj_list.sort(key=lambda obj: self.distance_to_ego(obj.position, ego_poses_list[i].position), reverse=False)
#     for agent_i, past_tracked_object in enumerate(agents_list[i - n].obstacle_list):
#         prev_agents_np[agent_i, self._num_past_poses - n, 0] = past_tracked_object.id
#     if past_tracked_object.velocity > 0.5:
#         prev_agents_np[agent_i, self._num_past_poses - n, 1] = past_tracked_object.velocity * math.sin(
#             past_tracked_object.vel_heading)
#         prev_agents_np[agent_i, self._num_past_poses - n, 2] = past_tracked_object.velocity * math.cos(
#             past_tracked_object.vel_heading)
#         prev_agents_np[agent_i, self._num_past_poses - n, 3] = past_tracked_object.vel_heading
#     else:
#         prev_agents_np[agent_i, self._num_past_poses - n, 1] = past_tracked_object.velocity * math.sin(
#             past_tracked_object.heading)  # 可能考虑直接为0
#         prev_agents_np[agent_i, self._num_past_poses - n, 2] = past_tracked_object.velocity * math.cos(
#             past_tracked_object.heading)
#         prev_agents_np[agent_i, self._num_past_poses - n, 3] = past_tracked_object.heading
#     prev_agents_np[agent_i, self._num_past_poses - n, 4] = past_tracked_object.width
#     prev_agents_np[agent_i, self._num_past_poses - n, 5] = past_tracked_object.length
#     prev_agents_np[agent_i, self._num_past_poses - n, 6] = past_tracked_object.position.x
#     prev_agents_np[agent_i, self._num_past_poses - n, 7] = past_tracked_object.position.y
#     # VEHICLE:[1, 0, 0] PEDESTRIAN:[0, 1, 0] other:[0, 0, 1
#     prev_agents_np[agent_i, self._num_past_poses - n, 8] = 1
#     prev_agents_np[agent_i, self._num_past_poses - n, 9] = 0
#     prev_agents_np[agent_i, self._num_past_poses - n, 10] = 0
#
# for k in range(self._num_future_poses):
#     for agent_i, past_tracked_object in enumerate(agents_list[i + k].obstacle_list):
#         future_agents_np[agent_i, k, 0] = past_tracked_object.id
#     if past_tracked_object.velocity > 0.5:
#         future_agents_np[agent_i, k, 1] = past_tracked_object.velocity * math.sin(past_tracked_object.vel_heading)
#         future_agents_np[agent_i, k, 2] = past_tracked_object.velocity * math.cos(past_tracked_object.vel_heading)
#         future_agents_np[agent_i, k, 3] = past_tracked_object.vel_heading
#     else:
#         future_agents_np[agent_i, k, 1] = past_tracked_object.velocity * math.sin(
#             past_tracked_object.heading)  # 可能考虑直接为0
#         future_agents_np[agent_i, k, 2] = past_tracked_object.velocity * math.cos(past_tracked_object.heading)
#         future_agents_np[agent_i, k, 3] = past_tracked_object.heading
#     future_agents_np[agent_i, k, 4] = past_tracked_object.width
#     future_agents_np[agent_i, k, 5] = past_tracked_object.length
#     future_agents_np[agent_i, k, 6] = past_tracked_object.position.x
#     future_agents_np[agent_i, k, 7] = past_tracked_object.position.y
#     # VEHICLE:[1, 0, 0] PEDESTRIAN:[0, 1, 0] other:[0, 0, 1
#     future_agents_np[agent_i, k, 8] = 1
#     future_agents_np[agent_i, k, 9] = 0
#     future_agents_np[agent_i, k, 10] = 0
