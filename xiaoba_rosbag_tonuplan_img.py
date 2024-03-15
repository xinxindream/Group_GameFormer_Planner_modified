import numpy as np
import math
import argparse
import os
import sys
import pdb
import cv2

import rosbag
from cv_bridge import CvBridge
bridge = CvBridge()

class process_rosbag(object):
    def __init__(self, bag_path, save_path):
        self._bag_path = bag_path
        self._save_path = save_path
        self._num_agents = 20
        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK']  # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5}  # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 60, 'ROUTE_LANES': 60, 'CROSSWALK': 30}  # maximum number of points per feature to extract per feature layer.
        self._radius = 20  # [m] query radius scope relative to the current pose.
        self._sample_interval = 0.2
        self._interpolation_method = 'linear'
        self._past_time_horizon = 2  # [seconds]
        self._num_past_poses = 10 * self._past_time_horizon
        self._future_time_horizon = 5  # [seconds]
        self._num_future_poses = 10 * self._future_time_horizon
        self._drop_start_frames = 0
        self._drop_end_frames = 0

        self._original_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/original_route_lane_data_x.npz"
        self._original_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/original_route_lane_data_y.npz"
        self._shift_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/shift_route_lane_data_x.npz"
        self._shift_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/shift_route_lane_data_y.npz"

    def run(self):
        print(self._bag_path)
        bag = rosbag.Bag(self._bag_path)

        agents_list = []
        ego_poses_list = []
        ego_time_list = []
        img_color_list = []

        agent_topic = '/kxdun/perception/obstacles'
        ego_pose_topic = '/kxdun/ego_vehicle/localization'
        img_color_topic = '/zkhy_stereo/left/color'
        ego_pose_flag, agent_flag, img_flag = False, False, False
        #读取消息并存储时间戳和信息
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
                if dt > 0.10: continue
                ego_pose_flag = True
                ego_pose_msg = msg
            # print(ego_pose_flag , agent_flag , img_flag)
            if ego_pose_flag and agent_flag and img_flag:  #and (abs(last_ego_time - ego_pose_time)>0.005)
                ego_pose_flag, agent_flag, img_flag = False, False, False
                ego_poses_list.append(ego_pose_msg)
                ego_time_list.append(agent_time)
                agents_list.append(agent_msg)
                img_color_list.append(img_msg)
                rgb = bridge.imgmsg_to_cv2(img_msg, "8UC3")
                save_agent_time = "{: .2f}".format(agent_time)
                cv2.imwrite(f"{self._save_path}/{os.path.basename(self._bag_path)[:-4]}_{save_agent_time}.jpg", rgb)

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
    parser.add_argument('--bag_path', type=str, help='path to bag', default="/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2024.1.10/2024-01-10-16-20-00_part1.bag")
    parser.add_argument('--save_dir', type=str, help='path to save',
                        default="/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2024.1.10/2024-01-10-16-20-00_train_likenuplan/")
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
