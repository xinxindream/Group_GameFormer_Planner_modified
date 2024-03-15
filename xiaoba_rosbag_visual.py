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

import rosbag
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, Vector3
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from kxdun_localization_msgs.msg import Localization
from kxdun_perception_msgs.msg import PerceptionObstacle, PerceptionObstacleArray, PerceptionLaneArray, PerceptionLane
from sensor_msgs.msg import Image

class process_rosbag:
    def __init__(self):
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

        self._agents_past_queue = deque(maxlen=self._num_past_poses+1)
        self._ego_past_queue = deque(maxlen=self._num_past_poses+1)
        self._ego_past_time_queue = deque(maxlen=self._num_past_poses+1)
        self._agents_future_queue = deque(maxlen=self._num_future_poses+1)
        self._ego_future_queue = deque(maxlen=self._num_future_poses)

        self._marker_msg = None
        self._img_msg = None
        self._ego_pose_msg = None

        self._kxdun_localization_sub = rospy.Subscriber('/kxdun/ego_vehicle/localization', Localization, self.localization_callback)
        self._kxdun_perception_sub = rospy.Subscriber('/objects_markers', MarkerArray, self.marker_callback)
        self._kxdun_img_sub = rospy.Subscriber('/zkhy_stereo/left/color', Image, self.img_callback)

        self._original_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/original_route_lane_data_x.npz"
        self._original_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/original_route_lane_data_y.npz"
        self._shift_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/shift_route_lane_data_x.npz"
        self._shift_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/lane_center_line/shift_route_lane_data_y.npz"

        self._ego_markers_pub = rospy.Publisher('/ego_car_markers', MarkerArray, queue_size=10)
        self._obj_markers_pub = rospy.Publisher('/objects_perception_markers', MarkerArray, queue_size=10)
        self._img_pub = rospy.Publisher('/cur_img_pub', Image, queue_size=10)
        self._centerline_path_pub_0 = rospy.Publisher('/centerline_path_pub_0', Path, queue_size=10)
        self._centerline_path_pub_1 = rospy.Publisher('/centerline_path_pub_1', Path, queue_size=10)

        original_route_lane_data_x = np.load(self._original_route_lane_data_x_path)
        original_route_lane_data_y = np.load(self._original_route_lane_data_y_path)
        shift_route_lane_data_x = np.load(self._shift_route_lane_data_x_path)
        shift_route_lane_data_y = np.load(self._shift_route_lane_data_y_path)
        original_route_lane_data_xy = np.stack((original_route_lane_data_x, original_route_lane_data_y)).transpose()
        shift_route_lane_data_xy = np.stack((shift_route_lane_data_x, shift_route_lane_data_y)).transpose()

        self._path0 = Path()
        self._path1 = Path()
        for i in range(original_route_lane_data_xy.shape[0]):
            pose = PoseStamped()
            pose.pose.position.x = original_route_lane_data_xy[i,0]
            pose.pose.position.y = original_route_lane_data_xy[i,1]
            pose.pose.position.z = 40
            pose.pose.orientation.w = 1.0
            self._path0.poses.append(pose)

            pose = PoseStamped()
            pose.pose.position.x = shift_route_lane_data_xy[i, 0]
            pose.pose.position.y = shift_route_lane_data_xy[i, 1]
            pose.pose.position.z = 40
            pose.pose.orientation.w = 1.0
            # if (shift_route_lane_data_xy[i, 0]!=0):
            self._path1.poses.append(pose)

    def marker_callback(self, msg):
        self._marker_msg = msg
        cur_header = self._img_msg.header
        # cur_header = Header()
        # cur_header.stamp = rospy.Time.now()
        cur_header.frame_id = "map"

        pose = Pose()
        pose.position.x = self._ego_pose_msg.position.x
        pose.position.y = self._ego_pose_msg.position.y
        pose.position.z = self._ego_pose_msg.position.z
        pose.orientation.x = self._ego_pose_msg.orientation.qx
        pose.orientation.y = self._ego_pose_msg.orientation.qy
        pose.orientation.z = self._ego_pose_msg.orientation.qz
        pose.orientation.w = self._ego_pose_msg.orientation.qw
        markers = MarkerArray()
        marker = Marker(header=cur_header)
        marker.ns = 'mesh'
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = marker.ADD
        marker.mesh_resource = "file:///media/xingchen24/xingchen4T/datasets/Luce/3Dbox-test/models/car.dae"
        marker.pose = pose
        marker.scale = Vector3(1.0, 1.0, 1.0)
        # marker.color = ColorRGBA(color_mask[0][0]/256.0,color_mask[0][1]/256.0,color_mask[0][2]/256.0,color_mask[0][3]/256.0)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        markers.markers.append(marker)

        for marker_msg in self._marker_msg.markers:
            marker_msg.header = cur_header
        self._img_msg.header = cur_header

        self._path0.header = cur_header
        self._path1.header = cur_header

        self._ego_markers_pub.publish(markers)
        self._obj_markers_pub.publish(self._marker_msg)
        self._img_pub.publish(self._img_msg)
        self._centerline_path_pub_0.publish(self._path0)
        self._centerline_path_pub_1.publish(self._path1)
        print('pub msg')

    def img_callback(self, msg):
        self._img_msg = msg

    def localization_callback(self, msg):
        self._ego_pose_msg = msg
        # print('localization_callback')

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node('xiaoba_rosbag_visual')
    node = process_rosbag()
    node.run()

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
