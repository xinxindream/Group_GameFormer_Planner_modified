import time

import numpy as np
import math
import argparse
import os
import sys
import pdb
import torch
from scipy.spatial.distance import cdist
from collections import deque

import rosbag
import rospy
from message_filters import TimeSynchronizer, Subscriber
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, Vector3
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from kxdun_localization_msgs.msg import Localization
from kxdun_perception_msgs.msg import PerceptionObstacle, PerceptionObstacleArray, PerceptionLaneArray, PerceptionLane
from sensor_msgs.msg import Image

from kxdun_control_msgs.msg import PlanningADCTrajectory, TrajectoryPoint

from run_nuplan_ros_utils import *

class gameformer_planner:
    def __init__(self, bag_path):
        self._bag_path = bag_path
        self._bag_path = bag_path
        self._num_agents = 20
        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK']  # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5}  # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 60, 'ROUTE_LANES': 60, 'CROSSWALK': 30}  # maximum number of points per feature to extract per feature layer.
        self._radius = 25  # [m] query radius scope relative to the current pose.
        self._sample_interval = 0.2
        self._interpolation_method = 'linear'
        self._past_time_horizon = 2  # [seconds]
        self._num_past_poses = 10 * self._past_time_horizon
        self._future_time_horizon = 5  # [seconds]
        self._num_future_poses = 10 * self._future_time_horizon
        self._drop_start_frames = 0
        self._drop_end_frames = 1

        self._frame_id = 'map'

        self._perception_flag = False
        self._ego_flag = False
        self._lanes_flag = False
        self._routes_flag = False
        self._loc_connt = 0

        self._agents_past_queue = deque(maxlen = 22)
        self._ego_past_queue = deque(maxlen = 22)
        self._ego_future_queue = deque(maxlen=self._num_future_poses)

        self._marker_msg = None
        self._img_msg = None
        self._ego_pose_msg = None

        self._per_flag = False
        self._loc_flag = False
        self._img_flag = False

        # self._kxdun_perception_sub = rospy.Subscriber('/kxdun/perception/obstacles', PerceptionObstacleArray, self.perception_callback)
        # self._kxdun_localization_sub = rospy.Subscriber('/kxdun/ego_vehicle/localization', Localization, self.localization_callback)
        # self._kxdun_img_sub = rospy.Subscriber('/zkhy_stereo/left/color', Image, self.img_callback)

        self._original_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/original_route_lane_data_x.npz"
        self._original_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/original_route_lane_data_y.npz"
        self._shift_route_lane_data_x_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/shift_route_lane_data_x.npz"
        self._shift_route_lane_data_y_path = "/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2023.12.14/route_lane/shift_route_lane_data_y.npz"

        self._ego_markers_pub = rospy.Publisher('/ego_car_markers', MarkerArray, queue_size=10)
        self._obj_markers_pub = rospy.Publisher('/objects_perception_markers', MarkerArray, queue_size=10)
        self._img_pub = rospy.Publisher('/cur_img_pub', Image, queue_size=10)
        self._centerline_path_pub_0 = rospy.Publisher('/centerline_path_pub_0', Path, queue_size=10)
        self._centerline_path_pub_1 = rospy.Publisher('/centerline_path_pub_1', Path, queue_size=10)
        self._planner_path_pub = rospy.Publisher('/learn_based_planner_path_pub', Path, queue_size=10)
        self._future_gt_path_pub = rospy.Publisher('/future_gt_path_pub', Path, queue_size=10)
        self._border3dpts_bev0_pub = rospy.Publisher('/border3dpts_bev0', Path, queue_size=10)
        self._border3dpts_bev1_pub = rospy.Publisher('/border3dpts_bev1', Path, queue_size=10)
        self._border3dpts_bev2_pub = rospy.Publisher('/border3dpts_bev2', Path, queue_size=10)
        self._border3dpts_bev3_pub = rospy.Publisher('/border3dpts_bev3', Path, queue_size=10)

        original_route_lane_data_x = np.load(self._original_route_lane_data_x_path)
        original_route_lane_data_y = np.load(self._original_route_lane_data_y_path)
        shift_route_lane_data_x = np.load(self._shift_route_lane_data_x_path)
        shift_route_lane_data_y = np.load(self._shift_route_lane_data_y_path)
        original_route_lane_data_xy = np.stack((original_route_lane_data_x, original_route_lane_data_y)).transpose()
        shift_route_lane_data_xy = np.stack((shift_route_lane_data_x, shift_route_lane_data_y)).transpose()

        original_route_lane_data_x = np.load(self._original_route_lane_data_x_path)
        original_route_lane_data_y = np.load(self._original_route_lane_data_y_path)
        shift_route_lane_data_x = np.load(self._shift_route_lane_data_x_path)
        shift_route_lane_data_y = np.load(self._shift_route_lane_data_y_path)

        self._original_route_lane_data_xy = np.stack((original_route_lane_data_x, original_route_lane_data_y)).transpose()
        self._shift_route_lane_data_xy = np.stack((shift_route_lane_data_x, shift_route_lane_data_y)).transpose()

        self._path0 = Path()
        self._path1 = Path()
        self._future_path = Path()

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

    def run(self):
        print(self._bag_path)
        bag = rosbag.Bag(self._bag_path)

        agents_list = []
        ego_poses_list = []
        ego_time_list = []
        img_color_list = []
        markers_list  =[]
        stamp_list = []
        border3dpts_bev0_list = []
        border3dpts_bev1_list = []
        border3dpts_bev2_list = []
        border3dpts_bev3_list = []

        agent_topic = '/kxdun/perception/obstacles'
        ego_pose_topic = '/kxdun/ego_vehicle/localization'
        img_color_topic = '/zkhy_stereo/left/color'
        markers_topic = '/objects_markers'
        border3dpts_bev0_topic = '/border3dpts_bev0'
        border3dpts_bev1_topic = '/border3dpts_bev1'
        border3dpts_bev2_topic = '/border3dpts_bev2'
        border3dpts_bev3_topic = '/border3dpts_bev3'
        ego_pose_flag, agent_flag, img_flag = False, False, False

        for i in range(self._original_route_lane_data_xy.shape[0]):
            pose = PoseStamped()
            pose.pose.position.x = self._original_route_lane_data_xy[i,0]
            pose.pose.position.y = self._original_route_lane_data_xy[i,1]
            pose.pose.position.z = 40
            pose.pose.orientation.w = 1.0
            self._path0.poses.append(pose)

            pose = PoseStamped()
            pose.pose.position.x = self._shift_route_lane_data_xy[i, 0]
            pose.pose.position.y = self._shift_route_lane_data_xy[i, 1]
            pose.pose.position.z = 40
            pose.pose.orientation.w = 1.0
            self._path1.poses.append(pose)

        border3dpts_bev0_msg = None
        border3dpts_bev1_msg = None
        border3dpts_bev2_msg = None
        border3dpts_bev3_msg = None

        for topic, msg, t in bag.read_messages(topics = [agent_topic, ego_pose_topic, img_color_topic, markers_topic, border3dpts_bev0_topic,border3dpts_bev1_topic,border3dpts_bev2_topic,border3dpts_bev3_topic]):
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

            if topic == markers_topic:
                marker_msg = msg

            if topic == border3dpts_bev0_topic:
                border3dpts_bev0_msg = msg
            if topic == border3dpts_bev1_topic:
                border3dpts_bev1_msg = msg
            if topic == border3dpts_bev2_topic:
                border3dpts_bev2_msg = msg
            if topic == border3dpts_bev3_topic:
                border3dpts_bev3_msg = msg

            if ego_pose_flag and agent_flag and img_flag:  #and (abs(last_ego_time - ego_pose_time)>0.005)
                ego_pose_flag, agent_flag, img_flag = False, False, False
                ego_poses_list.append(ego_pose_msg)
                ego_time_list.append(agent_time)
                agents_list.append(agent_msg)
                img_color_list.append(img_msg)
                markers_list.append(marker_msg)
                stamp_list.append(t)
                if (border3dpts_bev0_msg is not None)&(border3dpts_bev1_msg is not None)&(border3dpts_bev2_msg is not None)&(border3dpts_bev3_msg is not None):
                    border3dpts_bev0_list.append(border3dpts_bev0_msg)
                    border3dpts_bev1_list.append(border3dpts_bev1_msg)
                    border3dpts_bev2_list.append(border3dpts_bev2_msg)
                    border3dpts_bev3_list.append(border3dpts_bev3_msg)
        print('同步完成')
        print(len(ego_poses_list))

        for count_i in range(len(ego_poses_list)-self._num_future_poses-1):
            cur_header = Header()
            cur_header.stamp = stamp_list[count_i]
            cur_header.frame_id = "map"
            time.sleep(0.1)
            # print(ego_pose_flag,agent_flag, img_flag)

            self._future_path = Path()
            for j in range(self._num_future_poses):
                pose = PoseStamped()
                pose.pose.position.x = ego_poses_list[count_i + j + 1].position.x
                pose.pose.position.y = ego_poses_list[count_i + j + 1].position.y
                pose.pose.position.z = ego_poses_list[count_i + j + 1].position.z
                pose.pose.orientation.x = ego_poses_list[count_i + j + 1].orientation.qx
                pose.pose.orientation.y = ego_poses_list[count_i + j + 1].orientation.qy
                pose.pose.orientation.z = ego_poses_list[count_i + j + 1].orientation.qz
                pose.pose.orientation.w = ego_poses_list[count_i + j + 1].orientation.qw
                self._future_path.poses.append(pose)

            pose = Pose()
            pose.position.x = ego_poses_list[count_i].position.x
            pose.position.y = ego_poses_list[count_i].position.y
            pose.position.z = ego_poses_list[count_i].position.z
            pose.orientation.x = ego_poses_list[count_i].orientation.qx
            pose.orientation.y = ego_poses_list[count_i].orientation.qy
            pose.orientation.z = ego_poses_list[count_i].orientation.qz
            pose.orientation.w = ego_poses_list[count_i].orientation.qw
            markers = MarkerArray()
            marker = Marker(header=cur_header)
            marker.ns = 'mesh'
            marker.id = 0
            marker.type = Marker.MESH_RESOURCE
            marker.action = marker.ADD
            marker.mesh_resource = "file:///media/xingchen24/xingchen4T/datasets/Luce/3Dbox-test/models/car.dae"
            marker.pose = pose
            marker.scale = Vector3(1.0, 1.0, 1.0)
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
            markers.markers.append(marker)

            markers_msg = markers_list[count_i]
            # markers_msg = markers_list[count_i + 30]
            if markers_msg is not None:
                for marker_msg in markers_msg.markers:
                    marker_msg.header = cur_header
                self._obj_markers_pub.publish(markers_msg)
            img_msg = img_color_list[count_i]
            # img_msg = img_color_list[count_i + 30]
            img_msg.header = cur_header

            border3dpts_bev0_path = border3dpts_bev0_list[count_i]
            border3dpts_bev1_path = border3dpts_bev1_list[count_i]
            border3dpts_bev2_path = border3dpts_bev2_list[count_i]
            border3dpts_bev3_path = border3dpts_bev3_list[count_i]

            self._path0.header = cur_header
            self._path1.header = cur_header
            self._future_path.header = cur_header
            border3dpts_bev0_path.header = cur_header
            border3dpts_bev1_path.header = cur_header
            border3dpts_bev2_path.header = cur_header
            border3dpts_bev3_path.header = cur_header

            self._ego_markers_pub.publish(markers)
            self._img_pub.publish(img_msg)
            self._centerline_path_pub_0.publish(self._path0)
            self._centerline_path_pub_1.publish(self._path1)
            self._future_gt_path_pub.publish(self._future_path)
            self._border3dpts_bev0_pub.publish(border3dpts_bev0_path)
            self._border3dpts_bev1_pub.publish(border3dpts_bev1_path)
            self._border3dpts_bev2_pub.publish(border3dpts_bev2_path)
            self._border3dpts_bev3_pub.publish(border3dpts_bev3_path)
        print('结束')

        # rospy.spin()

def main(args):
    rospy.init_node('xiaoba_rosbag_test')
    node = gameformer_planner(args.bag_path)
    node.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run xiaoba rosbag test')
    parser.add_argument('--bag_path', type=str, help='path to bag',
                        default="/media/xingchen24/xingchen/datasets/learn_based_planner/xiaoba/2024.1.11/2024-01-11-17-20-37_part1_with_det_2.bag")
    args = parser.parse_args()

    main(args)