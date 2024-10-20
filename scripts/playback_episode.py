"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np
from PIL import Image as im 
import os
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
# from matplotlib import pyplot as plt
import copy
import time


from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped
# import rospy
# import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
import torch
np.set_printoptions(suppress=True,precision=4)
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32, Header, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Joy, PointCloud2, PointField
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Path

def get_transform2( trans_7D):
    trans = trans_7D[0:3]
    quat = trans_7D[3:7]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

def get_transform( trans, quat):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)

        self.idx = 0
        self.bimanual_ee_publisher = self.create_publisher(Float32MultiArray, "bimanual_ee_cmd", 1)
        self.left_hand_ee_publisher = self.create_publisher(Path, "left_ee_goal", 1)
        self.right_hand_ee_publisher = self.create_publisher(Path, "right_ee_goal", 1)
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def print_action(self, action):
        action = action.reshape(-1, 2, 8)
        left_path_np = action[:, 0, :]
        right_path_np = action[:, 1, :]
        start = time.time()

        header = Header()
        header.frame_id = "world"
        ros_time = self.get_clock().now()
        header.stamp = ros_time.to_msg()

        left_path = Path()
        left_path.header = header
        for i in range(action.shape[0]):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(action[i,0,0])
            pose.pose.position.y = float(action[i,0,1])
            pose.pose.position.z = float(action[i,0,2])
            pose.pose.orientation.x = float(action[i,0,3])
            pose.pose.orientation.y = float(action[i,0,4])
            pose.pose.orientation.z = float(action[i,0,5])
            pose.pose.orientation.w = float(action[i,0,6])
            left_path.poses.append(pose)

        right_path = Path()
        right_path.header = header
        for i in range(action.shape[0]):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(action[i,1,0])
            pose.pose.position.y = float(action[i,1,1])
            pose.pose.position.z = float(action[i,1,2])
            pose.pose.orientation.x = float(action[i,1,3])
            pose.pose.orientation.y = float(action[i,1,4])
            pose.pose.orientation.z = float(action[i,1,5])
            pose.pose.orientation.w = float(action[i,1,6])
            right_path.poses.append(pose)

        end = time.time()
        self.left_hand_ee_publisher.publish(left_path)
        self.right_hand_ee_publisher.publish(right_path)
        print( "took: ", end - start)

    def timer_callback(self):
        print("idx: ", self.idx)
        file_dir = "test"
        sample = np.load("./{}/step_{}.npy".format(file_dir, self.idx) , allow_pickle = True)
        sample = sample.item()
        action = sample['action']
        self.print_action(action)
        array_msg = Float32MultiArray()
        
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim.append(MultiArrayDimension())

        array_msg.layout.dim[0].label = "steps"
        array_msg.layout.dim[1].label = "hands"
        array_msg.layout.dim[2].label = "pose"

        array_msg.layout.dim[0].size = action.shape[0]
        array_msg.layout.dim[1].size = action.shape[1]
        array_msg.layout.dim[1].size = action.shape[2]
        array_msg.layout.data_offset = 0

        array_msg.data = action.reshape([1, -1])[0].tolist();
        # array_msg.layout.dim[0].stride = width*height
        # array_msg.layout.dim[1].stride = width
        # self.low_level_finsihed = False
        self.bimanual_ee_publisher.publish(array_msg)
        self.idx += 1

def main():
    rclpy.init()

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (1, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (1, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 8)
    # List of tensors
