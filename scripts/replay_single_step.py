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

# import rospy
# import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
import torch
from numpy import linalg as LA
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32, Header, Float32MultiArray, MultiArrayDimension


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

def visualize_pcd(pcd, lefts = None, rights = None, curr_pose = None):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    if(lefts is not None):
        for left in lefts:
            left_mesh = copy.deepcopy(mesh).transform(left)
            vis.add_geometry(left_mesh)

    if(rights is not None):
        for right in rights:
            right_mesh = copy.deepcopy(mesh).transform(right)
            vis.add_geometry(right_mesh)

    if(curr_pose is not None):
        for pose in curr_pose:
            curr_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            curr_mesh.scale(0.2, center=(0., 0., 0.) )
            curr_mesh = curr_mesh.transform(pose)
            vis.add_geometry(curr_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()




class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        task = "" 
        # data_idxs = [1, 4, 31, 32, 33, 34, 35]
        # data_idxs =  [1, 4, 31, 32, 33, 34, 35]
        # idx =  2
        file_dir = "case5"
        length = 25
        # self.data = None
        idx = 0
        self.sample = np.load("./{}/step_{}.npy".format(file_dir, idx) , allow_pickle = True).item()

        self.bimanual_ee_publisher = self.create_publisher(Float32MultiArray, "bimanual_ee_cmd", 1)

        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        action = self.sample['action']
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

        self.bimanual_ee_publisher.publish(array_msg)
        print("sent !!!!!")
        print("sent !!!!!") 
        print("sent !!!!!")

def main(args=None):
    rclpy.init(args=args)

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
