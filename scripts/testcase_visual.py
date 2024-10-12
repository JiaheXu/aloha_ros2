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
np.set_printoptions(suppress=True,precision=4)
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

from math_tools import *
from utils import *

import matplotlib.pyplot as plt

def get_transform( trans_7D):
    trans = trans_7D[0:3]
    quat = trans_7D[3:7]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

def get_cube_corners( bound_box ):
    corners = []
    corners.append( [ bound_box[0][0], bound_box[1][0], bound_box[2][0] ])
    corners.append( [ bound_box[0][0], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][0], bound_box[2][0] ])

    return corners

def visualize_pcd(pcd, traj_lists = None, curr_pose = None):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.05, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    # if(use_arrow):
    #     mesh = o3d.geometry.TriangleMesh.create_arrow( cylinder_radius=0.01, cone_radius=0.01, cylinder_height=0.005, cone_height=0.01, resolution=20, cylinder_split=4, cone_split=1 )
    
    if(traj_lists is not None):
        for traj in traj_lists:
            for point in traj:
                new_mesh = copy.deepcopy(mesh).transform(point)
                vis.add_geometry(new_mesh)



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


def main():
    
    # data = np.load("./2arms_open_pen/1.npy", allow_pickle = True)
    task = "" 
    # data_idxs = [1, 4, 31, 32, 33, 34, 35]
    # data_idxs =  [1, 4, 31, 32, 33, 34, 35]
    # idx =  2
    file_dir = "test4"
    ep_idx = 0
    print("idx: ", ep_idx)
    sample = np.load("./{}/step_{}.npy".format(file_dir, ep_idx) , allow_pickle = True)
    sample = sample.item()
    # print("sample: ", sample)
    # print(sample.item())
    rgb = sample["rgb"]
    xyz = sample["xyz"]
    action = sample['action']
    curr_gripper = sample['curr_gripper'][0,0]
    
    pcd_rgb = rgb.reshape(-1, 3)/255.0
    pcd_xyz = xyz.reshape(-1, 3)
    

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
    pcd.points = o3d.utility.Vector3dVector( pcd_xyz )

    left = []
    right = []
    goals = []
    # smooth_traj = []
    
    curr_pose = []
    last_moment = None

    visualize_pcd(pcd, [left, goals], curr_pose)


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
