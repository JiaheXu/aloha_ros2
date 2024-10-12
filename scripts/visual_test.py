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

import time
np.set_printoptions(suppress=True,precision=4)
from utils import *
from math_tools import *




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

def visualize_pcd_delta_transform(pcd, start_t, delta_transforms, object_pcd = None):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0., 0., 0.) )
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.))

    new_mesh = copy.deepcopy(mesh).transform( get_transform(start_t[0:3], start_t[3:7]) )
    vis.add_geometry(new_mesh)
    last_trans = get_transform( start_t[0:3], start_t[3:7] )
    new_object_pcd = copy.deepcopy(object_pcd).transform(last_trans)
    vis.add_geometry(new_object_pcd)

    for delta_t in delta_transforms:
        last_trans = get_transform( delta_t[0:3], delta_t[3:7] ) @ get_transform(start_t[0:3], start_t[3:7])

        new_object_pcd = copy.deepcopy(object_pcd).transform(last_trans)
        vis.add_geometry(new_object_pcd)

        new_mesh = copy.deepcopy(mesh).transform(last_trans)
        vis.add_geometry(new_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()
    return last_trans



def main():
    
    # data = np.load("./2arms_open_pen/1.npy", allow_pickle = True)
    task = "" 
    # data_idxs = [1, 4, 31, 32, 33, 34, 35]
    # data_idxs =  [1, 4, 31, 32, 33, 34, 35]
    # idx =  2
    file_dir = "case5"
    length = 25
    #length = 1
    for idx in range(length):
    # for idx in range(2,3):
        print("idx: ", idx)
        sample = np.load("./{}/step_{}.npy".format(file_dir, idx) , allow_pickle = True)
        # print("sample: ", sample)
        # print(sample.item())
        sample = sample.item()
        rgb = sample["rgb"]
        xyz = sample["xyz"]
        action = sample['action']
        curr_gripper = sample['curr_gripper'][0,0]
        
        pcd_rgb = rgb.reshape(-1, 3)/255.0
        pcd_xyz = xyz.reshape(-1, 3)
        

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
        pcd.points = o3d.utility.Vector3dVector( pcd_xyz )
        # visualize_pcd(pcd)
        right = []
        left = []
        curr_pose = []
        
        start = time.time()
        
        # curr_pose.append( get_transform2(curr_gripper[0,0:7]) )
        # curr_pose.append( get_transform2(curr_gripper[1,0:7]) )

        trans = FwdKin(sample["left_joints"])
        trans[1,3] += 0.315
        curr_pose.append( copy.deepcopy(trans) )

        trans = FwdKin(sample["right_joints"])
        trans[1,3] -= 0.315
        curr_pose.append( copy.deepcopy(trans) )

        # print("action: ", action.shape)
        trajectory = action
        print("trajectory: ", trajectory.shape)
        # left.append( get_transform2( trajectory[-1,0,0:7]))
        # right.append( get_transform2( trajectory[-1,1,0:7]))

        # print("last goal: ", get_transform2( trajectory[-1,0,0:7]))
        print("last goal: ", get_transform2( trajectory[-1,0,0:7]))
        left_stack = [curr_gripper[0,0:8]]
        right_stack = [curr_gripper[1,0:8]]
        dist_threshold = 0.01

        for action_idx in range(trajectory.shape[0]):
            dist = LA.norm( trajectory[action_idx,0,0:3] - left_stack[-1][0:3] )
            if(dist < dist_threshold):
                continue
            left_stack.append(trajectory[action_idx,0,0:8])
            
        
        for action_idx in range(trajectory.shape[0]):
            dist = LA.norm( trajectory[action_idx,1,0:3] - right_stack[-1][0:3] )
            if(dist < dist_threshold):
                continue
            right_stack.append(trajectory[action_idx,1,0:8])
            
        left_action = copy.deepcopy( left_stack )
        left_stack = traj_interpolation(left_stack)
        right_stack = traj_interpolation(right_stack)
        
        left_mid_point = get_mid_point(left_stack[:,0:3])
        right_mid_point = get_mid_point(right_stack[:,0:3])
        
        # left = [ get_transform2(left_stack[0]), get_transform2(left_stack[left_mid_point]), get_transform2(left_stack[-1]) ]
        # right = [ get_transform2(right_stack[0]), get_transform2(right_stack[right_mid_point]), get_transform2(right_stack[-1]) ]


        # current_pose = [sample["left_joints"], sample["right_joints"]]
        # current_pose = [ left_stack[0], right_stack[0] ]


        mid_goals = [ left_stack[left_mid_point], right_stack[right_mid_point] ]
        goals = [left_stack[-1], right_stack[-1]]


        left_hand = get_transform2(left_stack[0])
        left_hand[1,3] -= 0.315
        right_hand = get_transform2(right_stack[0])
        right_hand[1,3] += 0.315

        left_ik, err, success_left = RRcontrol( left_hand, sample["left_joints"][0:6], debug=False )
        right_ik, err, success_right = RRcontrol( right_hand, sample["right_joints"][0:6], debug=False )
        left_ik = np.concatenate( [left_ik, np.array([0]) ] )
        right_ik = np.concatenate( [right_ik, np.array([0]) ] )
        current_joints = [left_ik, right_ik]
        
        left_traj, right_traj = get_trajectory( current_joints, mid_goals, goals)

        end = time.time()
        print("time cost: ", end - start)

        # trans = FwdKin(sample["left_joints"])
        # trans[1,3] += 0.315
        # print("FWK: ",  trans)
        # print("curr_gripper: ", get_transform2(curr_gripper[0,0:7]))
        # print()
        # curr_gripper = [get_transform2(curr_gripper[0,0:7]), get_transform2(curr_gripper[1,0:7])]
        curr_gripper = []
        for idx in range(left_traj.shape[0]):
            trans = FwdKin( left_traj[idx, 0:6] )
            trans[1,3] += 0.215
            left.append( trans )

        for idx in range(right_traj.shape[0]):
            trans = FwdKin( right_traj[idx, 0:6] )
            trans[1,3] -= 0.315
            right.append( trans )
        
        raw_left = []
        for left_point in left_action:
            raw_left.append( get_transform2( left_point ) )
            

        visualize_pcd(pcd, left, raw_left, curr_gripper)


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
