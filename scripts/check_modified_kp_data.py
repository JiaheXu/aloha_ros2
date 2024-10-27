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

from utils import *

def main():
    
    # data = np.load("./2arms_open_pen/1.npy", allow_pickle = True)
    # task = "close_pen"
    # task = "pouring_into_bowl" # not yet
    # task = "put_block_into_bowl"  
    # task = "pick_up_plate"
    # task = "stack_block"
    # task = "stack_bowl_single_arm"
    task = "stack_bowl_dual_arm"  
    # data_idxs = [1, 4, 31, 32, 33, 34, 35]
    # data_idxs =  [1, 4, 31, 32, 33, 34, 35]
    start_ep = 38
    end_ep = 50
    data_idxs =  range(start_ep,end_ep+1)
    interpolation_length = 26
    for data_idx in data_idxs:
        episode = np.load("./{}/ep{}_rgb.npy".format(data_idx,data_idx) , allow_pickle = True)
        print("data_idx: ", data_idx)
        for idx, frame in enumerate(episode[0]):
            # if(idx % 2 !=0):
                # continue
            #if(idx !=0):
            #    continue
            rgb  = episode[1][idx][0][0]
            print("idx: ", idx)
            # print("episode: ", episode.shape)
            rgb = rgb.numpy()
            # print("rgb: ", rgb.shape)
            resized_img_data = np.transpose(rgb, (1, 2, 0) ).astype(float) # (0,1)
            resized_img_data = resized_img_data
            xyz  = episode[1][idx][0][1]
            xyz = xyz.numpy()
            resized_xyz = np.transpose(xyz, (1, 2, 0) ).astype(float) # (0,1)

            
            pcd_rgb = resized_img_data.reshape(-1, 3)
            pcd_xyz = resized_xyz.reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
            pcd.points = o3d.utility.Vector3dVector( pcd_xyz )

            left = []
            right = []
            current_state = episode[4][idx].numpy()
            # print("current_state: ", current_state)
            traj_np = episode[2][idx].flatten(1, -1).numpy()
            #traj = traj_interpolation( traj_np )
            print("traj: ", traj_np.shape)
            # print("episode[5][idx]: ", traj_np)
            
            left.append( get_transform(traj_np[0, 0:7]) )    
            right.append( get_transform(traj_np[1, 0:7]) )

            current_state = episode[4][idx].numpy()
            curr_pose = []
            curr_pose.append(get_transform(current_state[0, 0:7]))
            curr_pose.append(get_transform(current_state[1, 0:7]))

            visualize_pcd(pcd, [left, right], curr_pose)

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
