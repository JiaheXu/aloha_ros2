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
from numpy import linalg as LA
import PIL.Image as PIL_Image
from sklearn.neighbors import NearestNeighbors

# def denoise(rgb, xyz):
#     # cropped_rgb
#     up = xyz[1:-1, 1:-1] - xyz[ 0:-2, 1: -1]
#     down = xyz[1:-1, 1:-1] - xyz[ 2:, 1: -1]
#     left = xyz[1:-1, 1:-1] - xyz[ 1:-1, 0: -2]
#     right = xyz[1:-1, 1:-1] - xyz[ 1:-1, 2: ]

#     up = LA.norm(up, axis = 2)
#     down = LA.norm(down, axis = 2)
#     left = LA.norm(left, axis = 2)
#     right = LA.norm(right, axis = 2)
#     stacked = np.stack( [up, down, left, right], axis = 2 )

#     nearest_neighbor = np.min(stacked, axis = 2)      
#     invalid_idx = np.where( nearest_neighbor > 0.2 )
#     print("invalid_idx: ", invalid_idx)
#     # invalid_idx[:][:] += 1
#     invalid_idx = np.array(invalid_idx)
#     # invalid_idx += 1
#     rgb[invalid_idx] = np.array([1.0, 0., 0.])
  
#     return rgb, xyz


def main():
    
    bound_box = np.array( [ [0.05, 0.55], [ -0.5 , 0.5], [ -0.3 , 0.6] ] )

    task = "./processed_bimanual/plate"
    end_ep = 42
    data_idxs =  range(1,end_ep+1)

    interpolation_length = 26

    for data_idx in data_idxs:

        episode = np.load("./{}/ep{}.npy".format(task, data_idx) , allow_pickle = True)

        for idx, frame in enumerate( episode[0] ):
            rgb  = episode[1][idx][0][0]
            # print("episode: ", episode.shape)
            rgb = rgb.numpy()
            # print("rgb: ", rgb.shape)
            rgb = np.transpose(rgb, (1, 2, 0) ).astype(float) # (0,1)
            # rgb = (rgb*255).astype(np.uint8)
            save_np_image( rgb )
            # resized_img_data = resized_img_data
            xyz  = episode[1][idx][0][1]
            xyz = xyz.numpy()
            xyz = np.transpose(xyz, (1, 2, 0) ).astype(float) # (0,1)
            

            # filtered_rgb, filtered_xyz = denoise(rgb, xyz, debug= True)
            # save_np_image( filtered_rgb )
            # pcd_rgb = filtered_rgb.reshape(-1, 3)
            # pcd_xyz = filtered_xyz.reshape(-1, 3)

            pcd_rgb = rgb.reshape(-1, 3)
            pcd_xyz = xyz.reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
            pcd.points = o3d.utility.Vector3dVector( pcd_xyz )

            left = []
            right = []
            current_state = episode[4][idx].numpy()
            # print("current_state: ", current_state)
            traj_np = episode[5][idx].flatten(1, -1).numpy()
            traj = traj_interpolation( traj_np )
            # print("traj: ", traj.shape)
            # print("episode[5][idx]: ", traj_np)
            for step in range(traj.shape[0]):
                left.append( get_transform(traj[step][0:7]) )    
                right.append( get_transform(traj[step][8:15]) )    
            visualize_pcd(pcd, [left, right] )

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
