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
    start_ep = 39
    end_ep = 39
    data_idxs =  range(start_ep,end_ep+1)
    interpolation_length = 26

    to_plane = False
    use_real_env = True
    use_real_obj = True
    change_rgb = False

    for data_idx in data_idxs:
        

        episode = np.load("./processed_bimanual_keypose/{}/ep{}.npy".format(task, data_idx) , allow_pickle = True)
        print("loading: ", "./processed_bimanual_keypose/{}/ep{}.npy".format(task, data_idx))
        print("data_idx: ", data_idx)
        for idx, frame in enumerate(episode[0]):
            if(idx ==8 ):
                break
            realworld_data = np.load("./39_realworld/step_{}.npy".format(idx), allow_pickle = True)
            realworld_data = realworld_data.item()

            rgb  = episode[1][idx][0][0]
            print("idx: ", idx)
            # print("episode: ", episode.shape)
            rgb = rgb.numpy()
            # print("rgb: ", rgb.shape)
            resized_img_data = np.transpose(rgb, (1, 2, 0) ).astype(float) # (0,1)
            resized_img_data = resized_img_data
            xyz  = episode[1][idx][0][1]
            xyz = xyz.numpy()

            red = resized_img_data[:,:,0]
            green = resized_img_data[:,:,1]
            blue = resized_img_data[:,:,2]

            resized_xyz = np.transpose(xyz, (1, 2, 0) ).astype(float) # (0,1)

            x = resized_xyz[:,:,0]
            y = resized_xyz[:,:,1]
            z = resized_xyz[:,:,2]




            real_rgb = realworld_data["rgb"]
            real_rgb = real_rgb[0,0].numpy()
            real_rgb = np.transpose(real_rgb, (1, 2, 0) ).astype(float) # (0,1)
            real_xyz = realworld_data["xyz"][0][0].numpy()
            real_xyz = np.transpose(real_xyz, (1, 2, 0) ).astype(float) # (0,1)
            
            # resized_img_data[105:150, 70:130,:] *= 0.0
            # resized_img_data[97:140, 190:240,:] *= 0.0
            if(use_real_obj):
                resized_img_data[105:150, 70:130,:] = real_rgb[105:150, 70:130,:]
                resized_xyz[105:150, 70:130,:] = real_xyz[105:150, 70:130,:]            

            if(use_real_env):
                real_rgb[105:150, 70:130,:] = resized_img_data[105:150, 70:130,:]
                real_xyz[105:150, 70:130,:] = resized_xyz[105:150, 70:130,:]
                resized_img_data = real_rgb
                resized_xyz = real_xyz

            if(to_plane):
                resized_xyz[105:150, 70:130, 2] = 0.01
            # green_idx = np.where( (red<0.7) & (green>=0.4) & (blue<=0.6) & (z > - 0.06) & (x > 0.15) & (x < 0.5) )
            # print("green_idx: ", green_idx)
            
        
            if(change_rgb):
                # green_idx = np.where( (red<0.7) & (green>=0.4) & (blue<=0.6) & (z > - 0.06) & (x > 0.15) & (x < 0.5) )
                # test = resized_img_data[green_idx]
                # test[:,0] -= 0.1
                # test[:,1] -= 0.1
                # test[:,2] -= 0.1

                color_idx = np.where( (red > 0.1) & (green> 0.1) & (blue > 0.1) )
                test = resized_img_data[color_idx]
                test[:,0] -= 0.1
                test[:,1] -= 0.1
                test[:,2] -= 0.1

                test = np.clip( test, 0, 1)
                resized_img_data[color_idx] = test
                # resized_rgb = np.transpose(resized_img_data, (2, 0, 1) ).astype(float)
                # episode[1][idx][0][0] = torch.from_numpy( resized_rgb )

            pcd_rgb = resized_img_data.reshape(-1, 3)
            pcd_xyz = resized_xyz.reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
            pcd.points = o3d.utility.Vector3dVector( pcd_xyz )

            left = []
            right = []
            current_state = episode[4][idx].numpy()
            traj_np = episode[2][idx].flatten(1, -1).numpy()
            # print("traj: ", traj_np.shape)
            
            left.append( get_transform(traj_np[0, 0:7]) )    
            right.append( get_transform(traj_np[1, 0:7]) )    
            visualize_pcd(pcd, [left, right] )

            resized_xyz2 = np.transpose(resized_xyz, (2, 0, 1) ).astype(float)
            episode[1][idx][0][1] = torch.from_numpy( resized_xyz2 )
            resized_rgb = np.transpose(resized_img_data, (2, 0, 1) ).astype(float)
            episode[1][idx][0][0] = torch.from_numpy( resized_rgb )

        # if(use_real_obj):
        #     np.save("ep{}_real_obj".format(data_idx), episode)
        # if(use_real_env):
        #     np.save("ep{}_real_env".format(data_idx), episode)
        # if(to_plane):
            np.save("ep{}_test".format(data_idx), episode)

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
