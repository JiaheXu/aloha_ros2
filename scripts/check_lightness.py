"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np
from PIL import Image
import os
import argparse
from PIL import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
# from matplotlib import pyplot as plt
import copy
import torch

# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
from utils import *

OPENESS_TH = 0.35 # Threshold to decide if a gripper opens

def get_proccessed_data( data , cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box):

    prcessed_rgb = []
    prcessed_xyz = [] 
    for idx, point in enumerate(data, 0):    
        
        point = data[idx]
        bgr = point['bgr']
        depth = point['depth']

        rgb, depth = transfer_camera_param(bgr, depth, o3d_intrinsic.intrinsic_matrix, original_image_size, resized_intrinsic_o3d.intrinsic_matrix, resized_image_size )
        # print("rgb: ", type(rgb))
                
        im_color = o3d.geometry.Image(rgb)
        im_depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        
        all_valid_resized_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                resized_intrinsic_o3d,
        )
        all_valid_resized_pcd.transform( cam_extrinsic )

        # visualize_pcd(all_valid_resized_pcd)
        xyz = xyz_from_depth(depth, resized_intrinsic_o3d.intrinsic_matrix, cam_extrinsic )
        cropped_rgb, cropped_xyz = cropping( rgb, xyz, bound_box)
        prcessed_rgb.append(cropped_rgb)
        prcessed_xyz.append(cropped_xyz)

        # save_np_image(cropped_rgb)
        # filtered_rgb, filtered_xyz = denoise(cropped_rgb, cropped_xyz, debug= True)
        # prcessed_rgb.append(filtered_rgb)
        # prcessed_xyz.append(filtered_xyz)
    
    return prcessed_rgb, prcessed_xyz

def check_within_same_data( processed_rgb ):
    for idx in range(1, 10):
        print("data {} total diff: ".format(idx) ,LA.norm(processed_rgb[idx] - processed_rgb[0]) / 255.0)
    print("")

def main():
    
    # parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    # parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    # parser.add_argument("-t", "--task", default="plate",  help="Input task name.")
    
    # args = parser.parse_args()
    # # bag_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".bag"
    # # traj_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".npy"

    cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044 , -0.63127772, 0.64917582, -0.31329509, 0.28619116])
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

    resized_image_size = (256,256)
    original_image_size = (1080, 1920) #(h,)
    # resized_intrinsic = o3d.camera.PinholeCameraIntrinsic( 256., 25, 80., 734.1779174804688*scale_y, 993.6226806640625*scale_x, 551.8895874023438*scale_y)
    fxfy = 256.0
    resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)
    resized_intrinsic_np = np.array([
        [fxfy, 0., 128.0],
        [0. ,fxfy,  128.0],
        [0., 0., 1.0]
    ])


    bound_box = np.array( [ [0.05, 0.65], [ -0.5 , 0.5], [ -0.1 , 0.6] ] )

    data1 = np.load( "./env/oct29_11am.npy", allow_pickle = True)
    data2 = np.load( "./env/oct28_16pm.npy", allow_pickle = True)
    
    bgr1 = data1[0]['bgr']
    bgr2 = data2[0]['bgr']
    # bgr2 = data1[1]['bgr']
    # print("diff")()
    data1_prcessed_rgb, data1_prcessed_xyz = get_proccessed_data( data1 , cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box)
    data2_prcessed_rgb, data2_prcessed_xyz = get_proccessed_data( data2 , cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box)
    
    check_within_same_data(data1_prcessed_rgb)
    check_within_same_data(data2_prcessed_rgb)

    img1 = data1_prcessed_rgb[0]
    img2 = data2_prcessed_rgb[0]    
    # img2 = data1_prcessed_rgb[1]

    # img1[150:210, 125:190,:] = img2[150:210, 125:190,:]
    save_np_image(img1, "img1.jpg")
    save_np_image(img2, "img2.jpg")


    diff = (img1 - img2)
    print("max: ",np.max( np.abs((img1 - img2)) ))
    # print("diff: ",LA.norm(img1 - img2))
    print("total diff: ",LA.norm(img1 - img2) / 255)
    # print("obj diff: ",LA.norm( img1[150:210, 125:190,:]  - img2[150:210, 125:190,:]  ))
    # check_between_data( data1, data2 )

if __name__ == "__main__":
    main()

    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (2, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (2, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 2, 8)
    # List of tensors