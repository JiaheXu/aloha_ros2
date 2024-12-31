"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""
import sys
import warnings
import os
import yaml

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, '../'))
from utils import *
from math_tools import *

import numpy as np
from PIL import Image
import os
import argparse
from PIL import Image
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
# from matplotlib import pyplot as plt
import copy
import torch
np.set_printoptions(suppress=True,precision=4)
# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from utils import *
from utils.visualize_keypose_frames import visualize_keyposes_and_point_clouds

def process_observation( rgb, depth, cam_intrinsic_o3d, cam_extrinsic, bound_box, return_pcd = True):
    im_color = o3d.geometry.Image(rgb)
    im_depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
    all_valid_resized_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            cam_intrinsic_o3d,
    )
    all_valid_resized_pcd.transform( cam_extrinsic )
    xyz = xyz_from_depth(depth, cam_intrinsic_o3d.intrinsic_matrix, cam_extrinsic )
    cropped_rgb, cropped_xyz = cropping( rgb, xyz, bound_box)
    filtered_rgb, filtered_xyz = denoise(cropped_rgb, cropped_xyz, debug= False)

    pcd = None
    if ( return_pcd ):
        pcd_rgb = cropped_rgb.reshape(-1, 3) / 255.0
        pcd_xyz = cropped_xyz.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
        pcd.points = o3d.utility.Vector3dVector( pcd_xyz )
        # visualize_pcd( pcd )

    if( len( np.where( np.isnan(xyz))[0] ) > 0 ):
        print(np.where( np.isnan(xyz)))
        print("head cam")
        print(" x y z has invalid point !!!!!")
        print(" x y z has invalid point !!!!!")
        print(" x y z has invalid point !!!!!")
        raise

    resized_rgb = np.transpose(filtered_rgb, (2, 0, 1) ).astype(float)
    resized_rgb = resized_rgb / 255.0
    resized_xyz = np.transpose(filtered_xyz, (2, 0, 1) ).astype(float)
    return resized_rgb, resized_xyz, pcd
# process_episode(data, world_2_head, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias)
def process_episode(data, cam_extrinsic, cam_intrinsic_o3d,  head_bound_box, hand_bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias, left_ee_2_left_cam, right_ee_2_right_cam):

    episode = []
    frame_ids = []
    obs_tensors = []
    action_tensor =  []
    camera_dicts = [ {'front': (0, 0), 'left_wrist': (0, 0), 'right_wrist': (0, 0) } ]
    gripper_tensor = []
    trajectories_tensor = []

    left_trajectory = []
    right_trajectory = []
    
    left_openess_real = []
    right_openess_real = []

    left = []
    right = []

    for idx, point in enumerate(data, 0): 

        left_transform = FwdKin(point['left_pos'][0:6] )
        # left_transform = left_bias @ left_transform @ left_tip_bias @ get_transform( np.array([0.087, 0, 0., 0., 0., 0., 1.]) )
        left_transform = left_bias @ left_transform @ left_tip_bias 
        left_rot = Rotation.from_matrix(left_transform[:3,:3])
        left_quat = left_rot.as_quat()
        left_gripper_joint = point["left_controller_pos"][6]
        left_openess = 0

        left.append(left_transform)

        if(left_gripper_joint < 0.5):
            left_openess = 0
        else:
            left_openess = 1        
        left_openess_real.append(left_openess)
        left_trajectory.append(np.array( [left_transform[0][3], left_transform[1][3], left_transform[2][3], left_quat[0], left_quat[1], left_quat[2], left_quat[3], left_openess ] ))

        right_transform = FwdKin(point['right_pos'][0:6] )
        # right_transform = right_bias @ right_transform @ right_tip_bias @ get_transform( np.array([0.087, 0, 0., 0., 0., 0., 1.]) )
        right_transform = right_bias @ right_transform @ right_tip_bias 
        right_rot = Rotation.from_matrix(left_transform[:3,:3])
        right_quat = right_rot.as_quat()
        right_gripper_joint = point["right_controller_pos"][6]
        right_openess = 0

        right.append(right_transform)

        if(right_gripper_joint < 0.5):
            right_openess = 0
        else:
            right_openess = 1
        right_openess_real.append(right_openess)
        right_trajectory.append(np.array( [right_transform[0][3], right_transform[1][3], right_transform[2][3], right_quat[0], right_quat[1], right_quat[2], right_quat[3], right_openess] ))  

    left_openess_real = np.array(left_openess_real)
    right_openess_real = np.array(right_openess_real)
    # print("left_openess_real: ", left_openess_real)
    # print("right_openess_real: ", right_openess_real)
    # use absolute pose
    left_trajectories = np.array(left_trajectory)
    # print("left gripper: ", left_trajectories[:, 7])
    left_trajectories = left_trajectories.reshape(-1,1,8)
        
    right_trajectories = np.array(right_trajectory)
    # print("right gripper: ", right_trajectories[:, 7])
    right_trajectories = right_trajectories.reshape(-1,1,8)

    trajectories = np.concatenate( [left_trajectories, right_trajectories], axis = 1)
    trajectories_tensor = torch.from_numpy(trajectories)
    
    first_frame_pcd = None

    for idx, point in enumerate(data, 0): 
        if( idx == len(data) - 1 ):
            break
        

        point = data[idx]

        head_rgb = point['head_rgb']
        head_depth = point['head_depth']
        # print("head_rgb: ", head_rgb.shape)
        # print("head_depth: ", head_depth.shape)

        resized_head_rgb, resized_head_xyz, head_pcd = process_observation( head_rgb, head_depth, cam_intrinsic_o3d, cam_extrinsic, head_bound_box)
        visualize_pcd(head_pcd, [ [left[idx]], [right[idx]] ] , drawlines = True)
        
        
        if(first_frame_pcd is None):
            first_frame_pcd = head_pcd
        # visualize_pcds( [head_pcd] )

        left_rgb = point['left_rgb']
        left_depth = point['left_depth']
        # print("left_rgb: ", head_rgb.shape)
        # print("left_depth: ", head_depth.shape)
        left_transform = FwdKin(point['left_pos'][0:6] )
        left_extrinsic = left_bias @ left_transform @ left_ee_2_left_cam
        resized_left_rgb, resized_left_xyz, left_pcd = process_observation( left_rgb, left_depth, cam_intrinsic_o3d, left_extrinsic, head_bound_box)

        right_rgb = point['right_rgb']
        right_depth = point['right_depth']
        # print("right_rgb: ", right_rgb.shape)
        # print("right_depth: ", right_depth.shape)
        right_transform = FwdKin(point['right_pos'][0:6] )
        right_extrinsic = right_bias @ right_transform @ right_ee_2_right_cam
        resized_right_rgb, resized_right_xyz, right_pcd = process_observation( right_rgb, right_depth, cam_intrinsic_o3d, right_extrinsic, head_bound_box)

        # visualize_pcds( [head_pcd, left_pcd, right_pcd] )
        n_cam = 3
        obs = np.zeros( (n_cam, 2, 3, 256, 256) )
        obs[0][0] = resized_head_rgb
        obs[0][1] = resized_head_xyz

        obs[1][0] = resized_left_rgb
        obs[1][1] = resized_left_xyz

        obs[2][0] = resized_right_rgb
        obs[2][1] = resized_right_xyz

        obs = obs.astype(float)
        obs_tensors.append( torch.from_numpy(obs) )
        # print("obs: ", obs.shape)

    frame_ids = range( len(data) )
    episode = []
    episode.append(frame_ids[:-1]) # 0
    # print("obs_tensors: ", len(obs_tensors), " ", obs_tensors[0].shape)
    episode.append(obs_tensors ) # 1
    # episode.append([trajectories_tensor[i] for i in frame_ids[1:]]) # 2, action
    # episode.append(camera_dicts) # 3
    # episode.append([trajectories_tensor[i] for i in frame_ids[:-1]]) # 4 gripper tensor
    # episode.append([trajectories_tensor[i:j+1] for i, j in zip(frame_ids[:-1], frame_ids[1:])]) # 5, traj
    # visualize_pcd(first_frame_pcd, [left, right] , drawlines = True)
    return episode

    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (2, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (2, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 2, 8)
    # List of tensors

def main():

    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="test",  help="Input task name.")
    parser.add_argument("-p", "--project", default="aloha",  help="project name.") 
    
    args = parser.parse_args()
    # bag_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".bag"
    # traj_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".npy"
    env = load_yaml( "/home/jiahe/data/config/env.yaml" )

    world_2_head = np.array( env.get("world_2_head") )
    
    world_2_left_base = np.array( env.get("world_2_left_base") )
    world_2_right_base = np.array( env.get("world_2_right_base") )
    
    left_ee_2_left_cam = np.array( env.get("left_ee_2_left_cam") )    
    right_ee_2_right_cam = np.array( env.get("right_ee_2_right_cam") )

    left_bias = world_2_left_base
    left_tip_bias = np.array( env.get("left_ee_2_left_tip") )

    right_bias = world_2_right_base
    right_tip_bias = np.array( env.get("right_ee_2_right_tip") )

    head_bound_box = np.array( env.get("head_bounding_box") )
    hand_bound_box = np.array( env.get("hand_bounding_box") )

    o3d_data = env.get("intrinsic_o3d")[0]
    cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])

    # bound_box = np.array( [ [0.2, 0.6], [ -0.4 , 0.4], [ -0.1 , 0.6] ] )
    task_name = args.task 
    print("task_name: ", task_name)

    processed_data_dir = "/home/jiahe/data/aloha/keypose"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    dir_path = "/home/jiahe/data/raw_demo/" + task_name + '/'

    save_data_dir = processed_data_dir + '/' + task_name + "+0"
    if ( os.path.isdir(save_data_dir) == False ):
        os.mkdir(save_data_dir)
        
   
    file =  str(args.data_index) + "_keypose.npy"
    print("processing: ", dir_path+file)
    data = np.load(dir_path+file, allow_pickle = True)


    
    episode = process_episode(data, world_2_head, cam_intrinsic_o3d, head_bound_box, hand_bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias, left_ee_2_left_cam, right_ee_2_right_cam)


    np.save("{}/{}/ep{}".format(processed_data_dir,task_name + "+0",args.data_index), episode)

    print("finished ", task_name, " data: ", args.data_index)
    print("")

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
