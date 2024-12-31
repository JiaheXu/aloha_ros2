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

    # resized_rgb = np.transpose(cropped_rgb, (2, 0, 1) ).astype(float)
    # resized_rgb = resized_rgb / 255.0
    # resized_xyz = np.transpose(cropped_xyz, (2, 0, 1) ).astype(float)
    return resized_rgb, resized_xyz, pcd

# def process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias, frame_rate = 8, future_length = 30 ):
def process_episode(data, cam_extrinsic, cam_intrinsic_o3d, head_bound_box, hand_bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias, left_ee_2_left_cam, right_ee_2_right_cam, left_gripper_threshold= 0.4, right_gripper_threshold = 0.4, frame_rate = 3, future_length = 15):

    episode = []
    frame_ids = []
    obs_tensors = []
    action_tensor =  []
    # camera_dicts = [ {'front': (0, 0), 'left_wrist': (0, 0), 'right_wrist': (0, 0) } ]
    camera_dicts = [ {'front': (0, 0)}]
    gripper_tensor = []
    trajectories_tensor = []

    left_trajectory = []
    right_trajectory = []
    
    left_openess_real = []
    right_openess_real = []

    left = []
    right = []

    for idx, point in enumerate(data, 0):    
        if(idx % frame_rate != 0):
            continue

        if( idx >= len(data) -2 ):
            continue

        left_openess_real = []
        right_openess_real = []

        
        frame_ids.append(idx)

        point = data[idx]
        head_rgb = point['head_rgb']
        head_depth = point['head_depth']

        resized_head_rgb, resized_head_xyz, head_pcd = process_observation( head_rgb, head_depth, cam_intrinsic_o3d, cam_extrinsic, head_bound_box, return_pcd=True)
        # print("rgb: ", type(rgb))
                
        # left_rgb = point['left_rgb']
        # left_depth = point['left_depth']
        # left_transform = FwdKin(point['left_pos'][0:6] )
        # left_extrinsic = left_bias @ left_transform @ left_ee_2_left_cam
        # resized_left_rgb, resized_left_xyz, left_pcd = process_observation( left_rgb, left_depth, cam_intrinsic_o3d, left_extrinsic, head_bound_box, return_pcd=False)

        # right_rgb = point['right_rgb']
        # right_depth = point['right_depth']
        # right_transform = FwdKin(point['right_pos'][0:6] )
        # right_extrinsic = right_bias @ right_transform @ right_ee_2_right_cam
        # resized_right_rgb, resized_right_xyz, right_pcd = process_observation( right_rgb, right_depth, cam_intrinsic_o3d, right_extrinsic, head_bound_box, return_pcd=False)

        # visualize_pcds( [head_pcd, left_pcd, right_pcd] )
        n_cam = len(camera_dicts)
        obs = np.zeros( (n_cam, 2, 3, 256, 256) )
        obs[0][0] = resized_head_rgb
        obs[0][1] = resized_head_xyz

        # obs[1][0] = resized_left_rgb
        # obs[1][1] = resized_left_xyz

        # obs[2][0] = resized_right_rgb
        # obs[2][1] = resized_right_xyz

        obs = obs.astype(float)
        obs_tensors.append( torch.from_numpy(obs) )
        

        left_trajectory = []
        right_trajectory = []

        left_transforms = []
        right_transforms = []

        eps = 1e-3
        for point in data[idx : idx + future_length]:
            left_transform = FwdKin(point['left_pos'][0:6] )
            # left_transform = left_bias @ left_transform @ left_tip_bias @ get_transform( np.array([0.087, 0, 0., 0., 0., 0., 1.]) )
            left_transform = left_bias @ left_transform @ left_tip_bias 
            left_rot = Rotation.from_matrix(left_transform[:3,:3])
            left_quat = left_rot.as_quat()
            left_gripper_joint = point["left_controller_pos"][6]
            left_openess = 0
            left_transforms.append(left_transform)
            # print("left_gripper_joint: ", left_gripper_joint)
            if(left_gripper_joint < left_gripper_threshold):
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
            right_transforms.append(right_transform)
            # print("right_gripper_joint: ", right_gripper_joint)
            if(right_gripper_joint < right_gripper_threshold):
                right_openess = 0
            else:
                right_openess = 1
            right_openess_real.append(right_openess)
            right_trajectory.append(np.array( [right_transform[0][3], right_transform[1][3], right_transform[2][3], right_quat[0], right_quat[1], right_quat[2], right_quat[3], right_openess] ))  
        
        # print("idx: ", idx)
        # print("openness: ", left_openess_real[0], " ", right_openess_real[0])
        # visualize_pcd(head_pcd, [ [left_transforms[0]], [right_transforms[0]] ] )


        left_gripper = copy.deepcopy( left_trajectory[0])
        left_gripper = left_gripper.reshape(1,8)
        left_action = copy.deepcopy( left_trajectory[-1])
        left_action = left_action.reshape(1,8)
        left_trajectories = np.array(left_trajectory)
        left_trajectories = left_trajectories.reshape(-1,1,8)

        right_gripper = copy.deepcopy( right_trajectory[0])
        right_gripper = right_gripper.reshape(1,8)
        right_action = copy.deepcopy( right_trajectory[-1])
        right_action = right_action.reshape(1,8)
        right_trajectories = np.array(right_trajectory)
        right_trajectories = right_trajectories.reshape(-1,1,8)

        action = np.concatenate( [left_action, right_action], axis = 0)
        action_tensor.append( torch.from_numpy(action) )
        # print("action: ", action.shape)
        gripper = np.concatenate( [left_gripper, right_gripper], axis = 0)
        gripper_tensor.append( torch.from_numpy(gripper) )

        trajectories = np.concatenate( [left_trajectories, right_trajectories], axis = 1)
        trajectories_tensor.append( torch.from_numpy(trajectories) )


    episode = []
    episode.append(range( len(frame_ids)) ) # 0

    episode.append(obs_tensors) # 1
        
    episode.append(action_tensor) # 2

    episode.append(camera_dicts) # 3

    episode.append(gripper_tensor) # 4

    episode.append(trajectories_tensor) # 5
    return episode



def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="lift_ball",  help="Input task name.")
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

    processed_data_dir = "/home/jiahe/data/aloha/trajectory"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    
    dir_path = "/home/jiahe/data/raw_demo/" + task_name + '/traj/'

    save_data_dir = processed_data_dir + '/' + task_name + "+0"
    if ( os.path.isdir(save_data_dir) == False ):
        os.mkdir(save_data_dir)

    right_gripper_threshold = 0.4
    if(task_name == "insert_battery"):
        right_gripper_threshold = 0.2
   
    # file = str(args.data_index) + "_trajectory.npy"
    file = str(args.data_index) + ".npy"
    print("processing: ", dir_path+file)
    data = np.load(dir_path+file, allow_pickle = True)
   
    right_gripper_threshold = 0.4
    if(task_name == "insert_battery"):
        # print("!!!!!!!!!!!!!!")
        if(int(args.data_index == 13)):
            right_gripper_threshold = 0.1
        if(int(args.data_index == 21)):
            right_gripper_threshold = 0.3
        if(int(args.data_index == 23)):
            right_gripper_threshold = 0.1
        if(int( args.data_index) == 24):
            right_gripper_threshold = 0.1
    # episode = process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_img_size, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias)
    episode = process_episode(data, world_2_head, cam_intrinsic_o3d, head_bound_box, hand_bound_box, world_2_left_base, left_tip_bias, world_2_right_base, right_tip_bias, left_ee_2_left_cam, right_ee_2_right_cam,
    #  left_gripper_threshold = args.left_gripper_threshold,
     right_gripper_threshold = right_gripper_threshold,
    )
    np.save("{}/{}/ep{}".format(processed_data_dir,task_name+ "+0",args.data_index), episode)
    
    print("finished ", task_name, " data: ", args.data_index)
    print("")

if __name__ == "__main__":
    main()

    # [frame_ids],  # we use chunk and max_episode_length to index it [0,1,2...n]
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (2, 8), predited action (keypose, next goal position)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (2, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 2, 8), (for traj, first frame is the current state)
    # List of tensors