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

# def process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias, frame_rate = 8, future_length = 30 ):
def process_episode(data, world_2_head, cam_intrinsic_o3d, head_bound_box, hand_bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias, left_ee_2_left_cam, right_ee_2_right_cam, frame_rate = 8, future_length = 30 ):

    episode = []
    frame_ids = []
    obs_tensors = []
    action_tensor =  []
    camera_dicts = []
    gripper_tensor = []
    trajectories_tensor = []



    # for point in data:
    #     left_gripper_max = max(left_gripper_max, point["left_pos"][6])
    #     left_gripper_min = min(left_gripper_min, point["left_pos"][6])
    #     right_gripper_max = max(right_gripper_max, point["right_pos"][6])
    #     right_gripper_min = min(right_gripper_min, point["right_pos"][6])

    for idx, point in enumerate(data, 0):    
        if(idx % frame_rate != 0):
            continue

        if( idx >= len(data) -2 ):
            continue
        frame_ids.append(idx)

        point = data[idx]
        bgr = point['bgr']
        # rgb = bgr[...,::-1].copy()
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
        # save_np_image(cropped_rgb)
        
        filtered_rgb, filtered_xyz = denoise(cropped_rgb, cropped_xyz, debug= True)

        # pcd_rgb = cropped_rgb.reshape(-1, 3) / 255.0
        # pcd_xyz = cropped_xyz.reshape(-1, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
        # pcd.points = o3d.utility.Vector3dVector( pcd_xyz )
        # visualize_pcd(pcd )
        if( len( np.where( np.isnan(xyz))[0] ) >0 ):
            print(np.where( np.isnan(xyz)))
            print(" x y z has invalid point !!!!!")
            print(" x y z has invalid point !!!!!")
            print(" x y z has invalid point !!!!!")
            raise

        # xyz_rgb_validation(rgb, xyz)

        resized_img_data = np.transpose(filtered_rgb, (2, 0, 1) ).astype(float)
        resized_img_data = resized_img_data / 255.0
        # print("resized_img_data: ", resized_img_data.shape)
        resized_xyz = np.transpose(filtered_xyz, (2, 0, 1) ).astype(float)
        # print("resized_xyz: ", resized_xyz.shape)


        n_cam = 1
        obs = np.zeros( (n_cam, 2, 3, 256, 256) )
        obs[0][0] = resized_img_data
        obs[0][1] = resized_xyz

        obs = obs.astype(float)
        obs_tensors.append( torch.from_numpy(obs) )
        

        left_trajectory = []
        right_trajectory = []
        delta_left_trajectory = []
        delta_right_trajectory = []

        eps = 1e-3
        for point in data[idx : idx + future_length]:
            left_transform = get_transform(point['left_ee'] )
            left_transform = left_bias @ left_transform @ left_tip_bias @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
            left_rot = Rotation.from_matrix(left_transform[:3,:3])
            left_quat = left_rot.as_quat()
            left_gripper_joint = max ( min( float(point["left_pos"][6]) , gripper_max), gripper_min )
            left_openess = ( left_gripper_joint - gripper_min ) / (gripper_max - gripper_min  + eps)
            if(left_openess < OPENESS_TH):
                left_openess = 0
            else:
                left_openess = 1
            print("left_openess: ", left_openess)
            left_trajectory.append(np.array( [left_transform[0][3], left_transform[1][3], left_transform[2][3], left_quat[0], left_quat[1], left_quat[2], left_quat[3], left_openess ] ))

            right_transform = get_transform(point['right_ee'] )
            right_transform = right_bias @ right_transform @ right_tip_bias @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
            right_rot = Rotation.from_matrix(right_transform[:3,:3])
            right_quat = right_rot.as_quat()
            right_gripper_joint = max ( min( float(point["right_pos"][6]) , gripper_max), gripper_min )
            right_openess = ( right_gripper_joint - gripper_min ) / (gripper_max - gripper_min + eps)
            if(right_openess < OPENESS_TH):
                right_openess = 0
            else:
                right_openess = 1
            print("right_openess: ", right_openess)
            right_trajectory.append(np.array( [right_transform[0][3], right_transform[1][3], right_transform[2][3], right_quat[0], right_quat[1], right_quat[2], right_quat[3], right_openess] ))  
    
            # print("right_openess: ", right_openess)


        for idx, trans in enumerate(left_trajectory, 0):
            if(idx == 0):
                continue
            delta_trans = get_transform(left_trajectory[idx]) @ inv( get_transform(left_trajectory[0] ) )
            delat_rot = Rotation.from_matrix(delta_trans[:3,:3])
            delta_quat = delat_rot.as_quat()
            openess = left_trajectory[idx][-1]
            # print("delta_openess: ", delta_openess)
            action = np.array( [delta_trans[0][3], delta_trans[1][3], delta_trans[2][3], delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3], openess] )
            delta_left_trajectory.append( action )

        for idx, trans in enumerate(right_trajectory, 0):
            if(idx == 0):
                continue
            delta_trans = get_transform(right_trajectory[idx]) @ inv( get_transform(right_trajectory[0] ) )
            delat_rot = Rotation.from_matrix(delta_trans[:3,:3])
            delta_quat = delat_rot.as_quat()
            openess = right_trajectory[idx][-1]
            # print("delta_openess: ", delta_openess)
            action = np.array( [delta_trans[0][3], delta_trans[1][3], delta_trans[2][3], delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3], openess] )
            delta_right_trajectory.append( action )

        # visualize_pcd_transform(all_valid_resized_pcd, left_trajectory)
        # visualize_pcd_transform(all_valid_resized_pcd, right_trajectory)

        # visualize_pcd_delta_transform(all_valid_resized_pcd, left_trajectory[0], delta_left_trajectory)
        # visualize_pcd_delta_transform(all_valid_resized_pcd, right_trajectory[0], delta_right_trajectory)


        delta_left_transform = get_transform(left_trajectory[-1]) @ inv( get_transform(left_trajectory[0]) )
        delat_left_rot = Rotation.from_matrix(delta_left_transform[:3,:3])
        delta_left_quat = delat_left_rot.as_quat()
        delta_left_openess = left_trajectory[-1][-1]
        left_action = np.array( [delta_left_transform[0][3], delta_left_transform[1][3], delta_left_transform[2][3], delta_left_quat[0], delta_left_quat[1], delta_left_quat[2], delta_left_quat[3], delta_left_openess] )
        left_action = left_action.reshape(1,8)
        left_gripper = copy.deepcopy( left_trajectory[0])
        left_gripper = left_gripper.reshape(1,8)
        # left_trajectories = np.array(delta_left_trajectory)

        # use absolute pose
        left_trajectories = np.array(left_trajectory[1:])
        left_trajectories = left_trajectories.reshape(-1,1,8)

        
        delta_right_transform = get_transform(right_trajectory[-1]) @ inv( get_transform(right_trajectory[0] ) )
        delat_right_rot = Rotation.from_matrix(delta_right_transform[:3,:3])
        delta_right_quat = delat_right_rot.as_quat()
        delta_right_openess = right_trajectory[-1][-1]

        right_action = np.array( [delta_right_transform[0][3], delta_right_transform[1][3], delta_right_transform[2][3], delta_right_quat[0], delta_right_quat[1], delta_right_quat[2], delta_right_quat[3], delta_right_openess] )
        right_action = right_action.reshape(1,8)
        right_gripper = copy.deepcopy( right_trajectory[0])
        right_gripper = right_gripper.reshape(1,8)
        # right_trajectories = np.array(delta_right_trajectory)
        right_trajectories = np.array(right_trajectory[1:])
        right_trajectories = right_trajectories.reshape(-1,1,8)
        # print("trajectories: ", trajectories.shape)

        action = np.concatenate( [left_action, right_action], axis = 0)
        action_tensor.append( torch.from_numpy(action) )

        gripper = np.concatenate( [left_gripper, right_gripper], axis = 0)
        gripper_tensor.append( torch.from_numpy(gripper) )

        trajectories = np.concatenate( [left_trajectories, right_trajectories], axis = 1)
        trajectories_tensor.append( torch.from_numpy(trajectories) )


    episode = []
    episode.append(frame_ids) # 0

    episode.append(obs_tensors) # 1
        
    episode.append(action_tensor) # 2

    episode.append(camera_dicts) # 3

    episode.append(gripper_tensor) # 4

    episode.append(trajectories_tensor) # 5

    return episode



def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="test",  help="Input task name.")
    parser.add_argument("-p", "--project", default="aloha",  help="project name.")    

    args = parser.parse_args()
    # bag_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".bag"
    # traj_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".npy"
    env = load_yaml( "~/data/config/env.yaml" )

    world_2_head = np.array( env.get("world_2_head") )
    world_2_left_base = np.array( env.get("world_2_left_base") )
    world_2_right_base = np.array( env.get("world_2_right_base") )
    
    left_ee_2_left_cam = np.array( env.get("left_ee_2_left_cam") )    
    right_ee_2_right_cam = np.array( env.get("right_ee_2_right_cam") )

    head_bound_box = np.array( env.get("head_bounding_box") )
    hand_bound_box = np.array( env.get("hand_bounding_box") )

    o3d_data = env.get("intrinsic_o3d")[0]
    cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])

    # bound_box = np.array( [ [0.2, 0.6], [ -0.4 , 0.4], [ -0.1 , 0.6] ] )
    task_name = args.task 
    print("task_name: ", task_name)

    processed_data_dir = "~/data/aloha/trajectory"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    
    dir_path = "~/data/raw_demo/" + task_name + '/'

    save_data_dir = processed_data_dir + '/' + task_name + "+0"
    if ( os.path.isdir(save_data_dir) == False ):
        os.mkdir(save_data_dir)
        
   
    # file = str(args.data_index) + "_trajectory.npy"
    file = str(args.data_index) + ".npy"
    print("processing: ", dir_path+file)
    data = np.load(dir_path+file, allow_pickle = True)

    left_bias = world_2_left_base
    left_tip_bias = get_transform( np.array([-0.028, 0.01, 0.01,      0., 0., 0., 1.] ))

    right_bias = world_2_right_base
    right_tip_bias = get_transform( np.array([-0.035, 0.01, -0.008,      0., 0., 0., 1.] ))
   
    # episode = process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_img_size, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias)
    episode = process_episode(data, world_2_head, cam_intrinsic_o3d, head_bound_box, hand_bound_box, world_2_left_base, left_tip_bias, world_2_right_base, right_tip_bias, left_ee_2_left_cam, right_ee_2_right_cam)
    np.save("{}/{}/ep{}".format(processed_data_dir,task_name,args.data_index), episode)
    
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