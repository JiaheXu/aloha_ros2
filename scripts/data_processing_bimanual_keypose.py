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
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from utils import *
from utils.visualize_keypose_frames import visualize_keyposes_and_point_clouds

# Threshold to decide if a gripper open
# OPENESS_TH = 0.35  # close pen task
# OPENESS_TH = 0.12  # pick_up_plate
# OPENESS_TH = 0.65  # pouring into bowl
# OPENESS_TH = 0.54  # put block into bowl
# OPENESS_TH = 0.54  # stack block

OPENESS_TH = 0.35 # Threshold to decide if a gripper opens
# 0.3
BUFFER_SIZE = 4
STOP_TH = 0.001 # Threshold to decide if a gripper stops


def get_eef_velocity_from_trajectories(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    velocities = trajectories[1:] - trajectories[:-1]

    V = np.linalg.norm(velocities[:, :3], axis=-1)
    W = np.linalg.norm(velocities[:, 3:6], axis=-1)

    velocities = np.concatenate(
        [velocities, [velocities[-1]]],
        # [velocities[[0]], velocities],
        axis=0
    )
    accelerations = velocities[1:] - velocities[:-1]

    A = np.linalg.norm(accelerations[:, :3], axis=-1)

    return V, W, A


def gripper_state_changed(trajectories):
    trajectories = np.stack(
        [trajectories[0]] + trajectories, axis=0
    )
    openess = trajectories[:, -1]
    changed = openess[:-1] != openess[1:]

    return np.where(changed)[0]


def keypoint_discovery(trajectories, buffer_size=5):
    """Determine way point from the trajectories.

    Args:
        trajectories: a list of 1-D np arrays.  Each array is
            7-dimensional (x, y, z, euler_x, euler_y, euler_z, opene).
        stopping_delta: the minimum velocity to determine if the
            end effector is stopped.

    Returns:
        an Integer array indicates the indices of waypoints
    """
    V, W, A = get_eef_velocity_from_trajectories(trajectories)

    # waypoints are local minima of gripper movement
    _local_max_A = argrelextrema(A, np.greater)[0]
    topK = np.sort(A)[::-1][int(A.shape[0] * 0.1)]
    large_A = A[_local_max_A] >= topK
    _local_max_A = _local_max_A[large_A].tolist()

    local_max_A = [_local_max_A.pop(0)]
    for i in _local_max_A:
        if i - local_max_A[-1] >= buffer_size:
            local_max_A.append(i)

    # waypoints are frame that suddenly stop / move
    smooth_kernel = 3
    smooth_V = gaussian_filter1d(V, smooth_kernel)
    gripper_motion_changed = smooth_V < STOP_TH
    pad_gripper_motion_changed = np.pad(gripper_motion_changed,
                                       (1, 1), mode='symmetric')
    gripper_motion_changed_ids = np.where(
        pad_gripper_motion_changed[:-1]
        != pad_gripper_motion_changed[1:]
    )[0]

    # waypoints are frames with changing gripper states
    gripper_state_changed_ids = gripper_state_changed(trajectories)

    last_frame = [len(trajectories) - 1]

    keyframe_inds = (
        local_max_A +
        gripper_state_changed_ids.tolist() +
        gripper_motion_changed_ids.tolist() +
        last_frame
    )
    keyframe_inds = np.unique(keyframe_inds)

    keyframes = [trajectories[i] for i in keyframe_inds]

    return keyframes, keyframe_inds


def process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias):

    episode = []
    frame_ids = []
    obs_tensors = []
    action_tensor =  []
    camera_dicts = []
    gripper_tensor = []
    trajectories_tensor = []

    # gripper end val is around 0.6 ~ 1.6
    left_gripper_max = 1.62
    left_gripper_min = 0.62

    right_gripper_max = 1.62
    right_gripper_min = 0.62

    gripper_max = 1.62
    gripper_min = 0.62
    # for point in data:
    #     left_gripper_max = max(left_gripper_max, point["left_pos"][6])
    #     left_gripper_min = min(left_gripper_min, point["left_pos"][6])
    #     right_gripper_max = max(right_gripper_max, point["right_pos"][6])
    #     right_gripper_min = min(right_gripper_min, point["right_pos"][6])

    left_trajectory = []
    right_trajectory = []
    for idx, point in enumerate(data, 0):    
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
        
        eps = 1e-3
        left_transform = get_transform(point['left_ee'] )
        left_transform = left_bias @ left_transform @ left_tip_bias @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
        left_rot = Rotation.from_matrix(left_transform[:3,:3])
        left_quat = left_rot.as_quat()
        left_gripper_joint = max ( min( float(point["left_pos"][6]) , gripper_max), gripper_min )
        left_openess = ( left_gripper_joint - gripper_min ) / (gripper_max - gripper_min  + eps)
        print("left_openess: ", left_openess)
        left_trajectory.append(np.array( [left_transform[0][3], left_transform[1][3], left_transform[2][3], left_quat[0], left_quat[1], left_quat[2], left_quat[3], left_openess ] ))

        right_transform = get_transform(point['right_ee'] )
        right_transform = right_bias @ right_transform @ right_tip_bias @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
        right_rot = Rotation.from_matrix(right_transform[:3,:3])
        right_quat = right_rot.as_quat()
        right_gripper_joint = max ( min( float(point["right_pos"][6]) , gripper_max), gripper_min )
        right_openess = ( right_gripper_joint - gripper_min ) / (gripper_max - gripper_min + eps)
        print("right_openess: ", right_openess)
        right_trajectory.append(np.array( [right_transform[0][3], right_transform[1][3], right_transform[2][3], right_quat[0], right_quat[1], right_quat[2], right_quat[3], right_openess] ))  

    # use absolute pose
    left_trajectories = np.array(left_trajectory)
    print("left gripper: ", left_trajectories[:, 7])
    left_trajectories = left_trajectories.reshape(-1,1,8)
        
    right_trajectories = np.array(right_trajectory)
    print("right gripper: ", right_trajectories[:, 7])

    right_trajectories = right_trajectories.reshape(-1,1,8)

    trajectories = np.concatenate( [left_trajectories, right_trajectories], axis = 1)
    # visualize_pcd_transform(all_valid_resized_pcd, left_trajectory)
    # visualize_pcd_transform(all_valid_resized_pcd, right_trajectory)
    trajectories_tensor = torch.from_numpy(trajectories)
    left_traj = [np.concatenate([traj[:3], [0, 0, 0], traj[-1:] >= OPENESS_TH], axis=0)
                 for traj in left_trajectory]
    _, left_keyframe_inds = keypoint_discovery(
        left_traj, buffer_size=BUFFER_SIZE
    )
    right_traj = [np.concatenate([traj[:3], [0, 0, 0], traj[-1:] >= OPENESS_TH], axis=0)
                  for traj in right_trajectory]
    _, right_keyframe_inds = keypoint_discovery(
        right_traj, buffer_size=BUFFER_SIZE
    )

    frame_ids = np.unique(np.concatenate([left_keyframe_inds, right_keyframe_inds]))
    frame_ids = [0] + frame_ids.tolist()

    episode = []
    episode.append(frame_ids[:-1]) # 0
    episode.append([obs_tensors[i] for i in frame_ids[:-1]]) # 1e    
    episode.append([trajectories_tensor[i] for i in frame_ids[1:]]) # 2
    episode.append(camera_dicts) # 3
    episode.append([trajectories_tensor[i] for i in frame_ids[:-1]]) # 2
    episode.append([trajectories_tensor[i:j+1] for i, j in zip(frame_ids[:-1], frame_ids[1:])]) # 2

    keypose_traj = (
        [trajectories_tensor[i, [0]].float() for i in frame_ids]
        + [trajectories_tensor[i, [1]].float() for i in frame_ids]
    )
    nonkeypose_traj = (
        [trajectories_tensor[i, [0]].float() for i in range(trajectories_tensor.shape[0]) if i not in frame_ids]
        + [trajectories_tensor[i, [1]].float() for i in range(trajectories_tensor.shape[0]) if i not in frame_ids]
    )
    visualize_keyposes_and_point_clouds(obs_tensors[0][:, [1]].float(),
                                        obs_tensors[0][:, [0]].float(),
                                        nonkeypose_traj,
                                        keypose_traj,
                                        save=True,
                                        rotation_param="quat_from_query",
                                        rand_inds=None)

    return episode



def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="plate",  help="Input task name.")
    
    args = parser.parse_args()
    # bag_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".bag"
    # traj_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".npy"

    cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044 , -0.63127772, 0.64917582, -0.31329509, 0.28619116])
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

    resized_img_size = (256,256)
    original_image_size = (1080, 1920) #(h,)
    # resized_intrinsic = o3d.camera.PinholeCameraIntrinsic( 256., 25, 80., 734.1779174804688*scale_y, 993.6226806640625*scale_x, 551.8895874023438*scale_y)
    fxfy = 256.0
    resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)
    resized_intrinsic_np = np.array([
        [fxfy, 0., 128.0],
        [0. ,fxfy,  128.0],
        [0., 0., 1.0]
    ])

    # bound_box = np.array( [ [0.0, 0.8], [ -0.4 , 0.4], [ -0.2 , 0.4] ] )
    bound_box = np.array( [ [0.05, 0.65], [ -0.5 , 0.5], [ -0.1 , 0.6] ] )
    task_name = args.task 
    print("task_name: ", task_name)


    # OPENESS_TH = 0.35  # close pen task
    if(task_name == "close_pen"):
        OPENESS_TH = 0.35  # pick_up_plate

    if(task_name == "pick_up_plate"):
        OPENESS_TH = 0.12  # pick_up_plate
    
    if(task_name == "pouring_into_bowl"):        
        OPENESS_TH = 0.65  # pouring into bowl

    if(task_name == "put_block_into_bowl"):        
        OPENESS_TH = 0.54  # put block into bowl
        BUFFER_SIZE = 2
        # STOP_TH = 0.1

    if(task_name == "stack_block"):        
        OPENESS_TH = 0.54  # stack block

    processed_data_dir = "./processed_bimanual_keypose"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    
    dir_path = './' + task_name + '/'

    save_data_dir = processed_data_dir + '/' + task_name
    if ( os.path.isdir(save_data_dir) == False ):
        os.mkdir(save_data_dir)
        
   
    file = str(args.data_index) + ".npy"
    print("processing: ", dir_path+file)
    data = np.load(dir_path+file, allow_pickle = True)

    left_bias = get_transform(   [ -0.01, 0.365, -0.0 ,0., 0.,0.02617695, 0.99965732] )
    left_tip_bias = get_transform( [-0.028, 0.01, 0.01,      0., 0., 0., 1.] )

    right_bias = get_transform(   [ 0.01, -0.315, 0.00, 0., 0., 0., 1.0] )
    right_tip_bias = get_transform( [-0.035, 0.01, -0.008,      0., 0., 0., 1.] )
    
    episode = process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_img_size, bound_box, left_bias, left_tip_bias, right_bias, right_tip_bias)
    
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
