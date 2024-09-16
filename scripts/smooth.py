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

def plot_func(src, dst, dir="", idx=0, save = False):

    if(src.shape[0] != 3):
        src = np.transpose(src)
    if(dst.shape[0] != 3):
        dst = np.transpose(dst)    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")

    ax.scatter(src[0, :], src[1, :], src[2, :], c = colors[0])
    ax.scatter(dst[0, :], dst[1, :], dst[2, :], c = colors[1])
    if(save == False):
        plt.show()
    else:
        plt.savefig(dir + "/" + str(idx) + ".svg")

def get_init_trans(src, dst):
    if(src.shape[0] == 3):
        src = np.transpose(src)
    if(dst.shape[0] == 3):
        dst = np.transpose(dst)
    trans = np.mean(src, axis = 0) - np.mean(dst, axis = 0)
    trans = trans.reshape(3,1)
    return trans


def print_plot(F_reg, src, dst, dir, idx = 0,  save = False):
    rot = F_reg[0:3, 0:3]
    trans = F_reg[0:3, 3]
    
    trans = trans.reshape(3,1)
    print("rot: ", rot)
    print("trans: ", trans)
    pcd = rot @ np.transpose(dst) + trans
    plot_func( src, pcd , dir, idx = idx, save = save)
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud])

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def colored_ICP(source, target):
    voxel_radius = [0.002, 0.002, 0.002]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)


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

def get_cube_corners( bound_box ):
    corners = []
    corners.append( [ bound_box[0][0], bound_box[1][0], bound_box[2][0] ])
    corners.append( [ bound_box[0][0], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][0], bound_box[2][0] ])

    return corners

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
    for idx in range(length):
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
        print("curr_gripper: ", get_transform2(curr_gripper[0,0:7]))
        curr_pose.append( get_transform2(curr_gripper[0,0:7]) )
        curr_pose.append( get_transform2(curr_gripper[1,0:7]) )

        # print("action: ", action.shape)
        trajectory = action
        print("trajectory: ", trajectory.shape)
        # left.append( get_transform2( trajectory[-1,0,0:7]))
        # right.append( get_transform2( trajectory[-1,1,0:7]))

        # print("last goal: ", get_transform2( trajectory[-1,0,0:7]))
        print("last goal: ", get_transform2( trajectory[-1,0,0:7]))
        left_stack = []
        right_stack = []
        dist_threshold = 0.01

        for action_idx in range(trajectory.shape[0]):
            if len(left_stack) < 1:
                dist = LA.norm( trajectory[action_idx,0,0:3] - curr_gripper[0,0:3] )
            else:
                dist = LA.norm( trajectory[action_idx,0,0:3] - left_stack[-1][0:3] )
            if(dist < dist_threshold):
                continue
            left_stack.append(trajectory[action_idx,0,0:7])
            left.append( get_transform2( trajectory[action_idx,0,0:7]))
        
        for action_idx in range(trajectory.shape[0]):
            if len(right_stack) < 1:
                dist = LA.norm( trajectory[action_idx,1,0:3] - curr_gripper[1,0:3] )
            else:
                dist = LA.norm( trajectory[action_idx,1,0:3] - right_stack[-1][0:3] )
            if(dist < dist_threshold):
                continue
            right_stack.append(trajectory[action_idx,1,0:7])
            right.append( get_transform2( trajectory[action_idx,1,0:7]))
        print("left: ", len(left))
        print("right: ", len(right))
        visualize_pcd(pcd, left, right, curr_pose)


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
