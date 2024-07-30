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

# import rospy
# import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation

class Projector:
    def __init__(self, cloud, label = None) -> None:
        self.cloud = cloud
        self.points = np.asarray(cloud.points)
        self.colors = np.asarray(cloud.colors)
        self.n = len(self.points)
        self.label = label

    # intri 3x3, extr 4x4
    def project_to_rgbd(self,
                        width,
                        height,
                        intrinsic,
                        extrinsic,
                        depth_scale,
                        depth_max
                        ):
        depth = 10.0*np.ones((height, width), dtype = float)
        depth_uint = np.zeros((height, width), dtype=np.uint16)
        color = np.zeros((height, width, 3), dtype=np.uint8)
        xyz =  np.full((height, width, 3), np.nan)


        for i in range(0, self.n):
            point4d = np.append(self.points[i], 1)
            new_point4d = np.matmul(extrinsic, point4d)
            point3d = new_point4d[:-1]
            zc = point3d[2]
            new_point3d = np.matmul(intrinsic, point3d)
            new_point3d = new_point3d/new_point3d[2]
            u = int(round(new_point3d[0]))
            v = int(round(new_point3d[1]))

            # Fixed u, v checks. u should be checked for width
            if (u < 0 or u > width - 1 or v < 0 or v > height - 1 or zc <= 0.0 or zc > depth_max):
                continue
            if(zc > depth[v][u]):
                continue

            depth[v][u] = zc
            depth_uint[v, u] = zc * 1000
            xyz[v,u,:] = self.points[i]
            color[v, u, :] = self.colors[i] * 255

        im_color = o3d.geometry.Image(color)
        im_depth = o3d.geometry.Image(depth_uint)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        # return rgbd
        return color, depth, xyz, rgbd


def get_init_trans(src, dst):
    if(src.shape[0] == 3):
        src = np.transpose(src)
    if(dst.shape[0] == 3):
        dst = np.transpose(dst)
    trans = np.mean(src, axis = 0) - np.mean(dst, axis = 0)
    trans = trans.reshape(3,1)
    return trans



    
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


def get_transform( trans, quat):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

def cropping(xyz, rgb, bound_box, label = None):

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    valid_idx = np.where( (x>=bound_box[0][0]) & (x <=bound_box[0][1]) & (y>=bound_box[1][0]) & (y<=bound_box[1][1]) & (z>=bound_box[2][0]) & (z<=bound_box[2][1]) )
    valid_xyz = xyz[valid_idx]
    valid_rgb = rgb[valid_idx]
    valid_label = None
    if(label is not None):
        valid_label = label[valid_idx]
            
    valid_pcd = o3d.geometry.PointCloud()
    valid_pcd.points = o3d.utility.Vector3dVector( valid_xyz)
    if(np.max(valid_rgb) > 1.0):
        valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb/255.0 )
    else:
        valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb )
    return valid_xyz, valid_rgb, valid_label, valid_pcd


def get_delta_transform(A, B): # A = delta @ B
    delta_trans = get_transform(A[0:3], A[3:7]) @ inv( get_transform(B[0:3], B[3:7] )) 
    delat_rot = Rotation.from_matrix(delta_trans[:3,:3])
    delta_quat = delat_rot.as_quat()
    delta_openess = left_trajectory[idx][-1] - left_trajectory[idx-1][-1]
    return 

def visualize_pcd(pcd, left, right):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=coor_frame.get_center())
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    left_mesh = copy.deepcopy(mesh).transform(left)
    left_mesh.scale(0.1, center=left_mesh.get_center())
    right_mesh = copy.deepcopy(mesh).transform(right)
    right_mesh.scale(0.1, center=right_mesh.get_center())
    vis.add_geometry(left_mesh)
    vis.add_geometry(right_mesh)
    # view_ctl.set_up([-0.4, 0.0, 1.0])
    # view_ctl.set_front([-4.02516493e-01, 3.62146675e-01, 8.40731978e-01])
    # view_ctl.set_lookat([0.0 ,0.0 ,0.0])
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    # view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def visualize_bimanual_traj(pcd, left_transforms, right_transforms):
       
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=coor_frame.get_center())
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    object_pcds = [pcd]
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coor_frame.scale(0.2, center=coor_frame.get_center())
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    for left in left_transforms:
        left_mesh = copy.deepcopy(mesh).transform(left)
        left_mesh.scale(0.1, center=left_mesh.get_center())
        vis.add_geometry(left_mesh)

    for right in right_transforms:
        right_mesh = copy.deepcopy(mesh).transform(right)
        right_mesh.scale(0.1, center=right_mesh.get_center())
        vis.add_geometry(right_mesh)
    # view_ctl.set_up([-0.4, 0.0, 1.0])
    # view_ctl.set_front([-4.02516493e-01, 3.62146675e-01, 8.40731978e-01])
    # view_ctl.set_lookat([0.0 ,0.0 ,0.0])
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    # view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def main():
    


    cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044], [-0.63127772, 0.64917582, -0.31329509, 0.28619116])
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

    img_size = (256,256)
    # resized_intrinsic = o3d.camera.PinholeCameraIntrinsic( 256., 25, 80., 734.1779174804688*scale_y, 993.6226806640625*scale_x, 551.8895874023438*scale_y)
    fxfy = 256.0
    resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)
    resized_intrinsic_np = np.array([
        [fxfy, 0., 128.0],
        [0. ,fxfy,  128.0],
        [0., 0., 1.0]
    ])

    bound_box = np.array( [ [0.0, 0.8], [ -0.4 , 0.4], [ -0.2 , 0.4] ] )
    tasks = ["duck_in_bowls+0"]
    processed_data_dir = "./processed"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    for task_idx, task_name in enumerate(tasks, 0):
        dir_path = './' + task_name + '/'
        res = []
        save_data_dir = processed_data_dir + '/' + task_name
        if ( os.path.isdir(save_data_dir) == False ):
            os.mkdir(save_data_dir)

        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                res.append(path)

        for data_idx, file in enumerate(res):
            data = np.load(dir_path+file, allow_pickle = True)
            # for point in data:
            point = data[0] # only read the first one
            bgr = point['bgr']
            rgb = bgr[...,::-1].copy()
            depth = point['depth']
            im_color = o3d.geometry.Image(rgb)
            im_depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                im_color, im_depth, depth_scale=1, depth_trunc=2, convert_rgb_to_intensity=False)
            original_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd,
                    o3d_intrinsic
                    # resized_intrinsic
                )
            original_pcd = original_pcd.transform(cam_extrinsic)
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            xyz = np.array(original_pcd.points)
            rgb = np.array(original_pcd.colors)
            valid_xyz, valid_rgb, valid_label, cropped_pcd = cropping( xyz, rgb, bound_box )
            
            p = Projector(cropped_pcd)

            rgb, depth, xyz, rgbd = p.project_to_rgbd(256, 256, resized_intrinsic_np, inv(cam_extrinsic), 1000,10)
            resized_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd,
                    # o3d_intrinsic
                    resized_intrinsic_o3d,
                    # remove_nan_points=False
                )
            # print(resized_pcd)
            resized_pcd.transform( cam_extrinsic )
            resized_img_data = im.fromarray(rgb)
            resized_img_data.save('cam_img.png')

            # verify xyz and depth value
            # valid_pcd = o3d.geometry.PointCloud()
            # xyz = xyz.reshape(-1,3)
            # rgb = (rgb/255.0).reshape(-1,3)
            # x = xyz[:,0]
            # valid_mask = np.where( np.logical_not(np.isnan(x) ) )
            # valid_pcd.points = o3d.utility.Vector3dVector( xyz[valid_mask] )
            # valid_pcd.colors = o3d.utility.Vector3dVector( rgb[valid_mask] )
            # o3d.visualization.draw_geometries([valid_pcd])

            resized_img_data = np.transpose(rgb, (2, 0, 1) ).astype(float)
            # print("resized_img_data: ", resized_img_data.shape)
            resized_xyz = np.transpose(xyz, (2, 0, 1) ).astype(float)
            # print("resized_xyz: ", resized_xyz.shape)
            n_cam = 1
            obs = np.zeros( (n_cam, 2, 3, 256, 256) )
            obs[0][0] = resized_img_data
            obs[0][1] = resized_xyz

            camera_dicts = []
            left_transforms = []
            right_transforms = []
            
            frame_ids = [0] # for now, only use the observation in the beginning

            # frame_ids = [i for i in range(len(data))]
            # print("frame_id: ", frame_ids)

            # gripper end val is around 0.6 ~ 1.6
            left_gripper_max = 0.0
            left_gripper_min = 2.0

            right_gripper_max = 0.0
            right_gripper_min = 2.0


            left_trajectory = []
            right_trajectory = []
            delta_left_trajectory = []
            delta_right_trajectory = []

            for point in data:
                left_gripper_max = max(left_gripper_max, point["left_pos"][6])
                left_gripper_min = min(left_gripper_min, point["left_pos"][6])
                right_gripper_max = max(right_gripper_max, point["right_pos"][6])
                right_gripper_min = min(right_gripper_min, point["right_pos"][6])

            for point in data:
                left_transform = get_transform(point['left_ee'][0:3], point['left_ee'][3:7] )
                left_transform = left_transform @ get_transform( [-0.02, -0.035, -0.045], [0., 0., 0., 1.] )
                left_rot = Rotation.from_matrix(left_transform[:3,:3])
                left_quat = left_rot.as_quat()
                left_openess = ( float(point["left_pos"][6]) - left_gripper_min ) / (left_gripper_max - left_gripper_min )
                left_trajectory.append(np.array( [left_transform[0][3], left_transform[1][3], left_transform[2][3], left_quat[0], left_quat[1], left_quat[2], left_quat[3], left_openess ] ))

                right_transform = get_transform(point['right_ee'][0:3], point['right_ee'][3:7] )
                right_transform = right_transform @ get_transform( [-0.005, -0.03, -0.036], [0., 0., 0., 1.] )
                right_rot = Rotation.from_matrix(right_transform[:3,:3])
                right_quat = right_rot.as_quat()
                right_openess = ( float(point["right_pos"][6]) - right_gripper_min ) / (right_gripper_max - right_gripper_min )
                right_trajectory.append(np.array( [right_transform[0][3], right_transform[1][3], right_transform[2][3], right_quat[0], right_quat[1], right_quat[2], right_quat[3], right_openess] ))  
                
            for idx, trans in enumerate(right_trajectory, 0):
                if(idx == 0):
                    continue
                delta_trans = get_transform(right_trajectory[idx][0:3], right_trajectory[idx][3:7]) @ inv( get_transform(right_trajectory[idx-1][0:3], right_trajectory[idx-1][3:7] ) )
                delat_rot = Rotation.from_matrix(delta_trans[:3,:3])
                delta_quat = delat_rot.as_quat()
                delta_openess = right_trajectory[idx][-1] - right_trajectory[idx-1][-1]
                action = np.array( [delta_trans[0,3], delta_trans[1,3], delta_trans[2,3], delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3], delta_openess] )
                delta_right_trajectory.append( action )

            delta_transform = get_transform(right_trajectory[-1][0:3], right_trajectory[-1][3:7]) @ inv( get_transform(right_trajectory[0][0:3], right_trajectory[0][3:7] ) )
            delat_rot = Rotation.from_matrix(delta_transform[:3,:3])
            delta_quat = delat_rot.as_quat()
            delta_openess = right_trajectory[-1][-1] - right_trajectory[0][-1]
            
            action = np.array( [delta_transform[0,3], delta_transform[1,3], delta_transform[2,3], delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3], delta_openess] )
            action = action.reshape(1,8)

            gripper = copy.deepcopy( right_trajectory[0])
            gripper = gripper.reshape(1,8)
            
            trajectories = np.array(delta_right_trajectory)
            trajectories = trajectories.reshape(-1,8)
            # print("trajectories: ", trajectories.shape)
            episode = []
            episode.append(frame_ids) # 0

            obs_tensors = [ torch.from_numpy(obs) ]
            episode.append(obs_tensors) # 1
           
            action_tensor =  [ torch.from_numpy(action) ]
            episode.append(action_tensor) # 2

            episode.append(camera_dicts) # 3

            gripper_tensor = [ torch.from_numpy(gripper) ]
            episode.append(gripper_tensor) # 4

            trajectories_tensor = [ torch.from_numpy(trajectories) ]
            episode.append(trajectories_tensor) # 5

            np.save("./processed/{}/ep{}".format(task_name,data_idx), episode)
            print("finished ", task_name, " data: ", data_idx)
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