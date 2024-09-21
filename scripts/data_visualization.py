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

def get_cube_corners( bound_box ):
    corners = []
    corners.append( [ bound_box[0][0], bound_box[1][0], bound_box[2][0] ])
    corners.append( [ bound_box[0][0], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][0], bound_box[2][0] ])

    return corners

def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    # parser.add_argument("-b", "--bag_in", default="./data/yellow_handle_mug.bag",  help="Input ROS bag name.")
    parser.add_argument("-t", "--task_dir", default="./",  help="Input ROS bag name.")
    parser.add_argument("-d", "--data_id", default="left_verify", help="data idx")
    args = parser.parse_args()
    
    # task_dir = "./play_around"
    task_dir = args.task_dir
    data_id = args.data_id
    print("processing data: ", data_id)
    data = np.load( task_dir + "/" + data_id + ".npy", allow_pickle = True)

    make_video = True

    cam_extrinsic = get_transform( [-0.1393031, 0.0539, 0.43911375], [-0.61860094, 0.66385477, -0.31162288, 0.2819945])
    cam_intrinsic_np = np.array([
        [734.1779174804688, 0., 993.6226806640625],
        [0. ,734.1779174804688,  551.8895874023438],
        [0., 0., 1.0]
    ])

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

    video_images = []

    max_diff = 0.065
    left_min_joint = 0.638
    left_max_joint = 1.626

    right_min_joint = 0.625
    right_max_joint = 1.610


    for idx, point in enumerate( data, 0 ):

        if(idx < 100):
            continue
        if(idx % 20 != 0):
            continue

        bgr = point['bgr']
        rgb = bgr[...,::-1].copy()

        depth = point['depth']
        # depth = depth.reshape(-1,3)
        # print("depth: ", depth.shape)

        # rgb = rgb.reshape(-1,3)
        # rgb = rgb.astype(float)
        # bgr = bgr.reshape(-1,3)
        # print("rgb: ", rgb.shape)
        # pcd = o3d.geometry.PointCloud()

        im_color = o3d.geometry.Image(rgb)
        im_depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        
        final_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                o3d_intrinsic
            )
        final_pcd = final_pcd.transform(cam_extrinsic)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        left_transform = get_transform(point['left_ee'][0:3], point['left_ee'][3:7] ) # xyz, xyzw(quat)
        right_transform = get_transform(point['right_ee'][0:3], point['right_ee'][3:7] )  # right hand palm
      
        # play set
        left_transform = left_transform @ get_transform(   [-0.05, 0.005, 0.00], [0., 0., 0., 1.] )

        right_transform = right_transform @ get_transform( [-0.05, -0.005, -0.005], [0., 0., 0., 1.] )

        # left_transform = left_transform @ get_transform( [ -0.075, 0.005,  -0.005], [0., 0., 0., 1.] )
        # right_transform = right_transform @ get_transform( [-0.05, 0.005, -0.005], [0., 0., 0., 1.] )

 
        assigned_color = np.array([0,0,255])
        point_3d = np.array( [ left_transform[0][3], left_transform[1][3], left_transform[2][3] ])
        bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)

        point_3d = np.array( [ right_transform[0][3], right_transform[1][3], right_transform[2][3] ])
        bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)

        # visualize_pcd(final_pcd, [left_transform], [right_transform])

        # left tips
        openness = np.clip( point["left_pos"][6], left_min_joint, left_max_joint)
        left_gripper_distance = (openness - left_min_joint) / (left_max_joint - left_min_joint) * max_diff / 2.0
        left_y = left_gripper_distance
        right_y = -1*left_gripper_distance
        lh_left_tip = left_transform @ get_transform([0.09, left_y, 0.], [0., 0., 0., 1.] )
        lh_right_tip = left_transform @ get_transform([0.09, right_y, 0.0], [0., 0., 0., 1.]  )
        # print("difference: ", left_y - right_y)
        # max_diff = max(max_diff, left_y - right_y)
        # min_joint = min(min_joint, point["left_pos"][6])
        # max_joint = max(max_joint, point["left_pos"][6])
        visualize_pcd(final_pcd, [left_transform, lh_left_tip, lh_right_tip], [right_transform])
        point_3d = np.array( [ lh_left_tip[0][3], lh_left_tip[1][3], lh_left_tip[2][3] ])
        bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        point_3d = np.array( [ lh_right_tip[0][3], lh_right_tip[1][3], lh_right_tip[2][3] ])
        bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)


        # right tips
        openness = np.clip( point["right_pos"][6], right_min_joint, right_max_joint) # joint positions
        right_gripper_distance = (openness - right_min_joint) / (right_max_joint - right_min_joint) * max_diff / 2.0
        left_y = right_gripper_distance
        right_y = -1*right_gripper_distance
        rh_left_tip = right_transform @ get_transform([0.09, left_y, 0.], [0., 0., 0., 1.] )
        rh_right_tip = right_transform @ get_transform([0.09, right_y, 0.0], [0., 0., 0., 1.]  )
        # print("difference: ", left_y - right_y)
        # max_diff = max(max_diff, left_y - right_y)
        # min_joint = min(min_joint, point["left_pos"][6])
        # max_joint = max(max_joint, point["left_pos"][6])

        # visualize_pcd(final_pcd, [left_transform], [right_transform, rh_left_tip, rh_right_tip])

        point_3d = np.array( [ rh_left_tip[0][3], rh_left_tip[1][3], rh_left_tip[2][3] ])
        bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        point_3d = np.array( [ rh_right_tip[0][3], rh_right_tip[1][3], rh_right_tip[2][3] ])
        bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        
        
        # visualize_pcd(final_pcd, [left_transform, lh_left_tip, lh_right_tip], [right_transform, rh_left_tip, rh_right_tip])

        if make_video:
            video_images.append(bgr)
        
    # print("max_diff: ", max_diff)
    # print("min_joint: ", min_joint)
    # print("max_joint: ", max_joint)

    if make_video:
        video_name = 'video{}.avi'.format(data_id)
        height, width, layers = video_images[0].shape
        video = cv2.VideoWriter(video_name, 0, 15, (width,height))
        for image in video_images:
            video.write(image)
        video.release()

if __name__ == "__main__":
    main()
