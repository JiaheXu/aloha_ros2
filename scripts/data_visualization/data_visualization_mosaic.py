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
# from ..utils import *

def get_transform( t_7d ):
    t = np.eye(4)
    trans = t_7d[0:3]
    quat = t_7d[3:7]
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

def visualize_pcd(pcd, left = None, right = None):
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
    
    # left_mesh.scale(0.1, center=(left[0][3], left[1][3], left[2][3]))

    # right_mesh.scale(0.1, center=(right[0][3], right[1][3], right[2][3]))
    
    if left is not None:
        for trans in left:
            left_mesh = copy.deepcopy(mesh).transform(trans)
            vis.add_geometry(left_mesh)

    if right is not None:
        for trans in right:
            right_mesh = copy.deepcopy(mesh).transform(trans)
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

def project_color( point_3d, color, image, extrinsic, intrinsic):
                
    point4d = np.append(point_3d, 1)
    new_point4d = np.matmul(extrinsic, point4d)
    point3d = new_point4d[:-1]
    zc = point3d[2]
    new_point3d = np.matmul(intrinsic, point3d)
    new_point3d = new_point3d/new_point3d[2]
    u = int(round(new_point3d[0]))
    v = int(round(new_point3d[1]))
    if(v<0 or v>= image.shape[0] or u<0 or u>= image.shape[1]):
        return image
    radius = 3
    image[max(0, v-radius): min(v+radius, image.shape[0]), max(0, u-radius): min(u+radius, image.shape[1]) ] = color
    # print("updated")
    return image

def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    # parser.add_argument("-b", "--bag_in", default="./data/yellow_handle_mug.bag",  help="Input ROS bag name.")
    parser.add_argument("-t", "--task_dir", default="close_marker",  help="Input ROS bag name.")
    parser.add_argument("-d", "--data_id", default="1", help="data idx")
    args = parser.parse_args()
    
    # task_dir = "./play_around"
    task_dir = args.task_dir
    data_id = args.data_id

    data_dir = "../raw_data"
    processed_data_dir = "../processed_bimanual_keypose"

    print("processing data: ", task_dir + "/" + data_id + ".npy" )
    
    data = np.load( data_dir + "/" + task_dir + "/" + data_id + ".npy", allow_pickle = True)
    processed_data = np.load( processed_data_dir + "/" + task_dir + "+0" + "/" + "ep" + data_id + ".npy", allow_pickle = True)
    keyposes_idx = processed_data[0]
    print("keypose idx: ", processed_data[0] )

    make_video = True

    cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044 , -0.63127772, 0.64917582, -0.31329509, 0.28619116] )
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

    buff = 20
    for idx, point in enumerate( data, 0 ):

        # if(idx < 180):
        #   continue
        # if(idx % 20 != 0):
        #     continue

        print("idx: ", idx)

        bgr = point['bgr']
        rgb = bgr[...,::-1].copy()
        depth = point['depth']

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

        left_transform = get_transform(point['left_ee'][0:7]) # xyz, xyzw(quat)
        right_transform = get_transform(point['right_ee'][0:7])  # right hand palm
      

        left_transform = left_transform @ get_transform(   [-0.085, 0.012, -0.005 ,0., 0., 0., 1.] )
        right_transform = right_transform @ get_transform( [-0.04, -0.005, 0.0, 0., 0., 0., 1.] )

 
        assigned_color = np.array([0,0,255])
        point_3d = np.array( [ left_transform[0][3], left_transform[1][3], left_transform[2][3] ])
        point_3d = np.array( [ right_transform[0][3], right_transform[1][3], right_transform[2][3] ])

        if( idx in keyposes_idx):
            keypose_count = buff
        


        if(keypose_count > 0):
            bgr[0:20, :, : ] = np.array([0,255,0])
            bgr[-20:, :, : ] = np.array([0,255,0])
            bgr[ : , 0:20 , : ] = np.array([0,255,0])
            bgr[ : , -20: , : ] = np.array([0,255,0])
            
            keypose_count -= 1
        
            if(keypose_count < buff - 5):
                continue
        # # left tips
        # openness = np.clip( point["left_pos"][6], left_min_joint, left_max_joint)
        # left_gripper_distance = (openness - left_min_joint) / (left_max_joint - left_min_joint) * max_diff / 2.0
        # left_y = left_gripper_distance
        # right_y = -1*left_gripper_distance
        # lh_left_tip = left_transform @ get_transform([0.087, left_y, 0, 0., 0., 0., 1.] )
        # lh_right_tip = left_transform @ get_transform([0.087, right_y, 0., 0., 0., 0., 1.]  )
        # # visualize_pcd(final_pcd, [left_transform, lh_left_tip, lh_right_tip], [right_transform])

        # # left ahnd base
        # point_3d = np.array( [ left_transform[0][3], left_transform[1][3], left_transform[2][3] ])
        # bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        # # left hand tips
        # point_3d = np.array( [ lh_left_tip[0][3], lh_left_tip[1][3], lh_left_tip[2][3] ])
        # bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        # point_3d = np.array( [ lh_right_tip[0][3], lh_right_tip[1][3], lh_right_tip[2][3] ])
        # bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)


        # # right tips
        # openness = np.clip( point["right_pos"][6], right_min_joint, right_max_joint) # joint positions
        # right_gripper_distance = (openness - right_min_joint) / (right_max_joint - right_min_joint) * max_diff / 2.0
        # left_y = right_gripper_distance
        # right_y = -1*right_gripper_distance
        # rh_left_tip = right_transform @ get_transform([0.087, left_y, 0., 0., 0., 0., 1.] )
        # rh_right_tip = right_transform @ get_transform([0.087, right_y, 0., 0., 0., 0., 1.]  )

        # # visualize_pcd(final_pcd, [left_transform], [right_transform, rh_left_tip, rh_right_tip])

        # # project right tips
        # point_3d = np.array( [ rh_left_tip[0][3], rh_left_tip[1][3], rh_left_tip[2][3] ])
        # bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        # point_3d = np.array( [ rh_right_tip[0][3], rh_right_tip[1][3], rh_right_tip[2][3] ])
        # bgr = project_color(point_3d, assigned_color, bgr, inv(cam_extrinsic), cam_intrinsic_np)
        
        # visualize_pcd(final_pcd, [left_transform, lh_left_tip, lh_right_tip], [right_transform, rh_left_tip, rh_right_tip])

        if make_video:
            video_images.append(bgr)
        

    video_dir = "./tasks"
    if ( os.path.isdir(video_dir) == False ):
        os.mkdir(video_dir)

    if make_video:
        video_name = '{}.avi'.format( task_dir )
        height, width, layers = video_images[0].shape
        video = cv2.VideoWriter(video_name, 0, 10, (width//4,height//3))
        for idx, image in enumerate( video_images ):
            if(idx%2 == 0):
                continue
            #     cv2.imwrite( str(idx)+".jpg", image)
            image = cv2.resize(image, dsize=(width//4,height//3), interpolation=cv2.INTER_CUBIC)
            video.write(image)
        video.release()

if __name__ == "__main__":
    main()
