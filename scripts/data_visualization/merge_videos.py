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

from moviepy.editor import *

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
    
    # # Create a VideoCapture object and read from input file
    # # If the input is the camera, pass 0 instead of the video file name
    # cap = cv2.VideoCapture('close_marker.avi')
    
    # # Check if camera opened successfully
    # if (cap.isOpened()== False): 
    #     print("Error opening video stream or file")
    
    # # Read until video is completed
    # while(cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     if ret == True:
    #         print("got an image")
    #     # Break the loop
    #     else: 
    #         break
    # # When everything done, release the video capture object
    # cap.release()
    
    # # Closes all the frames
    # cv2.destroyAllWindows()
    video_files = [
        'close_marker.avi',
        'hand_over_block.avi', 
        'lift_ball.avi', 
        'open_marker.avi', 
        'open_pill_case.avi', 

        'pick_up_notebook.avi', 
        'pick_up_plate.avi', 
        'pour_into_bowl.avi', 
        'stack_blocks.avi', 
        'stack_bowls.avi', 

        'straighten_rope.avi', 
        'straighten_rope.avi'
        ]
    rows = 3
    columns = 4
    clips = []
    video_dir = './tasks'
    for r in range( rows ):
        current_column = []
        for c in range( columns ):
            current_column.append( VideoFileClip( video_dir + "/" + video_files[r*columns + c] ) )

        clips.append(current_column)

    # stacking clips
    final = clips_array(clips)
    
    # showing final clip
    final.ipython_display(width = 1920)

if __name__ == "__main__":
    main()
