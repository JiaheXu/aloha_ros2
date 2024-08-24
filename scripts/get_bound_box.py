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
import json
# from numpyencoder import NumpyEncoder

def main():
    

    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="plate",  help="Input task name.")
    
    args = parser.parse_args()

    task_name = args.task 
    processed_data_dir = "./processed_bimanual"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    dir_path = './' + task_name + '/'
    res = []
    save_data_dir = processed_data_dir + '/' + task_name + '/'
    if ( os.path.isdir(save_data_dir) == False ):
        os.mkdir(save_data_dir)

    print("save_data_dir: ", save_data_dir)
    for path in os.listdir(save_data_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(save_data_dir, path)):
            res.append(path)
    print("res: ", res )
    max_x = -2.0
    max_y = -2.0
    max_z = -2.0
    min_x = 2.0
    min_y = 2.0
    min_z = 2.0

    for data_idx, file in enumerate(res):
        episode = np.load(save_data_dir+file, allow_pickle = True)
        print("processing: ", save_data_dir+file,)
        trajectories = episode[5]
        for trajectory in trajectories:
            # all_x = [max_x ,trajectory[0,:,0], trajectory[1,:,0]]
            trajectory= trajectory.numpy()
            # print("trajectory: ", trajectory.shape)
            current_max_x = np.max( trajectory[:,:,0])
            current_max_y = np.max( trajectory[:,:,1])
            current_max_z = np.max( trajectory[:,:,2])
            current_min_x = np.min( trajectory[:,:,0])
            current_min_y = np.min( trajectory[:,:,1])
            current_min_z = np.min( trajectory[:,:,2])
            max_x = max(max_x, current_max_x)
            max_y = max(max_y, current_max_y)
            max_z = max(max_z, current_max_z)
            min_x = min(min_x, current_min_x)
            min_y = min(min_y, current_min_y)
            min_z = min(min_z, current_min_z)

    print("min_x: ", min_x)
    print("min_y: ", min_y)
    print("min_z: ", min_z)

    print("max_x: ", max_x)
    print("max_y: ", max_y)
    print("max_z: ", max_z)
    
    # numpy_data = np.array
    # print(json.dumps(numpy_data, cls=NumpyEncoder))


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