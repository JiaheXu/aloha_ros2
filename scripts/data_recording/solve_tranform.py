import sys
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, '../'))
from utils import *
from math_tools import *

# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation

import yaml
import numpy as np

def load_yaml(file_dir):
    # Load the YAML file
    with open(file_dir, "r") as file:
        data = yaml.safe_load(file)

    return data

def main():
    tags = load_yaml("tags.yaml")
    tag_2_head = get_transform( np.array( tags.get("tag_2_head") ) )
    tag_2_left_cam = get_transform( np.array( tags.get("tag_2_left") ) )
    tag_2_right_cam = get_transform( np.array( tags.get("tag_2_right") ))
    # print("tag_2_head: ", tag_2_head)

    hands = load_yaml("hands.yaml")
    left_base_2_left_ee = get_transform( np.array( hands.get("left") ) )
    right_base_2_right_ee = get_transform( np.array( hands.get("right") ) )

    head = load_yaml("head.yaml")
    left_base_2_head = np.array( head.get("left") ) 
    right_base_2_head = np.array( head.get("right") )
    

    left_base_2_left_cam = left_base_2_head @ inv(tag_2_head) @ tag_2_left_cam
    left_ee_2_left_cam = inv(left_base_2_left_ee) @ left_base_2_left_cam
    print("left: ", left_ee_2_left_cam)

    right_base_2_right_cam = right_base_2_head @ inv(tag_2_head) @ tag_2_right_cam
    right_ee_2_right_cam = inv(right_base_2_right_ee) @ right_base_2_right_cam
    print("right: ", right_ee_2_right_cam)


    left_base_2_head_7d = get_7D_transform( left_base_2_head )
    right_base_2_head_7d = get_7D_transform( right_base_2_head )
    middle_point = (left_base_2_head[0:3,3] + right_base_2_head[0:3,3] ) / 2.
    print("left_quat: ", left_base_2_head_7d[3:7])
    print("right_quat: ", right_base_2_head_7d[3:7])

    middle_point_rot =  (left_base_2_head_7d[3:7] + right_base_2_head_7d[3:7] ) / 2.0
    middle_point = np.array( [ middle_point[0], middle_point[1], middle_point[2], middle_point_rot[0], middle_point_rot[1], middle_point_rot[2], middle_point_rot[3] ])
    print("7D: ", middle_point)

    root_2_head = get_transform(middle_point)
    print("root_2_head: ", root_2_head)

    head_2_left_base = inv( left_base_2_head )
    root_2_left_base = root_2_head @ head_2_left_base
    print("root_2_left_base: ", root_2_left_base)

    head_2_right_base = inv( right_base_2_head )
    root_2_right_base = root_2_head @ head_2_right_base
    print("root_2_right_base: ", root_2_right_base)


if __name__ == "__main__":
    main()



