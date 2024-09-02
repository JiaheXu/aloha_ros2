#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script contains 2 functions for converting cloud format between Open3D and ROS:   
* convertCloudFromOpen3dToRos  
* convertCloudFromRosToOpen3d
'''
# adopted from https://github.com/felixchenfy/open3d_ros_pointcloud_conversion
import open3d
import numpy as np
from ctypes import * # convert float to uint32

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
convert_rgbaUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff), (rgb_uint32 & 0xff000000)>>24
)


# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

def convertCloudFromRosToOpen3d(ros_cloud, bound_box = None):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    label = None
    # Set open3d_cloud
    # print("field_names: ", field_names)
    # if "rgba" in field_names:
    IDX_RGB_IN_FIELD=3 # x, y, z, rgb
    
    # Get xyz
    xyz = np.array([(x,y,z) for x,y,z,rgba in cloud_data ]) # (why cannot put this line below rgb?)

    # Get rgba
    rgba = [convert_rgbaUint32_to_tuple(rgba) for x,y,z,rgba in cloud_data ]
    
    # print("a: ",np.array(rgba)[:,3])
    rgba = np.array(rgba)
    rgb = np.array(rgba)[:,0:3]

    label = 255 - np.array(rgba)[:,3]
    
    open3d_cloud.points = open3d.utility.Vector3dVector( xyz )
    open3d_cloud.colors = open3d.utility.Vector3dVector( rgb/255.0 )


    segmented_pointclouds = []
    # 0 for background, 1 for active object, 2 for passive object
    for class_idx in range(3):
        object_idxs = None
        if(bound_box is not None):
            x = xyz[:,0]
            y = xyz[:,1]
            z = xyz[:,2]
            object_idxs = np.where( (label == class_idx)& (x>=bound_box[0][0]) & (x <=bound_box[0][1]) & (y>=bound_box[1][0]) & (y<=bound_box[1][1]) & (z>=bound_box[2][0]) & (z<=bound_box[2][1])  )
        else:
            object_idxs = np.where(label == class_idx)
        object_xyz = xyz[object_idxs]
        object_rgb = rgb[object_idxs]
        object_pcd = open3d.geometry.PointCloud()
        object_pcd.points = open3d.utility.Vector3dVector( object_xyz)
        object_pcd.colors = open3d.utility.Vector3dVector( object_rgb/255.0 )
        segmented_pointclouds.append(object_pcd)
    


    # return
    return  xyz, rgb, label, open3d_cloud, segmented_pointclouds