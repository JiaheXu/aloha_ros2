#3dda data:
# bgrd image 1920*1080 need post process
# start from everywhere
# episode length 10~100 steps
# joints position & velocity




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

import math
from geometry_msgs.msg import Twist

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Joy
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32,Header
import numpy as np 
import time
np.set_printoptions(suppress=True,precision=3)

from numpy.linalg import inv
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2 
from rclpy.qos import QoSProfile
from rclpy.clock import Clock
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
import open3d as o3d

from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
import sensor_msgs_py.point_cloud2 as pc2
from ctypes import * # convert float to uint32
# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
# from utils.ros2_o3d_utils import *
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

class DataCollector(Node):

    def __init__(self):
        super().__init__('aloha_3dda_data_collection_node')
        # print("in init")
        # Declare and acquire `target_frame` parameter
        self.left_hand_frame = "follower_left/ee_gripper_link"
        self.right_hand_frame = "follower_right/ee_gripper_link"

        self.left_hand_gripper_frames = ["follower_left/left_finger_link", "follower_left/right_finger_link"]
        self.right_hand_gripper_frames = ["follower_right/left_finger_link", "follower_right/right_finger_link"]
        self.left_base_frame = "follower_left/base_link"
        self.right_base_frame = "follower_right/base_link"
        # self.left_base_frame = "world"
        # self.right_base_frame = "world"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.left_hand_transform = TransformStamped()
        self.right_hand_transform = TransformStamped()

        self.lh_gripper_left_transform = TransformStamped()
        self.lh_gripper_right_transform = TransformStamped()
        self.rh_gripper_left_transform = TransformStamped()
        self.rh_gripper_right_transform = TransformStamped()

        #axes
        self.left_joystick_x = 0
        self.left_joystick_y = 1
        
        self.right_joystick_x = 2
        self.right_joystick_y = 3
        self.leftside_left_right_arrow = 4
        self.leftside_up_down_arrow = 5

        self.max_idx = 5
        
        # button mapping for wireless controller
        self.A_button = 0
        self.B_button = 1
        self.X_button = 2
        self.Y_button = 3

        # states
        self.recording = False

        self.last_data_time = time.time()
        # data
        self.current_stack = []
        self.current_keypose_stack = []

        self.success_stop_pressed_last = False
        self.failure_stop_pressed_last = False
        self.start_recording_pressed_last = False
        self.keypose_stop_pressed_last = False
        self.add_keypose = False

        # Call on_timer function every second
        self.timer_period = 2.0
        # self.timer = self.create_timer( self.timer_period, self.on_timer )
        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joyCallback,1)
        self.br = CvBridge()
        # self.subscription = self.create_subscription(Image, "/camera_1/left_image", self.img_callback, 1)

        # Todo: use yaml files
        # self.cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044, -0.63127772, 0.64917582, -0.31329509, 0.28619116])
        # self.head_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)
        self.head_cam = load_yaml( "../config/head.yaml" )
        self.head_cam_intrinsic_np = np.array( self.head_cam.get("intrinsic") )
        o3d_data = self.head_cam.get("intrinsic_o3d")[0]
        # print("o3d_data: ", o3d_data)
        self.head_cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])

        self.left_cam = load_yaml( "../config/left_wrist.yaml" )
        self.left_cam_intrinsic_np = np.array( self.left_cam.get("intrinsic") )
        o3d_data = self.left_cam.get("intrinsic_o3d")[0]
        self.left_cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])

        self.right_cam = load_yaml( "../config/right_wrist.yaml" )
        self.right_cam_intrinsic_np = np.array( self.right_cam.get("intrinsic") )
        o3d_data = self.right_cam.get("intrinsic_o3d")[0]
        self.right_cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])

        self.resized_image_size = (256,256)
        self.first_frame_image_size = (512, 512)
        self.first_frame_data = {}
        
        # self.original_image_size = (1080, 1920) #(h,)
        fxfy = 256.0
        
        self.resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)

        self.first_frame_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(512, 512,   512., 512.,   256., 256.)
        
        self.resized_intrinsic_np = np.array([
            [fxfy, 0., 128.0],
            [0. ,fxfy,  128.0],
            [0., 0., 1.0]
        ])    
        
        self.first_frame = True
        queue_size = 1000
        max_delay = 0.05 # 50ms
        self.time_diff = 0.10

        self.tf_broadcaster = TransformBroadcaster(self)

        self.bgr_sub = Subscriber(self, Image, "/camera_1/left_image")
        self.depth_sub = Subscriber(self, Image, "/camera_1/depth")
        #self.bgr_sub = Subscriber(self, Image, "/zed/zed_node/left/image_rect_color")
        #self.depth_sub = Subscriber(self, Image, "/zed/zed_node/depth/depth_registered" )

        self.left_rgb_sub = Subscriber(self, Image, "/cam1/cam1/color/image_rect_raw")
        self.left_depth_sub = Subscriber(self, Image, "/cam1/cam1/depth/image_rect_raw")

        self.right_rgb_sub = Subscriber(self, Image, "/cam2/cam2/color/image_rect_raw")
        self.right_depth_sub = Subscriber(self, Image, "/cam2/cam2/depth/image_rect_raw")

        self.left_hand_sub = Subscriber(self, JointState, "/follower_left/joint_states")
        self.right_hand_sub = Subscriber(self, JointState, "/follower_right/joint_states")
        self.left_controller_sub = Subscriber(self, JointState, "/leader_left/joint_states")
        self.right_controller_sub = Subscriber(self, JointState, "/leader_right/joint_states")

        self.time_sync = ApproximateTimeSynchronizer([
            self.bgr_sub, self.depth_sub, 
            self.left_rgb_sub, self.left_depth_sub, 
            self.right_rgb_sub, self.right_depth_sub, 
            self.left_hand_sub, self.right_hand_sub,
            self.left_controller_sub, self.right_controller_sub
            ]
            ,queue_size, max_delay)


        self.time_sync.registerCallback(self.SyncCallback)

        self.pcd_publisher = self.create_publisher(PointCloud2, "rgb_pcd", 1)

        # timer_period = 0.01 #100hz
        # self.timer = self.create_timer(timer_period, self.publish_tf)


    def SyncCallback(self, bgr, depth, left_rgb, left_depth, right_rgb, right_depth, left_hand_joints, right_hand_joints, left_controller_joints, right_controller_joints):
        
        # print("bgr timestamp:", bgr.header.stamp)
        # print("depth timestamp: ", depth.header.stamp)

        # print("left_rgb timestamp:", left_rgb.header.stamp)
        # print("left_depth timestamp: ", left_depth.header.stamp)

        # print("right_rgb timestamp:", right_rgb.header.stamp)
        # print("right_depth timestamp: ", right_depth.header.stamp)

        # print("left_hand_joints: ", left_hand_joints.header.stamp)
        # print("right_hand_joints: ", right_hand_joints.header.stamp)

        # print("left_hand_joints: ", left_controller_joints.header.stamp)
        # print("right_hand_joints: ", right_controller_joints.header.stamp)

        data_time = time.time()

        if(data_time - self.last_data_time < self.time_diff):
            return
        
        if( self.recording == False ):
            return
        print("in call back !!! ")
        print("time diff: ", data_time - self.last_data_time )

        self.last_data_time = data_time

        current_state = {}
        current_state["timestamp"] = data_time

        # start = time.time()

        head_bgr = np.array(self.br.imgmsg_to_cv2(bgr))[:,:,:3]
        head_rgb = head_bgr[...,::-1].copy()
        head_depth = np.array(self.br.imgmsg_to_cv2(depth, desired_encoding="mono16"))
        head_resized_rgb, head_resized_depth = transfer_camera_param(head_rgb, head_depth, self.head_cam_intrinsic_np, self.resized_intrinsic_o3d.intrinsic_matrix, self.resized_image_size )
        current_state["head_rgb"] = head_resized_rgb
        current_state["head_depth"] = head_resized_depth
        if ( self.first_frame == True):
            self.first_frame = False
            first_frame_resized_rgb, first_frame_resized_depth = transfer_camera_param(head_rgb, head_depth, self.head_cam_intrinsic_np, self.first_frame_intrinsic_o3d.intrinsic_matrix, self.first_frame_image_size )
            first_frame_data = {}
            first_frame_data["rgb"] = first_frame_resized_rgb
            first_frame_data["depth"] = first_frame_resized_depth
            self.first_frame_data = first_frame_data

        left_rgb = np.array(self.br.imgmsg_to_cv2(left_rgb))[:,:,:3]
        left_depth = np.array(self.br.imgmsg_to_cv2(left_depth, desired_encoding="16UC1"))
        left_resized_rgb, left_resized_depth = transfer_camera_param(left_rgb, left_depth, self.left_cam_intrinsic_np, self.resized_intrinsic_o3d.intrinsic_matrix, self.resized_image_size )
        current_state["left_rgb"] = left_resized_rgb
        current_state["left_depth"] = left_resized_depth

        right_rgb = np.array(self.br.imgmsg_to_cv2(right_rgb))[:,:,:3]
        right_depth = np.array(self.br.imgmsg_to_cv2(right_depth, desired_encoding="16UC1"))
        right_resized_rgb, right_resized_depth = transfer_camera_param(right_rgb, left_depth, self.right_cam_intrinsic_np, self.resized_intrinsic_o3d.intrinsic_matrix, self.resized_image_size )
        current_state["right_rgb"] = right_resized_rgb
        current_state["right_depth"] = right_resized_depth

        # save_np_image(head_resized_rgb, "head.jpg")
        # save_np_image(left_resized_rgb, "left.jpg")
        # save_np_image(right_resized_rgb, "right.jpg")
        # end = time.time()
        # print("end - start: ", end - start)

        # save_np_image(current_state["head_rgb"], file_name = "head.jpg")
        # save_np_image(current_state["left_rgb"], file_name = "left.jpg")
        # save_np_image(current_state["right_rgb"] , file_name = "right.jpg")
        # print("h depth min:", np.min(current_state["head_depth"]))
        # print("h depth max:", np.max(current_state["head_depth"]))
        # print("l depth min:", np.min(current_state["left_depth"]))
        # print("l depth max:", np.max(current_state["left_depth"]))
        # print("depth mid:", current_state["depth"][540, 920] )

        current_state["left_pos"] = np.array(left_hand_joints.position)
        current_state["left_vel"] = np.array(left_hand_joints.velocity) 
        current_state["right_pos"] = np.array(right_hand_joints.position) 
        current_state["right_vel"] = np.array(right_hand_joints.velocity)

        current_state["left_controller_pos"] = np.array(left_controller_joints.position)  
        current_state["right_controller_pos"] = np.array(right_controller_joints.position)

        # current_state["lh_grippers"] = np.array(lh_grippers) 
        # current_state["rh_grippers"] = np.array(rh_grippers) 

        print("added a point")
        self.current_stack.append(current_state)

        if(self.add_keypose):
            self.current_keypose_stack.append(current_state)
            self.add_keypose = False

        if( len(self.current_stack) > 1000):
            self.current_stack.clear()
            self.current_keypose_stack.clear()

    def save_data(self):
        now = time.time()
        print("saved ", len(self.current_stack) , "data !!!")
        print("saved ", len(self.current_stack) , "data !!!")
        print("saved ", len(self.current_stack) , "data !!!")
        print("saved ", len(self.current_keypose_stack) , "data !!!")
        print("saved ", len(self.current_keypose_stack) , "data !!!")
        print("saved ", len(self.current_keypose_stack) , "data !!!")
        np.save( "./first_frame/" + str(now) + "_first_frame", self.first_frame_data)
        np.save( "./traj/" + str(now) + "_trajectory", self.current_stack)
        np.save( "./keypose/" + str(now) + "_keypose", self.current_keypose_stack)

    def clean_data(self):
        self.current_stack.clear()
        self.current_keypose_stack.clear()

    def episode_end(self, success_flag):
        if( success_flag == True):
            self.save_data()
        self.clean_data()

    def on_timer(self):
        # t.transform.translation.x
        try:
            self.left_hand_transform = self.tf_buffer.lookup_transform(
                self.left_hand_frame,
                self.left_base_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.left_base_frame} to {self.left_hand_frame}: {ex}'
            )
            return

        try:
            self.right_hand_transform = self.tf_buffer.lookup_transform(
                self.right_hand_frame,
                self.right_base_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.right_base_frame} to {self.right_hand_frame}: {ex}'
            )
            return
        # return   
        # print("updated trans:")
        # print("left hand: ", self.left_hand_transform)
        # print("right hand: ", self.right_hand_transform)

    def joyCallback(self, msg):

        start_recording_pressed = msg.buttons[self.Y_button]
        success_stop_pressed = msg.buttons[self.A_button]
        failure_stop_pressed = msg.buttons[self.B_button]
        keypose_pressed = msg.buttons[self.X_button]

        if( (start_recording_pressed == True) and (self.start_recording_pressed_last == False) ):
            if( self.recording == False ):
                self.recording = True
                self.add_keypose = True
                self.first_frame = True
                self.get_logger().info('start recording!!!')
                # self.get_logger().info('start recording!!!')
            else:
                self.recording = True
                self.episode_end(False)
                self.get_logger().info('start recording!!!')
                # self.get_logger().info('start recording!!!')                

        if( (success_stop_pressed == True) and (self.success_stop_pressed_last == False) ):
            if( self.recording == True ):
                self.recording = False
                self.episode_end(True)
                self.get_logger().info('episode succeed!!!')
                # self.get_logger().info('episode succeed!!!')

        if( (failure_stop_pressed == True) and (self.failure_stop_pressed_last == False) ):
            if( self.recording == True ):
                self.recording = False
                self.episode_end(False)
                self.get_logger().info('episode failed!!!')
                # self.get_logger().info('episode failed!!!')

        if( (keypose_pressed == True) and (self.keypose_pressed_last == False) ):
            if( self.recording == True ):
                self.add_keypose = True
                self.get_logger().info('add keypose!!!')

        self.start_recording_pressed_last = start_recording_pressed
        self.success_stop_pressed_last = success_stop_pressed           
        self.failure_stop_pressed_last = failure_stop_pressed
        self.keypose_pressed_last = keypose_pressed

def main():

    rclpy.init()
    node = DataCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
if __name__ == '__main__':
    main()
