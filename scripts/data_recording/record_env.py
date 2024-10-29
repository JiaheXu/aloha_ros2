#3dda data:
# bgrd image 1920*1080 need post process
# start from everywhere
# episode length 10~100 steps
# joints position & velocity


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

        self.success_stop_pressed_last = False
        self.failure_stop_pressed_last = False
        self.start_recording_pressed_last = False
        # Call on_timer function every second
        self.timer_period = 2.0
        # self.timer = self.create_timer( self.timer_period, self.on_timer )
        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joyCallback,1)
        self.br = CvBridge()
        # self.subscription = self.create_subscription(Image, "/camera_1/left_image", self.img_callback, 1)

        # Todo: use yaml files
        self.cam_extrinsic = self.get_transform( [-0.13913296, 0.053, 0.43643044, -0.63127772, 0.64917582, -0.31329509, 0.28619116])
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

        self.resized_image_size = (256,256)
        self.original_image_size = (1080, 1920) #(h,)
        fxfy = 256.0
        self.resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)
        self.resized_intrinsic_np = np.array([
            [fxfy, 0., 128.0],
            [0. ,fxfy,  128.0],
            [0., 0., 1.0]
        ])    

        queue_size = 1000
        max_delay = 0.01 #10ms
        self.time_diff = 0.05

        self.tf_broadcaster = TransformBroadcaster(self)

        self.bgr_sub = Subscriber(self, Image, "/camera_1/left_image")
        self.depth_sub = Subscriber(self, Image, "/camera_1/depth")
        #self.bgr_sub = Subscriber(self, Image, "/zed/zed_node/left/image_rect_color")
        #self.depth_sub = Subscriber(self, Image, "/zed/zed_node/depth/depth_registered" )
        self.left_hand_sub = Subscriber(self, JointState, "/follower_left/joint_states")
        self.right_hand_sub = Subscriber(self, JointState, "/follower_right/joint_states")
        self.left_controller_sub = Subscriber(self, JointState, "/leader_left/joint_states")
        self.right_controller_sub = Subscriber(self, JointState, "/leader_right/joint_states")

        self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, self.depth_sub],queue_size, max_delay)
        # self.time_sync = ApproximateTimeSynchronizer([self.bgr_sub, 
        #     self.depth_sub, 
        #     self.left_hand_sub, 
        #     self.right_hand_sub, 
        #     self.left_controller_sub, 
        #     self.right_controller_sub 
        #     ],queue_size, max_delay)

        self.time_sync.registerCallback(self.SyncCallback)

        self.pcd_publisher = self.create_publisher(PointCloud2, "rgb_pcd", 1)

        timer_period = 0.01 #100hz
        self.timer = self.create_timer(timer_period, self.publish_tf)

    def get_transform(self, transf_7D):
        trans = transf_7D[0:3]
        quat = transf_7D[3:7]
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
        t[:3, 3] = trans
        return t
        
    # Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
    def convertCloudFromOpen3dToRos(self, open3d_cloud, frame_id="world"):
        # Set "header"

        ros_time = self.get_clock().now()
        header = Header()
        header.stamp = ros_time.to_msg()
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
        # fields=FIELDS_XYZ
        # cloud_data=points
        # print("fields: ", fields)
        return pc2.create_cloud(header, fields, cloud_data)

    def transform_to_numpy(self, ros_transformation):
        x = ros_transformation.transform.translation.x
        y = ros_transformation.transform.translation.y
        z = ros_transformation.transform.translation.z
        
        qx = ros_transformation.transform.rotation.x
        qy = ros_transformation.transform.rotation.y
        qz = ros_transformation.transform.rotation.z
        qw = ros_transformation.transform.rotation.w

        return np.array( [x, y, z, qx, qy, qz, qw] )

    def image_process(self, bgr, depth, intrinsic_np, original_img_size, resized_intrinsic_np, resized_img_size):
        
        cx = intrinsic_np[0,2]
        cy = intrinsic_np[1,2]

        fx_factor = resized_intrinsic_np[0,0] / intrinsic_np[0,0]
        fy_factor = resized_intrinsic_np[1,1] / intrinsic_np[1,1]

        raw_fx = resized_intrinsic_np[0,0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        raw_fy = resized_intrinsic_np[1,1] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
        raw_cx = resized_intrinsic_np[0,2] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        raw_cy = resized_intrinsic_np[1,2] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]

        width = resized_img_size[0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        height = resized_img_size[0] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
        
        half_width = int( width / 2.0 )
        half_height = int( height / 2.0 )

        cropped_bgr = bgr[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width), :]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.resize(cropped_rgb, resized_img_size)

        cropped_depth = depth[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width)]
        processed_depth = cv2.resize(cropped_depth, resized_img_size, interpolation =cv2.INTER_NEAREST)

        return processed_rgb, processed_depth

    def publish_tf(self):
        left_t = TransformStamped()
        right_t = TransformStamped()
        master_cam_t = TransformStamped()
        
        # # Read message content and assign it to
        # # corresponding tf variables
        ros_time = self.get_clock().now()
        left_t.header.stamp = ros_time.to_msg()
        right_t.header.stamp = ros_time.to_msg()
        master_cam_t.header.stamp = ros_time.to_msg()

        left_t.header.frame_id = 'master_camera'
        left_t.child_frame_id = "follower_left/base_link"
        left_t.transform.translation.x = -0.288
        left_t.transform.translation.y = 0.249
        left_t.transform.translation.z = 0.373

        left_t.transform.rotation.x = -0.619
        left_t.transform.rotation.y = 0.664
        left_t.transform.rotation.z = -0.312
        left_t.transform.rotation.w = -0.282

        right_t.header.frame_id = 'master_camera'
        right_t.child_frame_id = "follower_right/base_link"
        right_t.transform.translation.x = 0.352
        right_t.transform.translation.y = 0.231
        right_t.transform.translation.z = 0.415

        right_t.transform.rotation.x = -0.619
        right_t.transform.rotation.y = 0.664
        right_t.transform.rotation.z = -0.312
        right_t.transform.rotation.w = -0.282

        master_cam_t.header.frame_id = 'world'
        master_cam_t.child_frame_id = "master_camera"
        master_cam_t.transform.translation.x = -0.1393031
        master_cam_t.transform.translation.y = 0.0539
        master_cam_t.transform.translation.z = 0.43911375

        master_cam_t.transform.rotation.x = -0.61860094
        master_cam_t.transform.rotation.y = 0.66385477
        master_cam_t.transform.rotation.z = -0.31162288
        master_cam_t.transform.rotation.w = 0.2819945

        # cam_t.header.frame_id = 'master_camera'
        # cam_t.child_frame_id = "zed_left_camera_frame"
        # cam_t.transform.translation.x = 0.0
        # cam_t.transform.translation.y = 0.0
        # cam_t.transform.translation.z = 0.0
        # cam_t.transform.rotation.x = -0.4996018
        # cam_t.transform.rotation.y =  -0.4999998
        # cam_t.transform.rotation.z = 0.4999998
        # cam_t.transform.rotation.w = 0.5003982

        # # Send the transformation
        self.tf_broadcaster.sendTransform(left_t)
        self.tf_broadcaster.sendTransform(right_t)
        self.tf_broadcaster.sendTransform(master_cam_t)
        # self.tf_broadcaster.sendTransform(cam_t)

    def SyncCallback(self, bgr, depth):
   

        current_state = {}


        current_state["bgr"] = np.array(self.br.imgmsg_to_cv2(bgr))[:,:,:3] # Todo check bgr order
        current_state["depth"] = np.array(self.br.imgmsg_to_cv2(depth, desired_encoding="mono16"))
              
        print("added a point")
        self.current_stack.append( current_state )
        if( len(self.current_stack) >= 10):
            self.episode_end(True)
 

    def save_data(self):
        now = time.time()
        print("saved ", len(self.current_stack) , "data !!!")
        print("saved ", len(self.current_stack) , "data !!!")
        print("saved ", len(self.current_stack) , "data !!!")
        np.save( str(now), self.current_stack)
    
    def clean_data(self):
        self.current_stack.clear()

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


        if( (start_recording_pressed == True) and (self.start_recording_pressed_last == False) ):
            if( self.recording == False ):
                self.recording = True
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

        self.start_recording_pressed_last = start_recording_pressed
        self.success_stop_pressed_last = success_stop_pressed           
        self.failure_stop_pressed_last = failure_stop_pressed

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
