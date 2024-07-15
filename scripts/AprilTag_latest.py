#3dda data:
# rgbd image 1920*1080 need post process
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
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3,TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32
import numpy as np 
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
from pupil_apriltags import Detector

aprilTag_R = np.array([
    [1.,0.,0.],
    [0.,-1.,0.],
    [0.,0.,-1.],
    ])

class DataCollector(Node):

    def __init__(self):
        super().__init__('aloha_3dda_data_collection_node')
        print("in init")
        # Declare and acquire `target_frame` parameter
        self.left_hand_frame = "follower_left/ee_gripper_link"
        self.right_hand_frame = "follower_right/ee_gripper_link"
        self.base_frame = "world"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.left_hand_transform = TransformStamped()
        self.right_hand_transform = TransformStamped()

        #axes
        self.left_joystick_x = 0
        self.left_joystick_y = 1
        self.l2 = 2
        self.right_joystick_x = 3
        self.right_joystick_y = 4
        self.right_trigger = 5
        self.leftside_left_right_arrow = 6
        self.l = leftside_up_down_arrow = 7

        self.max_idx = 7
        
        # button mapping for wireless controller
        self.x_button = 0
        self.o_button = 1
        self.triangle_button = 2
        self.square_button = 3

        self.l1 = 4
        self.r1 = 5
        self.l2 = 6
        self.r2 = 7


        self.share_button = 8
        self.opotions_button = 9

        self.max_button = 9

        # states
        self.recording = False

        # data
        self.current_stack = {}

        self.success_stop_pressed_last = False
        self.failure_stop_pressed_last = False
        
        self.br = CvBridge()
        self.detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
        )     
        
        self.odom=Odometry()
        self.odom_pub = self.create_publisher( "AprilTagOdom", Odometry, queue_size=1)
        
        self.cam_odom=Odometry()
        self.global_cam_pub = self.create_publisher("CamOdom", Odometry, queue_size=1)

        # Camera Instrinsics used for Undistorting the Fisheye Images
        self.DIM=(1080, 1920)
        self.K=np.array([[738.52671777, 0., 959.40116984], [0. ,739.11251938,  575.51338683], [0.0, 0.0, 1.0]])

        self.D=np.array([0.0, 0.0, 0.0, 0.0])

        # Call on_timer function every second
        self.timer_period = 0.01
        self.timer = self.create_timer( self.timer_period, self.on_timer )
        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joyCallback,1)
        
        self.subscription = self.create_subscription(Image, "/cam1/zed_node_A/left/image_rect_color", self.img_callback, 1)


    def img_callback(self, data):
        # Process the received image data here
        cv_image = self.br.imgmsg_to_cv2(data, desired_encoding="rgb8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # print()
        result = self.detector.detect(gray_image, True, camera_params=(738.52671777, 739.11251938, 959.40116984, 575.51338683),tag_size = 0.155) 
        # Camera Intrinsics after undistorting fisheye images, Tag Size is the length of the side of an aprilTag
        
        if result: 
            # print("*****************************************************************************************")
            # print(result)
            for tag in result: 
                if(tag.tag_id):  
                
                    original_estimated_rot = tag.pose_R 
                    original_estimated_trans = tag.pose_t
                    original_estimated_rot =   tag.pose_R @ aprilTag_R
                    # print("trans: ", original_estimated_trans)
                    # print("rot: ", original_estimated_rot)
                    # print("original_estimated_rot", type(original_estimated_rot) )

                    roll, pitch, yaw = euler_from_matrix(original_estimated_rot)
      
                    odom_quat = quaternion_from_euler(roll, pitch, yaw)  
                    # self.odom.pose.pose.position = Point(original_estimated_trans[2], -original_estimated_trans[0], -original_estimated_trans[1])
                    self.odom.pose.pose.position = Point(original_estimated_trans[0], original_estimated_trans[1], original_estimated_trans[2])

                    
                    self.odom.pose.pose.orientation.x=odom_quat[0]
                    self.odom.pose.pose.orientation.y=odom_quat[1]
                    self.odom.pose.pose.orientation.z=odom_quat[2]
                    self.odom.pose.pose.orientation.w=odom_quat[3]
                    
                    self.odom.header.stamp=rospy.Time.now()
                    self.odom.header.frame_id="map"
                    self.odom_pub.publish(self.odom)

                    
                    global_cam_rot = original_estimated_rot.transpose()
                    global_cam_trans = -1*global_cam_rot@original_estimated_trans

                    print("trans: ", global_cam_trans)
                    print("rot (x y z w): \n", self.odom.pose.pose.orientation)

                    roll, pitch, yaw = euler_from_matrix(global_cam_rot)
                    odom_quat = quaternion_from_euler(roll, pitch, yaw)
                    self.cam_odom.pose.pose.orientation.x=odom_quat[0]
                    self.cam_odom.pose.pose.orientation.y=odom_quat[1]
                    self.cam_odom.pose.pose.orientation.z=odom_quat[2]
                    self.cam_odom.pose.pose.orientation.w=odom_quat[3]
                    self.cam_odom.pose.pose.position = Point(global_cam_trans[0], global_cam_trans[1], global_cam_trans[2])
                    self.cam_odom.header.stamp=rospy.Time.now()
                    self.cam_odom.header.frame_id="map"
                    self.global_cam_pub.publish(self.cam_odom)

    def save_data(self):
        now = time.time()
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
                self.base_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.base_frame} to {self.left_hand_frame}: {ex}'
            )
            return

        try:
            self.right_hand_transform = self.tf_buffer.lookup_transform(
                self.right_hand_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.base_frame} to {self.right_hand_frame}: {ex}'
            )
            return
        # return   
        # print("updated trans:")
        # print("left hand: ", self.left_hand_transform)
        # print("right hand: ", self.right_hand_transform)

    def joyCallback(self, msg):

        start_recording_pressed = msg.buttons[self.triangle_button]
        success_stop_pressed = msg.buttons[self.o_button]
        failure_stop_pressed = msg.buttons[self.x_button]


        if( (start_recording_pressed == True) and (self.start_recording_pressed_last == False) ):
            if( self.recording == False ):
                self.recording = True
                self.get_logger().info('start recording!!!')
                self.get_logger().info('start recording!!!')
            else:
                self.recording = True
                self.episode_end(False)
                self.get_logger().info('start recording!!!')
                self.get_logger().info('start recording!!!')                

        if( (success_stop_pressed == True) and (self.success_stop_pressed_last == False) ):
            if( self.recording == True ):
                self.recording = False
                self.episode_end(True)
                self.get_logger().info('episode succeed!!!')
                self.get_logger().info('episode succeed!!!')

        if( (failure_stop_pressed == True) and (self.failure_stop_pressed_last == False) ):
            if( self.recording == True ):
                self.recording = False
                self.episode_end(False)
                self.get_logger().info('episode failed!!!')
                self.get_logger().info('episode failed!!!')

        self.start_recording_pressed_last = start_recording_pressed
           

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