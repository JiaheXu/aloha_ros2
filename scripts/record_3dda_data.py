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
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32
import numpy as np 
import time

class DataCollector(Node):

    def __init__(self):
        super().__init__('3dda_data_collection_node')

        # Declare and acquire `target_frame` parameter
        self.left_hand_frame = "follower_left/ee_gripper_link"
        self.left_hand
        self.right_hand_frame = "follower_right/ee_gripper_link"
        self.base_frame = "world"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.left_hand_transform = TransformStamped()
        self.right_hand_transform = TransformStamped()

        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self.on_timer)

        self.joystick_sub = self.create_subscription(Joy, "joy", self.joyCallback)


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

    def joyCallback(self, msg):
            start_recording_pressed = msg.buttons[self.triangle_button]
            success_stop_pressed = msg.buttons[self.o_button]
            failure_stop_pressed = msg.buttons[self.x_button]


            if( (start_recording_pressed == True) and (self.start_recording_pressed_last == False) ):
                if( self.recording == False ):
                    self.recording = True
                    self.get_logger().info('start')

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