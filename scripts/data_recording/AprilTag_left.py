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
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped, Point
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32
import numpy as np 
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
from rclpy.qos import QoSProfile
from rclpy.clock import Clock
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster

from pupil_apriltags import Detector

from nav_msgs.msg import Odometry
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

aprilTag_R = np.array([
    [1.,0.,0.],
    [0.,-1.,0.],
    [0.,0.,-1.],
    ])

class DataCollector(Node):

    def __init__(self):
        super().__init__('aloha_3dda_data_collection_node')
        # print("in init")
        # Declare and acquire `target_frame` parameter
        self.left_hand_frame = "follower_left/ee_gripper_link"
        self.right_hand_frame = "follower_right/ee_gripper_link"
        self.left_base_frame = "follower_left/base_link"
        self.right_base_frame = "follower_right/base_link"

        self.base_frame = "follower_left/base_link"
        self.tag_frame = "apriltag"

        self.last_data_time = time.time()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.left_hand_transform = TransformStamped()
        self.right_hand_transform = TransformStamped()
        self.hand_transform = TransformStamped()

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
        self.X_button = 3
        self.Y_button = 2

        # states
        self.recording = False

        # data
        self.current_stack = []

        self.success_stop_pressed_last = False
        self.failure_stop_pressed_last = False
        
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        # Camera Instrinsics used for Undistorting the Fisheye Images
        self.DIM=(1080, 1920)
        self.K=np.array([[734.1779174804688, 0., 993.6226806640625], [0. ,734.1779174804688,  551.8895874023438], [0.0, 0.0, 1.0]])
        self.D=np.array([0.0, 0.0, 0.0, 0.0])


        self.odom=Odometry()
        self.tag_odom_pub = self.create_publisher(Odometry, "AprilTagOdom", 1)
        
        self.cam_odom=Odometry()
        self.global_cam_pub = self.create_publisher(Odometry, "CamOdom", 1)
        # Call on_timer function every second
        self.timer_period = 0.01
        # self.timer = self.create_timer( self.timer_period, self.on_timer )
        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joyCallback,1)
        self.br = CvBridge()
        # self.subscription = self.create_subscription(Image, "/camera_1/left_image", self.img_callback, 1)
        

        queue_size = 10
        max_delay = 0.01 #10ms

        self.tf_broadcaster = TransformBroadcaster(self)

        self.rgb_sub = self.create_subscription(Image, "/camera_1/left_image", self.RgbCallback,1)
        # self.rgb_sub = Subscriber(self, Image, "/camera_1/left_image")
        # self.depth_sub = Subscriber(self, Image, "/camera_1/depth")
        
        # self.time_sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                                    #  queue_size, max_delay)
        # self.time_sync.registerCallback(self.SyncCallback)

        timer_period = 0.01 #100hz
        self.timer = self.create_timer(timer_period, self.publish_tf)
    
    def publish_tf(self):
        t = TransformStamped()

        # # Read message content and assign it to
        # # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'follower_left/gripper_link'
        t.child_frame_id = "apriltag"

        t.transform.translation.x = 0.0171
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.03935

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # # Send the transformation
        self.tf_broadcaster.sendTransform(t)
    
    def RgbCallback(self, rgb):
        try:
            self.hand_transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.tag_frame,
                    rgb.header.stamp
            )
        
        except TransformException as ex:
            # self.get_logger().info(
            #     f'Could not transform {self.base_frame} to {self.tag_frame}: {ex}'
            # )
            return

        data_time = time.time()
        if(data_time - self.last_data_time < 0.5):
            return

        print("in call backs")
        # print("base frame: ", self.hand_transform)
        x = self.hand_transform.transform.translation.x
        y = self.hand_transform.transform.translation.y
        z = self.hand_transform.transform.translation.z

        qx = self.hand_transform.transform.rotation.x
        qy = self.hand_transform.transform.rotation.y
        qz = self.hand_transform.transform.rotation.z
        qw = self.hand_transform.transform.rotation.w

        cv_image = self.br.imgmsg_to_cv2(rgb, desired_encoding="rgb8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        #print("image size: ", gray_image.shape)
        result = self.detector.detect(gray_image, True, camera_params=(self.K[0][0], self.K[1][1], self.K[0][2], self.K[1][2]),tag_size = 0.06) 
        if result: 
            # print("*****************************************************************************************")
            # print(result)
            for tag in result: 
                if(tag.tag_id==0):  
                    timestamp = self.get_clock().now().to_msg()
                    
                    

                    original_estimated_rot = tag.pose_R 
                    original_estimated_trans = tag.pose_t
                    original_estimated_rot =   tag.pose_R @ aprilTag_R
                    rot = Rotation.from_matrix(original_estimated_rot)
                    tag_odom_quat = rot.as_quat()

                    point = Point()
                    point.x = float(original_estimated_trans[0])
                    point.y = float(original_estimated_trans[1])
                    point.z = float(original_estimated_trans[2])
                    self.odom.pose.pose.position = point
                    
                    self.odom.pose.pose.orientation.x=tag_odom_quat[0]
                    self.odom.pose.pose.orientation.y=tag_odom_quat[1]
                    self.odom.pose.pose.orientation.z=tag_odom_quat[2]
                    self.odom.pose.pose.orientation.w=tag_odom_quat[3]
                    
                    self.odom.header.stamp=timestamp
                    self.odom.header.frame_id="map"
                    self.tag_odom_pub.publish(self.odom)

                    
                    global_cam_rot = original_estimated_rot.transpose()
                    global_cam_trans = -1.0*global_cam_rot@original_estimated_trans

                    # print("trans: ", point.x, point.y, point.z)
                    # print("rot (x y z w): \n", self.odom.pose.pose.orientation)

                    rot = Rotation.from_matrix(global_cam_rot)
                    odom_quat = rot.as_quat()
                    self.cam_odom.pose.pose.orientation.x=odom_quat[0]
                    self.cam_odom.pose.pose.orientation.y=odom_quat[1]
                    self.cam_odom.pose.pose.orientation.z=odom_quat[2]
                    self.cam_odom.pose.pose.orientation.w=odom_quat[3]
                    
                    point2 = Point()
                    point2.x = float(global_cam_trans[0])
                    point2.y = float(global_cam_trans[1])
                    point2.z = float(global_cam_trans[2])
                    self.cam_odom.pose.pose.position = point2
                    self.cam_odom.header.stamp=timestamp
                    self.cam_odom.header.frame_id="map"
                    self.global_cam_pub.publish(self.cam_odom)

                    current_state = {}
                    current_state["base_tag"] = np.array([x, y, z, qx, qy, qz, qw])
                    current_state["camera_tag"] = np.array([
                        point.x, point.y, point.z,
                        tag_odom_quat[0], tag_odom_quat[1], tag_odom_quat[2], tag_odom_quat[3]
                    ])

                    if( self.recording == True ):
                        self.current_stack.append(current_state)
                        self.last_data_time = data_time
                        self.recording = False
                        print("added data!!!!!!!!")
                        print("added data!!!!!!!!")
                        print("added data!!!!!!!!")   

    def save_data(self):
        now = time.time()
        print("collected ", len(self.current_stack), " pairs of data")
        print("collected ", len(self.current_stack), " pairs of data")
        print("collected ", len(self.current_stack), " pairs of data")
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
            # if( self.recording == False ):
            self.recording = True
            self.get_logger().info('adding data!!!')
            #     # self.get_logger().info('start recording!!!')
            # else:
            #     self.recording = True
            #     self.episode_end(False)
            #     self.get_logger().info('start recording!!!')
            #     # self.get_logger().info('start recording!!!')                

        if( (success_stop_pressed == True) and (self.success_stop_pressed_last == False) ):
            # if( self.recording == True ):
            self.recording = False
            self.episode_end(True)
            self.get_logger().info('episode succeed!!!')
            # self.get_logger().info('episode succeed!!!')

        if( (failure_stop_pressed == True) and (self.failure_stop_pressed_last == False) ):
            # if( self.recording == True ):
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
