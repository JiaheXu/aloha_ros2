#!/usr/bin/env python3

from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
    InterbotixRobotNode
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

from threading import Thread
from typing import Optional

from interbotix_common_modules.common_robot.exceptions import InterbotixException
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

import numpy as np
np.set_printoptions(suppress=True,precision=4)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout, Bool

from utils import *
from math_tools import *
import threading
import time

new_action = None
current_action = None
follower_bot_left = None
follower_bot_right = None
gripper_left_command = None
gripper_right_command = None
node = None
current_idx = 0
save_data_idx = 0
current_traj = []

lock = threading.Lock()

left_bias = get_transform(   [ -0.01, 0.365, 0.00 ,0., 0.,0.02617695, 0.99965732] )
left_tip_bias = get_transform( [-0.028, 0.01, 0.01,      0., 0., 0., 1.] ) @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
left_goal_bias = get_transform(   [ 0.0, 0., 0.00, 0., 0., 0., 1.0] )

right_bias = get_transform(   [ 0.01, -0.315, 0.0, 0., 0., 0., 1.0] )
right_tip_bias = get_transform( [-0.035, 0.01, -0.008,      0., 0., 0., 1.] ) @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
right_goal_bias = get_transform(   [ 0.0, 0.0, 0.0, 0., 0., 0., 1.0] )

def opening_ceremony(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(follower_bot_left)
    torque_on(follower_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    # print("start_arm_qpos: ", [start_arm_qpos] * 2)
    # start_arm_qpos[4] += 0.4
    start_poses = [ 
        # close marker
        [-0.39269909, -0.83908749,  0.43258259, -0.23930101,  0.81147587, 0.23469907],
        [0.23469907, -0.69949526,  0.32213598,  0.42644668,  0.85749531, -0.41877678]
        # stack bowl
        # [-0.27304858, -0.65194184,  0.82067972, -0.36968938,  0.5276894 , 0.39576706],
        # [ 0.39423308, -0.52001953,  0.77159238,  0.08130098,  0.57370883, 0.00613592]
        # start_arm_qpos
    ]
    # start_poses = [start_arm_qpos] * 2
    print("start_poses: ", start_poses)
    move_arms(
        [follower_bot_left, follower_bot_right],
        # [start_arm_qpos] * 2,
        start_poses,
        moving_time=4.0,
    )


    # move grippers to starting position
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [1.62, 1.62],
        moving_time=0.5
    )

def custom_ik(goal_transform, current_joints, debug=False ):

    K = 0.4
    result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K, debug = debug)
    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success



def callback(multiarray):
    with lock:
        global current_action
        global new_action
        global follower_bot_left
        global follower_bot_right

        action = np.array(multiarray.data).reshape(-1,2,8)

        trajectory = copy.deepcopy(action)
        print("right goal: ", trajectory[0,1,:])

        if(trajectory[0,1,2] < -0.03):
            trajectory[0,1,7] = 0

        dist_threshold = 0.01 # 1cm

        follower_left_state_joints = follower_bot_left.core.joint_states.position[:7]
        follower_right_state_joints = follower_bot_right.core.joint_states.position[:7]
        current_left_joints = np.array( follower_left_state_joints )
        current_left_joints[6] -= 0.62
        current_right_joints = np.array( follower_right_state_joints )
        current_right_joints[6] -= 0.62   
        
        # all convert to robot frame
        left_transform = FwdKin(current_left_joints)
        left_tranform_7D = get_7D_transform( left_transform )

        right_transform = FwdKin(current_right_joints)
        right_tranform_7D = get_7D_transform( right_transform )


        left_gripper = np.zeros( (8,) )
        left_gripper[0:7] = left_tranform_7D[0:7]
        left_gripper[7] = follower_bot_left.core.joint_states.position[6] - 0.62

        right_gripper = np.zeros( (8,) )
        right_gripper[0:7] = right_tranform_7D[0:7]
        right_gripper[7] = follower_bot_right.core.joint_states.position[6] - 0.62

        # convert goal to robot arm frame
        left_goal = np.zeros( (8,) )
        left_goal[0:7] = get_7D_transform( inv(left_bias) @ get_transform(trajectory[0, 0, 0:7]) @ inv(left_tip_bias) @ left_goal_bias)
        left_goal[7] = trajectory[0,0,7]

        right_goal = np.zeros( (8,) )
        right_goal[0:7] = get_7D_transform( inv(right_bias) @ get_transform(trajectory[0, 1, 0:7]) @ inv(right_tip_bias) @ right_goal_bias)
        right_goal[7] = trajectory[0,1,7]

        left_length = max(1, int(LA.norm(left_goal[0:3] - left_gripper[0:3])/0.01)*3 )
        right_length = max(1, int(LA.norm(right_goal[0:3] - right_gripper[0:3])/0.01)*3 )
        interpolation_length = max( left_length, right_length)

        current_joints = [current_left_joints, current_right_joints]
        goals = [left_goal, right_goal]

        left_stack, right_stack = get_two_points_trajectory( current_joints, goals, interpolation_length)

        left_traj = np.expand_dims(left_stack, axis=1)
        right_traj = np.expand_dims(right_stack, axis=1)
        # print("left_z: ", trajectory[0,0,3])
        if(trajectory[0,0,2] < -0.03):
            left_traj[-1,-1,-1] = 0


        # print("left_traj: ", left_traj.shape)
        # print("left_traj: ", left_traj)
        # with lock:
        new_action = np.concatenate( [left_traj, right_traj], axis = 1)
        print("new_action: ", new_action.shape)

def save_data():
    global save_data_idx
    global current__traj
    np.save("traj_track_{}".format(save_data_idx), current_traj, allow_pickle = True)
    save_data_idx += 1

def timer_callback():
    print("in timer call back")
    global current_action
    global new_action
    global current_idx
    global follower_bot_left
    global follower_bot_right
    global gripper_left_command
    global gripper_right_command
    global node
    global current_traj

    if(new_action is not None):
        current_action = copy.deepcopy( new_action )
        new_action = None
        current_idx = 0

    if(current_action is None):
        return

    if(current_idx >= current_action.shape[0]):
        # save_data()
        current_action = None
        msg = Bool()
        msg.data = True
        node.state_publisher.publish(msg)
        print("finished !!!")
        print("finished !!!")
        print("finished !!!")
        return
    # print("current: ", current_action[0:3,:,:])
    print("current_idx: ", current_idx)


    left_ik_result = current_action[current_idx,0,0:6]
    left_openness = current_action[current_idx,0,6]

    right_ik_result = current_action[current_idx,1,0:6]
    right_openness = current_action[current_idx,1,6]

    follower_left_state_joints = follower_bot_left.core.joint_states.position[:7]
    follower_right_state_joints = follower_bot_right.core.joint_states.position[:7]
    current_left_joints = np.array( follower_left_state_joints )
    current_right_joints = np.array( follower_right_state_joints )
    
    # all convert to robot frame
    left_transform = FwdKin(current_left_joints)
    left_tranform_7D = get_7D_transform( left_transform )

    # print("current left: ", left_tranform_7D)
    # print("goal: ", right_openness)
    # print("current: ", current_left_joints[6] - 0.6, " ", current_right_joints[6] - 0.6)
    # print("goal: ", left_openness, " ", left_openness)

    follower_bot_left.arm.set_joint_positions(left_ik_result, blocking=False)
    follower_bot_right.arm.set_joint_positions(right_ik_result, blocking=False)

    if(left_openness < 0.35 ):
        left_openness = 0
    else:
        left_openness = 1

    if(right_openness < 0.35):
        right_openness = 0
    else:
        right_openness = 1

    gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        left_openness 
    )
    gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        right_openness
    )
    # print("gripper: ", data_point["right_pos"][6])
    follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
    follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)

    current_idx += 1

def main() -> None:
    
    global current_action
    global new_action
    global follower_bot_left
    global follower_bot_right
    global gripper_left_command
    global gripper_right_command
    global node

    node = create_interbotix_global_node('aloha')
    # node = create_bimanual_global_node('bimanual')

    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    robot_startup(node)

    opening_ceremony(
        follower_bot_left,
        follower_bot_right,
    )

    # press_to_start(leader_bot_left, leader_bot_right)

    # Teleoperation loop
    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    node.bimanual_ee_cmd_sub = node.create_subscription( Float32MultiArray, "bimanual_ee_cmd", callback, 1)
    idx = 0
    timer_period = 0.05  # second
    node.timer = node.create_timer(timer_period, timer_callback)

    node.state_publisher = node.create_publisher(Bool, 'controller_finished', 1)

    rclpy.spin(node)

    robot_shutdown(node)


if __name__ == '__main__':
    main()
