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
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

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
lock = threading.Lock()

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
    start_arm_qpos[4] += 0.4
    move_arms(
        [follower_bot_left, follower_bot_right],
        [start_arm_qpos] * 2,
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE, FOLLOWER_GRIPPER_JOINT_CLOSE],
        moving_time=0.5
    )

def custom_ik(goal_ee_7D, current_joints, debug=False ):

    goal_transform = get_transform(goal_ee_7D)
    K = 0.8
    result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K)
    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success



def callback(multiarray):
    with lock:
        action = np.array(multiarray.data).reshape(-1,2,8)
        global current_action
        global new_action
        # print("action: ", action)
        if(current_action is None):
            current_action = copy.deepcopy(action)
            # print()
        else:
            new_action = copy.deepcopy(action)
    
        print("action: ", current_action.shape)

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

    if(new_action is not None):
        current_action = copy.deepcopy( new_action )
        new_action = None
        current_idx = 0

    if(current_action is None):
        return

    if(current_idx >= current_action.shape[0]):
        return
    # print("current: ", current_action)
    
    follower_left_state_joints = follower_bot_left.core.joint_states.position[:6]
    follower_right_state_joints = follower_bot_right.core.joint_states.position[:6]

    left_hand_goal = current_action[current_idx,0, 0:7 ]
    left_hand_goal[1] -= 0.315
    left_openness = current_action[current_idx,0, 7]

    right_hand_goal = current_action[current_idx,1, 0:7 ]
    right_hand_goal[1] += 0.315
    right_openness = current_action[current_idx,1, 7]

    current_left_joints = np.array( follower_left_state_joints )
    current_right_joints = np.array( follower_right_state_joints )
    # left hand
    start = time.time()
    
    left_ik_result, err, success = custom_ik( left_hand_goal, current_left_joints, debug=False )
    # right hand
    right_ik_result, err, success = custom_ik( right_hand_goal, current_right_joints, debug=False )
    end = time.time()
    print("ik: ", end -start)
    print("success: ", success)
    print()
    
    current_idx += 1
    if(success == False):
        return

    follower_bot_left.arm.set_joint_positions(left_ik_result, blocking=False)
    follower_bot_right.arm.set_joint_positions(right_ik_result, blocking=False)
    
    gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        left_openness
    )
    gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        right_openness
    )
    # print("gripper: ", data_point["right_pos"][6])
    follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
    follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)

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
    timer_period = 0.1  # second
    node.timer = node.create_timer(timer_period, timer_callback)



    rclpy.spin(node)

    robot_shutdown(node)


if __name__ == '__main__':
    main()