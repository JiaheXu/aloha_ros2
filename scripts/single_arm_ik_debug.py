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
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from sys import argv
from os.path import dirname, join, abspath
# import pinocchio
import time
from numpy.linalg import norm, solve
import copy
 
np.set_printoptions(suppress=True,precision=4)
from utils import *
from math_tools import *

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.05 #15hz
CONTROL_DT_DURATION = Duration(seconds=0, nanoseconds= CONTROL_DT * S_TO_NS)

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


def custom_ik( goal_ee_7D, current_joints, debug=False ):
    goal_transform = get_transform(goal_ee_7D)
    K = 0.8
    result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K)

    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success

def main() -> None:
   
    current_left = np.array( [-0.1066, -0.7721,  1.1692, -0.6235, -0.0772,  0.5919])
    goal = np.array( [ 0.3811,  0.1786,  0.1611,  0.0268,  0.1846, -0.1467,  0.9714] )
    left_ik_result = np.array( [-0.3729, -0.1258,  0.8831, -0.1782, -0.3866,  0.1354] )
    diff = np.array( [-0.2663,  0.6463, -0.2861,  0.4453, -0.3094, -0.4565] )
    goal[1] -= 0.315
    
    current_position = get_7D_transform( FwdKin(current_left) )
    gdesired = get_transform( goal )
    print("pose difference: ", current_position[0:3] - goal[0:3])

    K = 0.4
    start = time.time()
    left_ik_result, err, success = RRcontrol(gdesired, current_left , K, debug = True)
    print("current_left: ", current_left)
    print("left_ik_result: ", left_ik_result)
    print("diff: ", left_ik_result - current_left )
    if(np.max( np.abs( left_ik_result[0:3] - current_left[0:3]) ) > 0.2 ):
        print("big jump !!!!")

if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)


# diff:  [-0.1924  0.4442 -0.1618  0.3692 -0.2513 -0.3871]