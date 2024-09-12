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

    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    episode = np.load("11.npy", allow_pickle = True)

    print(episode[0])
    follower_bot_left.arm.set_joint_positions(episode[0]["left_pos"][0:6], blocking=False)
    follower_bot_right.arm.set_joint_positions(episode[0]["right_pos"][0:6], blocking=False)
    time.sleep(1)

    total_success = 0
    idx = 0
    # for data_point in episode[50:53]:
    # while rclpy.ok():
    for idx, data_point in enumerate( episode ):
        idx = idx + 1
        data_point = episode[idx]

        left_current_joints = follower_bot_left.core.joint_states.position[:6]
        right_current_joints = follower_bot_right.core.joint_states.position[:6]
        left_current_joints = np.array(left_current_joints)
        right_current_joints = np.array(right_current_joints)

        data_point["left_ee"][1] -= 0.315
        data_point["right_ee"][1] += 0.315

        print("\nidx: ", idx)
        # ik_result, err, success = get_ik(  model, data, data_point["right_ee"], current_joints, debug=False )
        left_ik_result, err, success = custom_ik( data_point["left_ee"], left_current_joints, debug=False )
        right_ik_result, err, success = custom_ik( data_point["right_ee"], right_current_joints, debug=False )

        if( success == False):
            print("failed !!!!!!!!!!!!")
            print("err: ", err)
            # print("gt: ", data_point['right_pos'])
            # print("trans: ", get_transform( data_point['right_ee'] ) )

        # joints = data_point["right_pos"][0:6]
        follower_bot_left.arm.set_joint_positions(left_ik_result, blocking=False)
        follower_bot_right.arm.set_joint_positions(right_ik_result, blocking=False)
        
        gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            data_point["left_pos"][6] - 0.7
        )
        follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
        
        gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            data_point["right_pos"][6] - 0.7
        )
        follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
        
        
        # sleep DT
        get_interbotix_global_node().get_clock().sleep_for(CONTROL_DT_DURATION)

    print("success: {} / {} ".format(str( total_success ), str( len(episode))) )
    print("finished !!!!!!!!!!" )

    # print("finished !!!!!!!!!!" )



if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)