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


from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.066 #15hz
CONTROL_DT_DURATION = Duration(seconds=0, nanoseconds= CONTROL_DT * S_TO_NS)

import time

def get_transform( t_7d ):
    t = np.eye(4)
    trans = t_7d[0:3]
    quat = t_7d[3:7]
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t
def un_normalize_gripper( position , min_joint = 0.6, max_joint = 1.6): 
    '''
    position should be a number between 0 and 1 
    '''
    if(position < 0.0):
        position = 0.0
    if(position > 1.0):
        position = 1.0

    joint_pos = position * (max_joint - min_joint) + min_joint

    return joint_pos

def opening_ceremony(

    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors


    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes(
        'single', 'gripper', 'current_based_position'
    )

    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    # torque_on(follower_bot_left)
    # torque_on(leader_bot_left)
    torque_on(follower_bot_right)
    # torque_on(leader_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [follower_bot_right],
        [start_arm_qpos],
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [ follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE],
        moving_time=0.5
    )

def press_to_start(
    leader_bot_left: InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
) -> None:
    # press gripper to start teleop
    # disable torque for only gripper joint of leader robot to allow user movement
    leader_bot_left.core.robot_torque_enable('single', 'gripper', False)
    leader_bot_right.core.robot_torque_enable('single', 'gripper', False)
    print('Close the grippers to start')
    pressed = False
    while rclpy.ok() and not pressed:
        pressed = (
            (get_arm_gripper_positions(leader_bot_left) < LEADER_GRIPPER_CLOSE_THRESH) and
            (get_arm_gripper_positions(leader_bot_right) < LEADER_GRIPPER_CLOSE_THRESH)
        )
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)
    torque_off(leader_bot_left)
    torque_off(leader_bot_right)
    print('Started!')


def main() -> None:
    node = create_interbotix_global_node('aloha')

    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )

    robot_startup(node)

    opening_ceremony
    (
        follower_bot_right,
    )
    gripper_right_command = JointSingleCommand(name='gripper')

    # todo, change this to ROS
    # episode = np.load("1.npy", allow_pickle = True)

    print(len(episode))
    for data_point in episode:

        # recorded_state_joints = data_point["right_pos"][0:6]

        follower_bot_right.arm.set_joint_positions(recorded_state_joints, blocking=False)
        gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            data_point["right_pos"][6] - 0.7
        )
        follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
        # sleep DT
        get_interbotix_global_node().get_clock().sleep_for(CONTROL_DT_DURATION)


    print("finished !!!!!!!!!!" )
    robot_shutdown(node)


if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)