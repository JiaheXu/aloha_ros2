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
def main() -> None:
    node = create_interbotix_global_node('aloha')
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

    # Teleoperation loop
    # gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    # todo, change this to ROS
    action = np.load("ep2_action.npy", allow_pickle = True)
    action = action.reshape(-1,8)
    length = action.shape[0]
    # while rclpy.ok():
    idx = 0
    while( idx < length):
        T_sd = get_transform( action[idx][0:7])
        right_gripper = un_normalize_gripper( action[idx][7] )
        gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            right_gripper
        )
        
        # follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)


        follower_bot_right.arm.set_ee_pose_matrix(T_sd)
        follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
        # sleep DT
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    robot_shutdown(node)


if __name__ == '__main__':
    main()
