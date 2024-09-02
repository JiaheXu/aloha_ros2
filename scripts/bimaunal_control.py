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
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from utils import *
from math_tools import *


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

class BimanualNode( InterbotixRobotNode ):
    def __init__(
        self,
        node_name: str = 'bimanual_node',
        namespace: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(node_name=self.node_name, namespace=self.namespace, *args, **kwargs)
        self.get_logger().info("Initialized InterbotixRobotNode!")

        self.cmd_sub = self.create_subscription(Float32MultiArray, "bi_ee_traj_cmd", self.ee_traj_cmd_callback, 1)

    def custom_ik(self, goal_ee_7D, current_joints, debug=False ):

        goal_transform = get_transform(goal_ee_7D)
        K = 0.8
        result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K)
        # print("FwdKin: ", FwdKin(result_q))
        # print("Goal: ",goal_transform)
        return result_q, finalerr, success

    def ee_traj_cmd_callback(self, msg):
        ee_np = to_numpy_f32(msg.data) # shoould be n*2*8

        for idx in range( ee_np.shape[0] ):
            # follower_left_state_joints = leader_bot_left.core.joint_states.position[:6]
            # follower_right_state_joints = leader_bot_right.core.joint_states.position[:6]
            left_hand_goal = ee_np[idx,0, : ]
            left_hand_goal[1] -= 0.315

            right_hand_goal = ee_np[idx,1, : ]
            right_hand_goal[1] += 0.315

            current_left_joints = np.zeros( (6,1) )
            current_right_joints = np.zeros( (6,1) )
            # left hand
            ik_result, err, success = self.custom_ik( left_hand_goal, current_left_joints, debug=False )
            # right hand
            ik_result, err, success = self.custom_ik( right_hand_goal, current_right_joints, debug=False )
        return

def create_bimanual_global_node(
    node_name: str = 'interbotix_robot_manipulation',
    namespace: str = None,
    *args,
    **kwargs
) -> InterbotixRobotNode:
    """
    Initialize the ROS context (if not already) and create a global generic node.

    :param node_name: The name of the node to create
    :param namespace: The namespace the node should be created under
    :return: A configured InterbotixRobotNode
    """
    if '__interbotix_global_node' in globals():
        raise InterbotixException(
            'Tried to create an Interbotix global node but one already exists.'
        )

    # Initialize the ROS context if not already
    if not rclpy.ok():
        rclpy.init(*args, **kwargs)

    # Instantiate a global InterbotixRobotNode
    global __interbotix_global_node
    __interbotix_global_node = BimanualNode(
        node_name=node_name,
        namespace=namespace,
        *args,
        **kwargs,
    )
    return __interbotix_global_node


def main() -> None:
    # node = create_interbotix_global_node('aloha')
    node = create_bimanual_global_node('bimanual')

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


    # while rclpy.ok():
    #     # sync joint positions
    #     leader_left_state_joints = leader_bot_left.core.joint_states.position[:6]
    #     leader_right_state_joints = leader_bot_right.core.joint_states.position[:6]
    #     follower_bot_left.arm.set_joint_positions(leader_left_state_joints, blocking=False)
    #     follower_bot_right.arm.set_joint_positions(leader_right_state_joints, blocking=False)
    #     # sync gripper positions
    #     gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
    #         leader_bot_left.core.joint_states.position[6]
    #     )
    #     gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
    #         leader_bot_right.core.joint_states.position[6]
    #     )
    #     follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
    #     follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
    #     # sleep DT
    #     get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    # robot_shutdown(node)


if __name__ == '__main__':
    main()
