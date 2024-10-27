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

from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from sys import argv
from os.path import dirname, join, abspath
# import pinocchio
import time
from numpy.linalg import norm, solve
import copy
from utils import *
from math_tools import *
import threading


import numpy as np
np.set_printoptions(suppress=True,precision=4)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout, Bool

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.066 #15hz
CONTROL_DT_DURATION = Duration(seconds=0, nanoseconds= CONTROL_DT * S_TO_NS)
SLEEP_DT_DURATION = Duration(seconds=0, nanoseconds= 2 * S_TO_NS)

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
        # stack bowl 39
        [-0.42644668, -0.10124274,  0.58444667, -0.59518456,  0.641204  , 0.3666214],
        [ 0.40190297, -0.61512631,  0.50467968,  0.11965051,  0.78539819, -0.03067962]
        # 31
        # [-0.65194184, -0.46633017,  0.56297094, -0.02761165,  0.87897104, 0.38963112],
        # [ 0.25157285, -0.53996128,  0.63506806,  0.11658254,  0.6657477 ,-0.06135923]
        # 1
        #[-0.27304858, -0.65194184,  0.82067972, -0.36968938,  0.5276894 , 0.39576706],
        #[ 0.39423308, -0.52001953,  0.77159238,  0.08130098,  0.57370883, 0.00613592]
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
        [0.62, 1.62],
        moving_time=0.5
    )

def custom_ik(goal_transform, current_joints, debug=False ):

    K = 0.4
    result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K, debug = debug)
    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success

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

    node.state_publisher = node.create_publisher(Bool, 'controller_finished', 1)

    opening_ceremony(
        follower_bot_left,
        follower_bot_right,
    )

    # press_to_start(leader_bot_left, leader_bot_right)

    # Teleoperation loop
    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    episode = np.load("ep39.npy", allow_pickle = True)


    left_bias = get_transform(   [ -0.01, 0.365, -0.0 ,0., 0.,0.02617695, 0.99965732] )
    left_tip_bias = get_transform( [-0.028, 0.01, 0.01,      0., 0., 0., 1.] ) @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )

    right_bias = get_transform(   [ 0.01, -0.315, 0.00, 0., 0., 0., 1.0] )
    right_tip_bias = get_transform( [-0.035, 0.01, -0.008,      0., 0., 0., 1.] ) @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )

    for episode_idx, data_point in enumerate( episode[2] ):
        print("episode_idx: ", episode_idx)
        data_point = data_point.numpy()
        trajectory = data_point
        # trajectory[0,1] -= 0.03
        # trajectory[1,1] += 0.03


        print("trajectory: ", trajectory.shape)
        follower_left_state_joints = follower_bot_left.core.joint_states.position[:7]
        follower_right_state_joints = follower_bot_right.core.joint_states.position[:7]
        current_left_joints = np.array( follower_left_state_joints )
        current_left_joints[6] -= 0.62
        current_right_joints = np.array( follower_right_state_joints )
        current_right_joints[6] -= 0.62        

        # all convert to robot frame
        left_transform = FwdKin(current_left_joints)
        # left_transform = left_bias @ left_transform @ left_tip_bias @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
        left_tranform_7D = get_7D_transform( left_transform )
        # left_tranform_7D[1] += 0.315

        right_transform = FwdKin(current_right_joints)
        # right_transform = right_bias @ right_transform @ right_tip_bias @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
        right_tranform_7D = get_7D_transform( right_transform )
        # right_tranform_7D[1] -= 0.315

        left_gripper = np.zeros( (8,) )
        left_gripper[0:7] = left_tranform_7D[0:7]
        left_gripper[7] = follower_bot_left.core.joint_states.position[6] - 0.62

        right_gripper = np.zeros( (8,) )
        right_gripper[0:7] = right_tranform_7D[0:7]
        right_gripper[7] = follower_bot_right.core.joint_states.position[6] - 0.62

        # left_stack = [left_gripper, trajectory[0,0:8]]
        # right_stack = [right_gripper, trajectory[1,0:8]]

        # convert goal to robot arm frame
        left_goal = np.zeros( (8,) )
        left_goal[0:7] = get_7D_transform( inv(left_bias) @ get_transform(trajectory[0,0:7]) @ inv(left_tip_bias) )
        left_goal[7] = trajectory[0,7]

        right_goal = np.zeros( (8,) )
        right_goal[0:7] = get_7D_transform( inv(right_bias) @ get_transform(trajectory[1,0:7]) @ inv(right_tip_bias) )
        right_goal[7] = trajectory[1,7]

        left_length = max(1, int(LA.norm(left_goal[0:3] - left_gripper[0:3])/0.01)*3 )
        right_length = max(1, int(LA.norm(right_goal[0:3] - right_gripper[0:3])/0.01)*3 )
        interpolation_length = max( left_length, right_length)

        current_joints = [current_left_joints, current_right_joints]
        goals = [left_goal, right_goal]

        left_stack, right_stack = get_two_points_trajectory( current_joints, goals, interpolation_length)

        left_traj = np.expand_dims(left_stack, axis=1)
        right_traj = np.expand_dims(right_stack, axis=1)
        # with lock:
        new_action = np.concatenate( [left_traj, right_traj], axis = 1)
        
        # print("new_action: ", new_action.shape)
        # print("openness: ", goals[0][7], " ", goals[1][7])

        for step_idx in range(new_action.shape[0]):

            left_ik_result = new_action[step_idx,0,0:6]
            left_openness = new_action[step_idx,0,6]

            right_ik_result = new_action[step_idx,1,0:6]
            right_openness = new_action[step_idx,1,6]

            follower_left_state_joints = follower_bot_left.core.joint_states.position[:7]
            follower_right_state_joints = follower_bot_right.core.joint_states.position[:7]
            current_left_joints = np.array( follower_left_state_joints )
            current_right_joints = np.array( follower_right_state_joints )
            print("current: ", current_right_joints[6] - 0.62)
            print("goal: ", right_openness)
            print("current: ", current_left_joints[6] - 0.6, " ", current_right_joints[6] - 0.6)
            print("goal: ", left_openness, " ", right_openness)

            follower_bot_left.arm.set_joint_positions(left_ik_result, blocking=False)
            follower_bot_right.arm.set_joint_positions(right_ik_result, blocking=False)

            gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
                left_openness
            )
            gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
                right_openness
            )

            follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
            follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
            get_interbotix_global_node().get_clock().sleep_for(CONTROL_DT_DURATION)

        msg = Bool()
        msg.data = True
        node.state_publisher.publish(msg)
        get_interbotix_global_node().get_clock().sleep_for(SLEEP_DT_DURATION)
        # print("data_point: ", data_point)

    print("finished !!!!!!!!!!" )
    robot_shutdown(node)


if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)

    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (2, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (2, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 2, 8)
    # List of tensors
