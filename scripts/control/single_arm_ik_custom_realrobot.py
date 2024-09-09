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
import pinocchio
import time
from numpy.linalg import norm, solve
import copy
 
from BodyJacobian import *
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm
from FwdKin import *
from getXi import *

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.05 #15hz
CONTROL_DT_DURATION = Duration(seconds=0, nanoseconds= CONTROL_DT * S_TO_NS)

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


def RRcontrol(gdesired, q, K):

    dist_threshold = 0.05 # m
    angle_threshold = (5.0*np.pi)/180 # rad
    Tstep = 0.2
    maxiter = 1000
    current_q = copy.deepcopy(q)
    
    current_q = current_q.reshape(6,1)
    start = time.time()
    for i in range(maxiter):
        gst = FwdKin(current_q)
        err = inv(gdesired) @ gst
        xi = getXi(err)
        
        J = BodyJacobian(current_q)

        current_q = current_q - K * Tstep * np.linalg.pinv(J) @ xi

        J = BodyJacobian(current_q)
        finalerr = (LA.norm(xi[0:3]), LA.norm(xi[3:6]))
        # translation_err = LA.norm(xi[0:3])
        # rotation_err = LA.norm(xi[3:6])

        if abs(np.linalg.det(J))<0.001:
            # print('Singularity position')
            current_q = current_q + 0.01
            # finalerr = -1
            # break
        
        if LA.norm(xi[0:3]) < dist_threshold and LA.norm(xi[3:6]) < angle_threshold :   
            break;
    
    end = time.time()
    print('Convergence achieved. Final error: {} cm     {}  rad'.format( finalerr[0]*10, finalerr[1]) )
    print("time cost: ", end - start)
    success = False
    if(finalerr[0] < dist_threshold and finalerr[1] < angle_threshold):
        success = True
    return current_q, finalerr, success

def custom_ik( goal_ee_7D, current_joints, debug=False ):
    goal_transform = get_transform(goal_ee_7D)
    K = 0.8
    result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K)

    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success

def main() -> None:
   
    

    # last_joints = episode[0]["right_pos"][0:6].tolist()
   

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

    episode = np.load("3.npy", allow_pickle = True)


    follower_bot_right.arm.set_joint_positions(episode[0]["right_pos"][0:6], blocking=False)
    time.sleep(1)

    total_success = 0
    idx = 0
    # for data_point in episode[50:53]:
    while rclpy.ok():
    # for idx, data_point in enumerate( episode ):
        idx = idx + 1
        data_point = episode[idx]
        current_joints = follower_bot_right.core.joint_states.position[:6]
        current_joints = np.array(current_joints)
        data_point["right_ee"][1] += 0.315

        print("\nidx: ", idx)
        # ik_result, err, success = get_ik(  model, data, data_point["right_ee"], current_joints, debug=False )
        ik_result, err, success = custom_ik( data_point["right_ee"], current_joints, debug=False )
        
        if( success == False):
            print("failed !!!!!!!!!!!!")
            print("err: ", err)
            # print("gt: ", data_point['right_pos'])
            # print("trans: ", get_transform( data_point['right_ee'] ) )
        else:
            total_success += 1
            last_joints = ik_result

        # joints = data_point["right_pos"][0:6]
        follower_bot_right.arm.set_joint_positions(ik_result, blocking=False)
        
        gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            data_point["right_pos"][6] - 0.7
        )
        print("gripper: ", data_point["right_pos"][6])
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