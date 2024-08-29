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

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.066 #15hz
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

def roundup_pi(current_joint, next_steps):
    rounded_result = []
    for i in range( next_steps.shape[0] ):
        rouneded = next_steps[i] - (next_steps[i] // np.pi) * np.pi
        rounded_result.append(rouneded)
    return rounded_result
        
def get_ik(
    model, data, goal_ee_7D, current_joints,
    debug = False
) -> list:
    # solve ik, 6dof, gripper not included
    eps = 1e-3
    IT_MAX = 1500
    DT = 5e-2
    damp = 1e-12

    JOINT_ID = 7

    goal_transform = get_transform(goal_ee_7D)
    rot = goal_transform[0:3, 0:3]
    trans = goal_transform[0:3, 3]
    oMdes = pinocchio.SE3( rot, trans)
    print("oMdes: ", oMdes)
    start = time.time()
    q = copy.deepcopy( current_joints )
    q.append(0.0)
    q.append(0.0)
    q = np.array(q)
    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector  # in joint frame
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * DT)
        
        if not (i % 10) and debug:
            print("%d: error = %s" % (i, err.T))
        i += 1
    end = time.time()
    
    # if debug:
    print("total time: ", end - start)
    # q = q[0:-2]
    # q = roundup_pi(current_joints, q)

    joints = []
    
    pinocchio.forwardKinematics(model, data, q)
    last_joints = q[0:6].tolist()
    
    for name, oMi in zip(model.names, data.oMi):
        # joints.append( [*oMi.rotation] )
        joints.append( oMi )
    
    print("ik_result: ", q)
    print("\nfinal error: %s" % err.T)
    print("goal: ", goal_transform[0:3, 0:3])
    # print("result: ", joints[-1].rotation )

    # pin.Quaternion(M.rotation)
    print("matrix: ", joints[-1].homogeneous )
    print("quat: ", pinocchio.Quaternion(joints[-1].rotation) )

    result_quat = pinocchio.Quaternion(joints[-1].rotation).coeffs()
    result_rot = (Rotation.from_quat(result_quat) )
    
    print("result_rot: ", result_rot.as_matrix())

    rot = (Rotation.from_matrix( goal_transform[0:3, 0:3]) )
    print("original: ", rot.as_matrix())

    print("difference: ", result_rot.as_matrix() @ (rot.as_matrix()).T)
    
    are_equal = np.allclose( result_rot.as_quat(), rot.as_quat())
    print("are_equal: ", result_rot.as_quat() * rot.as_quat())
    # print("rot: ", rot.as_matrix())

    # result_rot = joints[-1].rotation
    # rot = Rotation.from_euler('xyz', result_rot[0:3, 0] , degrees = False)
    # print("rot: ", rot.as_matrix())
    return q, err, success

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
    episode = np.load("1.npy", allow_pickle = True)

    print(len(episode))


    # pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

    # You should change here to set up your own URDF file or just pass it as an argument of this example.
    urdf_filename = (
        "../urdf/vx300s.urdf"
    )
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    data = model.createData()

    for data_point in episode:
        current_joints = follower_bot_right.core.joint_states.position[:6]
        recorded_state_joints = get_ik(  model, data, data_point["right_ee"], current_joints, debug=True )

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