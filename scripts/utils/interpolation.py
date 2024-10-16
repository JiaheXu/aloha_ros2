import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from numpy import linalg as LA
from utils import *
from math_tools import *

def traj_interpolation( trajectory, interpolation_length = 50):
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)
    # Calculate the current number of steps
    old_num_steps = trajectory.shape[0]

    # Create a 1D array for the old and new steps
    old_steps = np.linspace(0, 1, old_num_steps)
    new_steps = np.linspace(0, 1, interpolation_length)

    resampled = np.empty((interpolation_length, trajectory.shape[1]))
    interpolator = CubicSpline(old_steps, trajectory[:, :-1])

    resampled[:, :-1] = interpolator(new_steps)
    last_interpolator = interp1d(old_steps, trajectory[:, -1])
    resampled[:, -1] = last_interpolator(new_steps)

    if trajectory.shape[1] == 8:
        resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
    elif trajectory.shape[1] == 16:
        resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        resampled[:, 11:15] = normalise_quat(resampled[:, 11:15])

    return resampled
        
def normalise_quat(x):
    length = np.sqrt( np.square(x).sum(axis=-1) )
    length = np.expand_dims(length, axis=1)
    #print("debug: ", debug.shape)
    result = x / np.clip(length, a_min=1e-10, a_max = 1.0)
    #norm = LA.norm(result, axis = 1)
    #print("norm: ", norm)
    return result

def get_mid_point(trajectory):
    diff1 = LA.norm(trajectory - trajectory[0], axis = 1)
    diff2 = LA.norm(trajectory - trajectory[-1], axis = 1)
    
    diff3 = np.abs(diff1 -diff2)
    idx = np.argmin( diff3 )
    #print("diff1: ", diff1)
    #print("diff2: ", diff2)
    #print("diff3: ", diff3)
    #print("idx: ", idx)
    return idx


# def custom_ik(goal_ee_7D, current_joints, debug=False ):

#     goal_transform = get_transform(goal_ee_7D)
#     K = 0.4
#     result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K, debug = debug)
#     # print("FwdKin: ", FwdKin(result_q))
#     # print("Goal: ",goal_transform)
#     return result_q, finalerr, success
def bound_joints( joints ):
    joints = joints.reshape(-1)
    for idx in range( joints.shape[0] ):
        if np.abs( joints[idx] ) > np.pi:
            while( joints[idx] > np.pi ):
                joints[idx] -= 2*np.pi
            while( joints[idx] < -np.pi ):
                joints[idx] += 2*np.pi
            # print("idx: ", idx)
    return joints            

def get_three_points_trajectory(current_joints, goals, mid_goals = None, half_traj_length = 10):

    current_left_joints = current_joints[0][0:6]
    current_right_joints = current_joints[1][0:6]

    current_left_gripper = current_joints[0][6]
    current_right_gripper = current_joints[1][6]

    left_hand_mid_goal = mid_goals[0][0:7]
    left_hand_mid_goal[1] -= 0.315
    right_hand_mid_goal = mid_goals[1][0:7]
    right_hand_mid_goal[1] += 0.315

    left_hand_mid_goal_transform = get_transform(left_hand_mid_goal)
    right_hand_mid_goal_transform = get_transform(right_hand_mid_goal)

    left_ik_result1, err, success_left = RRcontrol( left_hand_mid_goal_transform, current_left_joints, debug=False )
    left_ik_result1 = bound_joints(left_ik_result1)

    right_ik_result1, err, success_right = RRcontrol( right_hand_mid_goal_transform, current_right_joints, debug=False )
    right_ik_result1 = bound_joints(right_ik_result1)

    success = success_left and success_left
    if(success == False ):
        print("first part failed")
        print("left: ", current_left_joints)
        print("right: ", current_right_joints)
        print("left goal: ", left_hand_mid_goal)
        print("right goal: ", right_hand_mid_goal)
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        return None, None

    left_hand_goal = goals[0][0:7]
    left_hand_goal[1] -= 0.315
    right_hand_goal = goals[1][0:7]
    right_hand_goal[1] += 0.315

    left_hand_goal_transform = get_transform(left_hand_goal)
    right_hand_goal_transform = get_transform(right_hand_goal)   

    left_ik_result2, err, success_left = RRcontrol( left_hand_goal_transform, left_ik_result1, debug=False )
    left_ik_result2 = bound_joints(left_ik_result2)

    right_ik_result2, err, success_right = RRcontrol( right_hand_goal_transform, right_ik_result1, debug=False )
    right_ik_result2 = bound_joints(right_ik_result2)

    if(success == False ):
        print("2nd part failed")
        print("left: ", left_ik_result1)
        print("right: ", right_ik_result1)
        print("left goal: ", left_hand_goal)
        print("right goal: ", right_hand_goal)
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        return None, None
    print("current_joints: ", current_joints)
    print("left_ik_result1: ", left_ik_result1)
    print("right_ik_result1: ", right_ik_result1)


    #  left_ik_result1.append(mid_goals[0][7])
    print("current_left_joints: ", current_left_joints.shape)
    print("left_ik_result1: ", left_ik_result1.shape)

    left_joints1 =  np.linspace(current_left_joints, left_ik_result1, half_traj_length, endpoint = False)
    left_gripper1 = np.linspace(current_joints[0][6], mid_goals[0][7], half_traj_length, endpoint = False)
    left_gripper1 = np.expand_dims(left_gripper1, axis=1)
    left_traj1 = np.concatenate([left_joints1, left_gripper1], axis = 1)
    # print("left_traj1: ", left_traj1.shape)

    right_joints1 =  np.linspace(current_right_joints, right_ik_result1, half_traj_length, endpoint = False)
    right_gripper1 = np.linspace(current_joints[1][6], mid_goals[1][7], half_traj_length, endpoint = False)
    right_gripper1 = np.expand_dims(right_gripper1, axis=1)
    right_traj1 = np.concatenate([right_joints1, right_gripper1], axis = 1)
    # print("left_traj1: ", left_traj1)
    # return left_traj1, right_traj1

    left_joints2 =  np.linspace(left_ik_result1, left_ik_result2, half_traj_length)
    left_gripper2 =  np.linspace(mid_goals[0][7], goals[0][7], half_traj_length)
    left_gripper2 = np.expand_dims(left_gripper2, axis=1)
    left_traj2 = np.concatenate([left_joints2, left_gripper2], axis = 1)

    right_joints2 =  np.linspace(right_ik_result1, right_ik_result2, half_traj_length)
    right_gripper2 =  np.linspace(mid_goals[1][7], goals[1][7], half_traj_length)
    right_gripper2 = np.expand_dims(right_gripper2, axis=1)
    right_traj2 = np.concatenate([right_joints2, right_gripper2], axis = 1)
    
    left_traj = np.concatenate([left_traj1, left_traj2], axis = 0)
    right_traj = np.concatenate([right_traj1, right_traj2], axis = 0)

    return left_traj, right_traj

def get_two_points_trajectory(current_joints, goals , traj_length = 10):

    current_left_joints = current_joints[0][0:6]
    current_right_joints = current_joints[1][0:6]

    current_left_gripper = current_joints[0][6]
    current_right_gripper = current_joints[1][6]

    left_hand_goal = goals[0][0:7]
    right_hand_goal = goals[1][0:7]

    left_hand_goal_transform = get_transform(left_hand_goal)
    right_hand_goal_transform = get_transform(right_hand_goal)

    left_ik_result, err, success_left = RRcontrol( left_hand_goal_transform, current_left_joints, debug=False )
    left_ik_result = bound_joints(left_ik_result)

    right_ik_result, err, success_right = RRcontrol( right_hand_goal_transform, current_right_joints, debug=False )
    right_ik_result = bound_joints(right_ik_result)

    success = success_left and success_left
    if(success == False ):
        print("first part failed")
        print("left: ", current_left_joints)
        print("right: ", current_right_joints)
        print("left goal: ", left_hand_mid_goal)
        print("right goal: ", right_hand_mid_goal)
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        return None, None

   

    left_joints =  np.linspace(current_left_joints, left_ik_result, traj_length)
    left_gripper = np.linspace(current_joints[0][6], goals[0][7], traj_length)
    left_gripper = np.expand_dims(left_gripper, axis=1)
    left_traj = np.concatenate([left_joints, left_gripper], axis = 1)
    # print("left_traj1: ", left_traj1.shape)

    right_joints =  np.linspace(current_right_joints, right_ik_result, traj_length)
    right_gripper = np.linspace(current_joints[1][6], goals[1][7], traj_length)
    right_gripper = np.expand_dims(right_gripper, axis=1)
    right_traj = np.concatenate([right_joints, right_gripper], axis = 1)
    # print("left_traj1: ", left_traj1)
    # return left_traj1, right_traj1

   

    return left_traj, right_traj