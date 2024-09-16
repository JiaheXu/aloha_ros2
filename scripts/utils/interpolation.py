import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from numpy import linalg as LA
from utils import *
from math_tools import *

def traj_interpolation( trajectory, interpolation_length = 20):
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

    resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])

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

def get_trajectory(current_joins, mid_goals, goals):
    left_hand_mid_goal = left_stack[left_mid_point, 0:7]
    right_hand_mid_goal = right_stack[right_mid_point, 0:7]

    left_hand_mid_goal_transform = get_transform(left_hand_mid_goal)
    right_hand_mid_goal_transform = get_transform(right_hand_mid_goal)     
    left_ik_result1, err, success_left = RRcontrol( left_hand_mid_goal_transform, current_left_joints, debug=False )
    right_ik_result1, err, success_right = RRcontrol( right_hand_mid_goal_transform, current_right_joints, debug=False )
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
        return

    left_hand_goal = left_stack[-1, 0:7]
    right_hand_goal = right_stack[-1, 0:7]
    left_hand_goal_transform = get_transform(left_hand_goal)
    right_hand_goal_transform = get_transform(right_hand_goal)   
    left_ik_result2, err, success_left = RRcontrol( left_hand_goal_transform, left_ik_result1, debug=False )
    right_ik_result2, err, success_right = RRcontrol( right_hand_goal_transform, right_ik_result1, debug=False )
    if(success == False ):
        print("2nd part failed")
        print("left: ", left_ik_result1)
        print("right: ", right_ik_result1)
        print("left goal: ", left_hand_goal)
        print("right goal: ", right_hand_goal)
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        return