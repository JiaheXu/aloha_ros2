import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from numpy import linalg as LA

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