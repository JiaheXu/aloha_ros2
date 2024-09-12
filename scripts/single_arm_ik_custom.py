#!/usr/bin/env python3



import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from sys import argv
from os.path import dirname, join, abspath
import time
from numpy.linalg import norm, solve
import copy
 

from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm

from utils import *
from math_tools import *
import threading
import time


def main() -> None:
   
    gdesired = get_transform( [0.1674,  0.3201,  0.202,   0.0907, -0.0097,  0.006,   0.9958] )
    gdesired[1,3] -= 0.315 

    K = 0.4
    current_joints = np.array( [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # current_joints = np.array( [0., 0., 0., 0.1, 0.1, 0.1])
    start = time.time()
    ik_result, err, success = RRcontrol(gdesired, current_joints , K)
    end = time.time()
    # print("took time: ", end - start)
    print("ik_result: ", ik_result)
    print("err: ", err)
    print("success: ", success)







if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)