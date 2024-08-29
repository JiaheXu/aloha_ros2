#!/usr/bin/env python3



import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from sys import argv
from os.path import dirname, join, abspath
import pinocchio
import time
from numpy.linalg import norm, solve
import copy
 



def get_transform( t_7d ):
    t = np.eye(4)
    trans = t_7d[0:3]
    quat = t_7d[3:7]
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t


def roundup_pi(current_joint, next_steps):
    rounded_result = []
    for i in range( next_steps.shape[0] ):
        rouneded = next_steps[i] - (next_steps[i] // np.pi) * np.pi
        rounded_result.append(rouneded)
    return rounded_result
# def jacobian_spacial(q):
#     q1 = q[0]
#     q2 = q[1]
#     q3 = q[2]
#     q4 = q[3]
#     q5 = q[4]
#     q6 = q[5]    
# JS =
 
# [                                                                                                                                                                                                                                                           -sign(L5*cos(q2 + q3)*sin(q1) + L6*cos(q2 + q3)*sin(q1) + L4*cos(q2)*sin(q1) + L3*sin(q1)*sin(q2) + L7*cos(q2 + q3)*cos(q5)*sin(q1) + L8*cos(q2 + q3)*cos(q5)*sin(q1) + L7*cos(q1)*sin(q4)*sin(q5) + L8*cos(q1)*sin(q4)*sin(q5) - L7*cos(q2)*cos(q4)*sin(q1)*sin(q3)*sin(q5) - L7*cos(q3)*cos(q4)*sin(q1)*sin(q2)*sin(q5) - L8*cos(q2)*cos(q4)*sin(q1)*sin(q3)*sin(q5) - L8*cos(q3)*cos(q4)*sin(q1)*sin(q2)*sin(q5))*Inf,                     -sign(cos(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) - L3*cos(q2) + L4*sin(q2) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                               -sign(cos(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                                                                                                                                                                                                                                                                                                                                                            sin(q5)*(L7 + L8)*(- cos(q4)*sin(q1) + cos(q1)*cos(q2)*sin(q3)*sin(q4) + cos(q1)*cos(q3)*sin(q2)*sin(q4))*Inf,                                                                                                                                                                                                                                                                                                                               -sign((L7 + L8)*(cos(q2 + q3)*cos(q1)*sin(q5) + cos(q5)*sin(q1)*sin(q4) + cos(q1)*cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q1)*cos(q3)*cos(q4)*cos(q5)*sin(q2)))*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0]
# [                                                                                                                                                                                                                                                                (L5*cos(q2 + q3)*cos(q1) + L6*cos(q2 + q3)*cos(q1) + L4*cos(q1)*cos(q2) + L3*cos(q1)*sin(q2) - L7*sin(q1)*sin(q4)*sin(q5) - L8*sin(q1)*sin(q4)*sin(q5) + L7*cos(q2 + q3)*cos(q1)*cos(q5) + L8*cos(q2 + q3)*cos(q1)*cos(q5) - L7*cos(q1)*cos(q2)*cos(q4)*sin(q3)*sin(q5) - L7*cos(q1)*cos(q3)*cos(q4)*sin(q2)*sin(q5) - L8*cos(q1)*cos(q2)*cos(q4)*sin(q3)*sin(q5) - L8*cos(q1)*cos(q3)*cos(q4)*sin(q2)*sin(q5))*Inf,                     -sign(sin(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) - L3*cos(q2) + L4*sin(q2) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                               -sign(sin(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                                                                                                                                                                                                                                                                                                                                                              sin(q5)*(L7 + L8)*(cos(q1)*cos(q4) + cos(q2)*sin(q1)*sin(q3)*sin(q4) + cos(q3)*sin(q1)*sin(q2)*sin(q4))*Inf,                                                                                                                                                                                                                                                                                                                               -sign((L7 + L8)*(cos(q2 + q3)*sin(q1)*sin(q5) - cos(q1)*cos(q5)*sin(q4) + cos(q2)*cos(q4)*cos(q5)*sin(q1)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q1)*sin(q2)))*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0]
# [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0,                                       -sign(L5*cos(q2 + q3) + L6*cos(q2 + q3) + L4*cos(q2) + L3*sin(q2) - (L7*sin(q2 + q3)*sin(q4 + q5))/2 - (L8*sin(q2 + q3)*sin(q4 + q5))/2 + L7*cos(q2 + q3)*cos(q5) + L8*cos(q2 + q3)*cos(q5) + (L7*sin(q4 - q5)*sin(q2 + q3))/2 + (L8*sin(q4 - q5)*sin(q2 + q3))/2)*Inf,                                                                                                                                         -sign(L5*cos(q2 + q3) + L6*cos(q2 + q3) + L7*cos(q2 + q3)*cos(q5) + L8*cos(q2 + q3)*cos(q5) - L7*sin(q2 + q3)*cos(q4)*sin(q5) - L8*sin(q2 + q3)*cos(q4)*sin(q5))*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                               cos(q2 + q3)*sin(q4)*sin(q5)*(L7 + L8)*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                             (sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5))*(L7 + L8)*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0]
# [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0,                                                   -(- sin(q2 + q3)*cos(q4)*sin(q5) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) + cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                   -(- sin(q2 + q3)*cos(q4)*sin(q5) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) + cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                                                                            -cos(q2 + q3)*(- cos(q4)*cos(q6) + cos(q4)*sin(q6) - sin(q4)*sin(q5) + cos(q5)*cos(q6)*sin(q4) + cos(q5)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                    -(- sin(q2 + q3)*sin(q5) + sin(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) + cos(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) + cos(q2 + q3)*cos(q4)*cos(q5))*Inf,                                                                                                                                                                                                                                                                                                                           -(- sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + cos(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q6)*sin(q4) + cos(q2 + q3)*sin(q4)*sin(q6))*Inf]
# [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0,                                                      (cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) - sin(q2 + q3)*cos(q4)*sin(q5) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                      (cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) - sin(q2 + q3)*cos(q4)*sin(q5) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                                                                             cos(q2 + q3)*(- cos(q4)*cos(q6) + cos(q4)*sin(q6) - sin(q4)*sin(q5) + cos(q5)*cos(q6)*sin(q4) + cos(q5)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                       (cos(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) + sin(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) - sin(q2 + q3)*sin(q5) + cos(q2 + q3)*cos(q4)*cos(q5))*Inf,                                                                                                                                                                                                                                                                                                                              (cos(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) - sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q6)*sin(q4) + cos(q2 + q3)*sin(q4)*sin(q6))*Inf]
# [(sin(q6)*(cos(q4)*sin(q1) - sin(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) - sin(q5)*(sin(q1)*sin(q4) + cos(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) - cos(q6)*(cos(q4)*sin(q1) - sin(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) + cos(q6)*(cos(q5)*(sin(q1)*sin(q4) + cos(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) + sin(q5)*(cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3))) + sin(q6)*(cos(q5)*(sin(q1)*sin(q4) + cos(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) + sin(q5)*(cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3))) + cos(q5)*(cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3)))*Inf, -sign(cos(q6)*sin(q1)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q1)*sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q5)*sin(q1) + cos(q2 + q3)*cos(q4)*sin(q1)*sin(q5) - cos(q2 + q3)*cos(q6)*sin(q1)*sin(q4) + cos(q2 + q3)*sin(q1)*sin(q4)*sin(q6))*Inf, -sign(cos(q6)*sin(q1)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q1)*sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q5)*sin(q1) + cos(q2 + q3)*cos(q4)*sin(q1)*sin(q5) - cos(q2 + q3)*cos(q6)*sin(q1)*sin(q4) + cos(q2 + q3)*sin(q1)*sin(q4)*sin(q6))*Inf, -(- sin(q5)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q6)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) - sin(q6)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q5)*cos(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q5)*sin(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))))*Inf, (cos(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q6)*(sin(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) - cos(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + sin(q6)*(sin(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) - cos(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + sin(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1)))*Inf, -(- sin(q6)*(cos(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + sin(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + sin(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q6)*(cos(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + sin(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + cos(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))))*Inf]
 
 
# JB =
 
# [                                                                                                                                                                                                                                                           -sign(L5*cos(q2 + q3)*sin(q1) + L6*cos(q2 + q3)*sin(q1) + L4*cos(q2)*sin(q1) + L3*sin(q1)*sin(q2) + L7*cos(q2 + q3)*cos(q5)*sin(q1) + L8*cos(q2 + q3)*cos(q5)*sin(q1) + L7*cos(q1)*sin(q4)*sin(q5) + L8*cos(q1)*sin(q4)*sin(q5) - L7*cos(q2)*cos(q4)*sin(q1)*sin(q3)*sin(q5) - L7*cos(q3)*cos(q4)*sin(q1)*sin(q2)*sin(q5) - L8*cos(q2)*cos(q4)*sin(q1)*sin(q3)*sin(q5) - L8*cos(q3)*cos(q4)*sin(q1)*sin(q2)*sin(q5))*Inf,                     -sign(cos(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) - L3*cos(q2) + L4*sin(q2) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                               -sign(cos(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                                                                                                                                                                                                                                                                                                                                                            sin(q5)*(L7 + L8)*(- cos(q4)*sin(q1) + cos(q1)*cos(q2)*sin(q3)*sin(q4) + cos(q1)*cos(q3)*sin(q2)*sin(q4))*Inf,                                                                                                                                                                                                                                                                                                                               -sign((L7 + L8)*(cos(q2 + q3)*cos(q1)*sin(q5) + cos(q5)*sin(q1)*sin(q4) + cos(q1)*cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q1)*cos(q3)*cos(q4)*cos(q5)*sin(q2)))*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0]
# [                                                                                                                                                                                                                                                                (L5*cos(q2 + q3)*cos(q1) + L6*cos(q2 + q3)*cos(q1) + L4*cos(q1)*cos(q2) + L3*cos(q1)*sin(q2) - L7*sin(q1)*sin(q4)*sin(q5) - L8*sin(q1)*sin(q4)*sin(q5) + L7*cos(q2 + q3)*cos(q1)*cos(q5) + L8*cos(q2 + q3)*cos(q1)*cos(q5) - L7*cos(q1)*cos(q2)*cos(q4)*sin(q3)*sin(q5) - L7*cos(q1)*cos(q3)*cos(q4)*sin(q2)*sin(q5) - L8*cos(q1)*cos(q2)*cos(q4)*sin(q3)*sin(q5) - L8*cos(q1)*cos(q3)*cos(q4)*sin(q2)*sin(q5))*Inf,                     -sign(sin(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) - L3*cos(q2) + L4*sin(q2) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                               -sign(sin(q1)*(L5*sin(q2 + q3) + L6*sin(q2 + q3) + L7*sin(q2 + q3)*cos(q5) + L8*sin(q2 + q3)*cos(q5) + L7*cos(q2)*cos(q3)*cos(q4)*sin(q5) + L8*cos(q2)*cos(q3)*cos(q4)*sin(q5) - L7*cos(q4)*sin(q2)*sin(q3)*sin(q5) - L8*cos(q4)*sin(q2)*sin(q3)*sin(q5)))*Inf,                                                                                                                                                                                                                                                                                                                                                                              sin(q5)*(L7 + L8)*(cos(q1)*cos(q4) + cos(q2)*sin(q1)*sin(q3)*sin(q4) + cos(q3)*sin(q1)*sin(q2)*sin(q4))*Inf,                                                                                                                                                                                                                                                                                                                               -sign((L7 + L8)*(cos(q2 + q3)*sin(q1)*sin(q5) - cos(q1)*cos(q5)*sin(q4) + cos(q2)*cos(q4)*cos(q5)*sin(q1)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q1)*sin(q2)))*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0]
# [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0,                                       -sign(L5*cos(q2 + q3) + L6*cos(q2 + q3) + L4*cos(q2) + L3*sin(q2) - (L7*sin(q2 + q3)*sin(q4 + q5))/2 - (L8*sin(q2 + q3)*sin(q4 + q5))/2 + L7*cos(q2 + q3)*cos(q5) + L8*cos(q2 + q3)*cos(q5) + (L7*sin(q4 - q5)*sin(q2 + q3))/2 + (L8*sin(q4 - q5)*sin(q2 + q3))/2)*Inf,                                                                                                                                         -sign(L5*cos(q2 + q3) + L6*cos(q2 + q3) + L7*cos(q2 + q3)*cos(q5) + L8*cos(q2 + q3)*cos(q5) - L7*sin(q2 + q3)*cos(q4)*sin(q5) - L8*sin(q2 + q3)*cos(q4)*sin(q5))*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                               cos(q2 + q3)*sin(q4)*sin(q5)*(L7 + L8)*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                             (sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5))*(L7 + L8)*Inf,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0]
# [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0,                                                   -(- sin(q2 + q3)*cos(q4)*sin(q5) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) + cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                   -(- sin(q2 + q3)*cos(q4)*sin(q5) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) + cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                                                                            -cos(q2 + q3)*(- cos(q4)*cos(q6) + cos(q4)*sin(q6) - sin(q4)*sin(q5) + cos(q5)*cos(q6)*sin(q4) + cos(q5)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                    -(- sin(q2 + q3)*sin(q5) + sin(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) + cos(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) + cos(q2 + q3)*cos(q4)*cos(q5))*Inf,                                                                                                                                                                                                                                                                                                                           -(- sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + cos(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q6)*sin(q4) + cos(q2 + q3)*sin(q4)*sin(q6))*Inf]
# [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0,                                                      (cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) - sin(q2 + q3)*cos(q4)*sin(q5) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                      (cos(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + sin(q6)*(cos(q2 + q3)*sin(q5) + sin(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q5) - sin(q2 + q3)*cos(q4)*sin(q5) + sin(q2 + q3)*cos(q6)*sin(q4) - sin(q2 + q3)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                                                                             cos(q2 + q3)*(- cos(q4)*cos(q6) + cos(q4)*sin(q6) - sin(q4)*sin(q5) + cos(q5)*cos(q6)*sin(q4) + cos(q5)*sin(q4)*sin(q6))*Inf,                                                                                                                                                                                                                                                                                                       (cos(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) + sin(q6)*(sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)) - sin(q2 + q3)*sin(q5) + cos(q2 + q3)*cos(q4)*cos(q5))*Inf,                                                                                                                                                                                                                                                                                                                              (cos(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) - sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + cos(q2 + q3)*cos(q6)*sin(q4) + cos(q2 + q3)*sin(q4)*sin(q6))*Inf]
# [(sin(q6)*(cos(q4)*sin(q1) - sin(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) - sin(q5)*(sin(q1)*sin(q4) + cos(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) - cos(q6)*(cos(q4)*sin(q1) - sin(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) + cos(q6)*(cos(q5)*(sin(q1)*sin(q4) + cos(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) + sin(q5)*(cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3))) + sin(q6)*(cos(q5)*(sin(q1)*sin(q4) + cos(q4)*(cos(q1)*cos(q2)*sin(q3) + cos(q1)*cos(q3)*sin(q2))) + sin(q5)*(cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3))) + cos(q5)*(cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3)))*Inf, -sign(cos(q6)*sin(q1)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q1)*sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q5)*sin(q1) + cos(q2 + q3)*cos(q4)*sin(q1)*sin(q5) - cos(q2 + q3)*cos(q6)*sin(q1)*sin(q4) + cos(q2 + q3)*sin(q1)*sin(q4)*sin(q6))*Inf, -sign(cos(q6)*sin(q1)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q1)*sin(q6)*(sin(q2 + q3)*sin(q5) - cos(q2 + q3)*cos(q4)*cos(q5)) + sin(q2 + q3)*cos(q5)*sin(q1) + cos(q2 + q3)*cos(q4)*sin(q1)*sin(q5) - cos(q2 + q3)*cos(q6)*sin(q1)*sin(q4) + cos(q2 + q3)*sin(q1)*sin(q4)*sin(q6))*Inf, -(- sin(q5)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q6)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) - sin(q6)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q5)*cos(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q5)*sin(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))))*Inf, (cos(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q6)*(sin(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) - cos(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + sin(q6)*(sin(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) - cos(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + sin(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1)))*Inf, -(- sin(q6)*(cos(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + sin(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + sin(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + cos(q6)*(cos(q5)*(cos(q1)*sin(q4) - cos(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))) + sin(q5)*(sin(q1)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*sin(q1))) + cos(q6)*(cos(q1)*cos(q4) + sin(q4)*(cos(q2)*sin(q1)*sin(q3) + cos(q3)*sin(q1)*sin(q2))))*Inf]
#                                                                                                                                                                                                                                                                                                                                                       cos(q2 + q3)*sin(q4),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2*cos(q4)*sin(q2)*sin(q3)*sin(q5) - 2*cos(q3)*cos(q5)*sin(q2) - 2*cos(q2)*cos(q3)*cos(q4)*sin(q5) - 2*cos(q2)*cos(q5)*sin(q3)]
def get_ik(
    model, data, goal_ee_7D, current_joints,
    debug = False
) -> list:
    # solve ik, 6dof, gripper not included
    eps = 1e-3
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-12

    JOINT_ID = 7

    goal_transform = get_transform(goal_ee_7D)
    rot = goal_transform[0:3, 0:3]
    trans = goal_transform[0:3, 3]
    oMdes = pinocchio.SE3( rot, trans)
    # print("oMdes: ", oMdes)
    start = time.time()
    q = copy.deepcopy( current_joints )
    q.append(0.0)
    q.append(0.0)
    q = np.array(q)

    # q = pinocchio.neutral(model)
    i = 0
    err = None
    while True:
        # forward_start = time.time()
        pinocchio.forwardKinematics(model, data, q)
        # forward_end = time.time()

        # print("foward: ", forward_end - forward_start)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector  # in joint frame
        # print("err: ", err.shape)
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
        # print("J:", J.shape) # (6, 7)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        # print("J:", J.shape) # (6, 7)

        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err)) # original method
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
    
    for name, oMi in zip(model.names, data.oMi):
        # joints.append( [*oMi.rotation] )
        joints.append( oMi )

    rot = (Rotation.from_matrix( goal_transform[0:3, 0:3]) )
    
    result_quat = pinocchio.Quaternion(joints[-1].rotation).coeffs()
    result_rot = (Rotation.from_quat(result_quat) )

    dot_result = result_rot.as_quat() * rot.as_quat()
    rotation_err = abs (1.0 - abs( dot_result[3] ) )

    if(rotation_err < 0.3 and np.sum( err[0:3] ) < 3*eps ):
        success = True
    else:
        print("rotation_err: ", rotation_err)
        print("translate: ", np.sum( err[0:3] ))
        # print("dot_result: ", dot_result)
        # debug = True
        print("goal: ", goal_transform)
        print("current_joints: ", current_joints)
        # print("result: ", joints[-1].translation)
        # print("err: ", err)

    if debug:
        print("ik_result: ", q)
        print("\nfinal error: %s" % err.T)
        print("goal: ", goal_transform[0:3, 0:3])

        print("matrix: ", joints[-1].homogeneous )
        print("quat: ", pinocchio.Quaternion(joints[-1].rotation) )

        result_quat = pinocchio.Quaternion(joints[-1].rotation).coeffs()
        result_rot = (Rotation.from_quat(result_quat) )
        print("result_rot: ", result_rot.as_matrix())
        rot = (Rotation.from_matrix( goal_transform[0:3, 0:3]) )
        print("original: ", rot.as_matrix())
        print("difference: ", result_rot.as_matrix() @ (rot.as_matrix()).T)
        print("are_equal: ", result_rot.as_quat() * rot.as_quat())
    return q, err, success



def main() -> None:
   
    episode = np.load("2.npy", allow_pickle = True)

    urdf_filename = (
        "../urdf/vx300s.urdf"
    )
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    data = model.createData()
    last_joints = episode[0]["right_pos"][0:6].tolist()
    last_joints_np = episode[0]["right_pos"][0:6]
    # for data_point in episode[0:3]:
    for data_point in episode:
        current_joints = last_joints

        data_point["right_ee"][1] += 0.315

        ik_result, err, success = get_ik(  model, data, data_point["right_ee"], current_joints, debug=False )
        # ik_result = search_ik(  model, data, data_point["right_ee"], current_joints, debug=False )
        # print("data_point: ", data_point["right_ee"])
        if( success == False):
            print("failed !!!!!!!!!!!!")
            print("failed !!!!!!!!!!!!")
            print("failed !!!!!!!!!!!!")

        # last_joints = ik_result[0:6].tolist()

        joints = data_point["right_pos"][0:6]
        print("joint diff: ", np.abs(last_joints_np - joints))
        last_joints_np = np.array( joints )
        



        # print("EE: ", joints[-1])
        # print("Goal: ", data_point["right_ee"][0:3])

    print("finished !!!!!!!!!!" )



if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)