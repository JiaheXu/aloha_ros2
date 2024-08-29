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
 
from BodyJacobian import *
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm
from FwdKin import *
from getXi import *
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


def RRcontrol(gdesired, q, K):

    dist_threshold = 0.5
    angle_threshold = (1.0*np.pi)/180
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
        finalerr = -1
        J = BodyJacobian(current_q)

        if abs(np.linalg.det(J))<0.001:
            print('Singularity position')
            current_q = current_q + 0.01
            finalerr = -1
            break
        
        if LA.norm(xi[0:3]) < dist_threshold and LA.norm(xi[3:6]) < angle_threshold :
            finalerr = LA.norm(xi[0:3])*10
            print('Convergence achieved. Final error: {} cm'.format( finalerr) )
            break;
    end = time.time()
    print("time cost: ", end - start)
    return current_q, finalerr

def custom_ik( goal_ee_7D, current_joints, debug=False ):
    goal_transform = get_transform(goal_ee_7D)
    K = 0.8
    success = False
    result_q, finalerr =  RRcontrol(goal_transform, current_joints, K)
    if(finalerr != -1):
        success = True

    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success

def main() -> None:
   
    episode = np.load("2.npy", allow_pickle = True)

    urdf_filename = (
        "../urdf/vx300s.urdf"
    )
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    data = model.createData()
    # last_joints = episode[0]["right_pos"][0:6].tolist()
    last_joints = episode[0]["right_pos"][0:6]
    last_joints_np = episode[0]["right_pos"][0:6]

    # gdesired = np.array( [
    #     [ 0.40977936, -0.39541217,  0.82202804,  0.41683976],
    #     [0.72954536,  0.68302983, -0.03512571,  0.15501925],
    #     [ -0.54758054,  0.61410053,  0.56836264,  0.14072071],
    #     [0.,          0.,          0.,          1.        ]
    #     ])

    gdesired = np.array( [
        [ 0.36335202, -0.01862422,  0.93146575,  0.34016989],
        [ 0.57107482,  0.79440055, -0.20688479,  0.13208353],
        [-0.73610384,  0.60710864,  0.29928287,  0.07789062],
        [ 0.,          0.,          0.,          1.        ]]
    )

    K = 0.1
    # current_joints = np.array( [0.3190680146217346, -0.43411657214164734, 0.8789710402488708, 0.647339940071106, 0.26077672839164734, -0.5737088322639465])
    # ik_result, err = RRcontrol(gdesired, current_joints , K)
    # print("ik_result: ", ik_result)
    # print("err: ", err)    



    total_success = 0
    # for data_point in episode[50:53]:
    for idx, data_point in enumerate( episode ):
        current_joints = last_joints
        data_point["right_ee"][1] += 0.315
        print("\nidx: ", idx)
        # ik_result, err, success = get_ik(  model, data, data_point["right_ee"], current_joints, debug=False )
        ik_result, err, success = custom_ik( data_point["right_ee"], current_joints, debug=False )
        if( success == False):
            print("failed !!!!!!!!!!!!")
            print("err: ", err)
            print("gt: ", data_point['right_pos'])
            print("trans: ", get_transform( data_point['right_ee'] ) )
            # print("")
        else:
            total_success += 1
        joints = data_point["right_pos"][0:6]
        # print("joint diff: ", np.abs(last_joints_np - joints))
        last_joints = np.array( joints )
        # print("EE: ", joints[-1])
        # print("Goal: ", data_point["right_ee"][0:3])
    print("success: {} / {} ".format(str( total_success ), str( len(episode))) )
    # print("finished !!!!!!!!!!" )



if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)