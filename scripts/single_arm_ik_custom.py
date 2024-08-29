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


def RRcontrol(gdesired, q, K):

    dist_threshold = 0.005
    angle_threshold = (0.15*np.pi)/180
    Tstep = 0.1
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
            print('Singularity position \n')
            finalerr = -1
            break
        
        if LA.norm(xi[0:3]) < dist_threshold and LA.norm(xi[3:6]) < angle_threshold :
            finalerr = LA.norm(xi[0:3])*10
            print('Convergence achieved. Final error: {} cm\n'.format( finalerr) )
            break;
    end = time.time()
    print("time cost: ", end - start)
    return current_q, finalerr

def custom_ik( goal_ee_7D, current_joints, debug=False ):
    goal_transform = get_transform(goal_ee_7D)
    K = 0.5
    success = False
    result_q, finalerr =  RRcontrol(goal_transform, current_joints, K)
    if(finalerr != -1):
        success = True
    return result_q, finalerr, success

def main() -> None:
   
    episode = np.load("1.npy", allow_pickle = True)

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
    # K = 0.1
    # current_joints = np.array( [0.3190680146217346, -0.43411657214164734, 0.8789710402488708, 0.647339940071106, 0.26077672839164734, -0.5737088322639465])
    # ik_result, err = RRcontrol(gdesired, current_joints , K)
    # print("ik_result: ", ik_result)
    # print("err: ", err)    

    # for data_point in episode[50:53]:
    for data_point in episode:
        current_joints = last_joints
        data_point["right_ee"][1] += 0.315
        
        # ik_result, err, success = get_ik(  model, data, data_point["right_ee"], current_joints, debug=False )
        ik_result, err, success = custom_ik( data_point["right_ee"], current_joints, debug=False )
        if( success == False):
            print("failed !!!!!!!!!!!!")
            print("err: ", err)
        joints = data_point["right_pos"][0:6]
        # print("joint diff: ", np.abs(last_joints_np - joints))
        last_joints_np = np.array( joints )
        # print("EE: ", joints[-1])
        # print("Goal: ", data_point["right_ee"][0:3])

    # print("finished !!!!!!!!!!" )



if __name__ == '__main__':
    main()

# ### ALOHA Fixed Constants
# DT = 0.02
#     from rclpy.duration import Duration
#     from rclpy.constants import S_TO_NS
#     DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)