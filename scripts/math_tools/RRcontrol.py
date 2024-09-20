import time
import copy
import numpy as np
from math_tools.FwdKin import *
from math_tools.getXi import *
from math_tools.BodyJacobian import *
from numpy.linalg import inv

def RRcontrol(gdesired, q, K = 0.4, debug =True):

    dist_threshold = 0.003 # m
    angle_threshold = (10.0*np.pi)/180 # rad

    maxiter = 100
    current_q = copy.deepcopy(q)
    
    current_q = current_q.reshape(6,-1)
    start = time.time()
    i = 0

    for i in range(maxiter):
        gst = FwdKin(current_q)
        err = inv(gdesired) @ gst
        xi = getXi(err)
        
        J = BodyJacobian(current_q)
        grad = np.linalg.pinv(J) @ xi 
        # grad[0:3] *= 0.1
        current_q = current_q - K * grad

        J = BodyJacobian(current_q)
        finalerr = (LA.norm(xi[0:3]), LA.norm(xi[3:6]))
        # translation_err = LA.norm(xi[0:3])
        # rotation_err = LA.norm(xi[3:6])

        if abs(np.linalg.det(J))<0.0001:
            # print('Singularity position')
            current_q = current_q + 0.001
            # finalerr = -1
            # break
        
        if LA.norm(xi[0:3]) < dist_threshold and LA.norm(xi[3:6]) < angle_threshold :   
            break

    end = time.time()
    if debug:
        print("iter: ", i)
        print('Final error: {} cm     {}  rad'.format( finalerr[0]*10, finalerr[1]) )
        print("time cost: ", end - start)
    
    success = False
    current_q = current_q.reshape(-1)
    if(finalerr[0] < dist_threshold and finalerr[1] < angle_threshold):
        success = True
    return current_q, finalerr, success
