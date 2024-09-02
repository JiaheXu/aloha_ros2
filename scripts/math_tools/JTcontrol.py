import time
import copy
import numpy as np
def TJcontrol(gdesired, q, K, debug =True):

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

        current_q = current_q - K * Tstep * ( J.Transpose() )  @ xi

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
    if debug:
        print('Convergence achieved. Final error: {} cm     {}  rad'.format( finalerr[0]*10, finalerr[1]) )
        print("time cost: ", end - start)
    
    success = False
    
    if(finalerr[0] < dist_threshold and finalerr[1] < angle_threshold):
        success = True
    
    return current_q, finalerr, success
