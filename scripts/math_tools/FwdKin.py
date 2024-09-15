import numpy as np
from math_tools.TWIST import *

def FwdKin(q):

    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]
    theta5 = q[4]
    theta6 = q[5]

    xi1 = np.array( [0.,0.,0.,           0.,0.,1.])  #z
    xi2 = np.array( [-0.127, 0., 0.,       0.,1.,0.]) #y
    xi3 = np.array( [-0.427, 0., 0.0595,       0.,1.,0.]) #y
    xi4 = np.array( [0., 0.427, 0.,    1.,0.,0.] )#x
    xi5 = np.array( [-0.427, 0. ,0.3596,         0.,1.,0.] )#y
    xi6 = np.array( [0., 0.427, 0.,          1.,0.,0] )#x

    exp1 = TWIST(xi1,theta1)
    exp2 = TWIST(xi2,theta2)
    exp3 = TWIST(xi3,theta3)
    exp4 = TWIST(xi4,theta4)
    exp5 = TWIST(xi5,theta5)
    exp6 = TWIST(xi6,theta6)


    g0 = np.eye(4)
    g0[0,3] = 0.5361
    g0[2,3] = 0.427

    gst = exp1 @ exp2 @ exp3 @ exp4 @ exp5 @ exp6 @ g0

    return gst

if __name__ == "__main__":
    q = [0., 0.4, 0.4, 0.4, 0., 0.]
    q = np.array(q)
    print("q: ",q)
    print( FwdKin(q) )
