import numpy as np
from numpy.linalg import inv
from math_tools.TWIST import *
from math_tools.SKEW3 import *
import time

np.set_printoptions(suppress=True)
def get_Ad( R, p_hat):
    Ad = np.zeros( (6,6) )
    Ad[0:3, 0:3] = R
    Ad[0:3, 3:6] = p_hat @ R 
    Ad[3:6, 3:6] = R
    return Ad

def BodyJacobian(q):
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

    g1 = inv(exp1 @ exp2 @ exp3 @ exp4 @ exp5 @ exp6 @ g0)
    g2 = inv(exp2 @ exp3 @ exp4 @ exp5 @ exp6 @ g0)
    g3 = inv(exp3 @ exp4 @ exp5 @ exp6 @ g0)
    g4 = inv(exp4 @ exp5 @ exp6 @ g0)
    g5 = inv(exp5 @ exp6 @ g0)
    g6 = inv(exp6 @ g0)

    R1 = g1[0:3,0:3]
    R2 = g2[0:3,0:3]
    R3 = g3[0:3,0:3]
    R4 = g4[0:3,0:3]
    R5 = g5[0:3,0:3]
    R6 = g6[0:3,0:3]

    p1 = g1[0:3,3]
    p2 = g2[0:3,3]
    p3 = g3[0:3,3]
    p4 = g4[0:3,3]
    p5 = g5[0:3,3]
    p6 = g6[0:3,3]

    p1_hat = SKEW3(p1)
    p2_hat = SKEW3(p2)
    p3_hat = SKEW3(p3)
    p4_hat = SKEW3(p4)
    p5_hat = SKEW3(p5)
    p6_hat = SKEW3(p6)

    Ad1 = get_Ad(R1, p1_hat)
    Ad2 = get_Ad(R2, p2_hat)
    Ad3 = get_Ad(R3, p3_hat)
    Ad4 = get_Ad(R4, p4_hat)
    Ad5 = get_Ad(R5, p5_hat)
    Ad6 = get_Ad(R6, p6_hat)

    xi1_prim = Ad1 @ xi1
    xi2_prim = Ad2 @ xi2
    xi3_prim = Ad3 @ xi3
    xi4_prim = Ad4 @ xi4
    xi5_prim = Ad5 @ xi5
    xi6_prim = Ad6 @ xi6

    JB = np.array( [xi1_prim,xi2_prim,xi3_prim,xi4_prim,xi5_prim,xi6_prim] ).transpose()
    # print("JB: ", JB.shape)
    return JB

if __name__ == "__main__":
    q = [0.2, 0.4, 0.4, 0.3, 0.4, 0.4]
    q = np.array(q)
    print("q: ",q)
    start = time.time()
    # print(  )
    BodyJacobian(q)
    end = time.time()
    print( "time (ms): ", (end - start)*1000 )
