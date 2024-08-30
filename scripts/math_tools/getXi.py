import numpy as np
from numpy import linalg as LA
from math_tools.VECTORIZE import *
from math_tools.SKEW3 import *
from numpy.linalg import inv

def getXi(g):
    
    R = g[0:3, 0:3]
    p = g[0:3, 3]

    # % Compute the rotation angle theta
    theta = np.arccos( (np.trace(R) - 1) / 2)

    if abs(theta) < 1e-4:
        omega = np.array( [0, 0, 0])
        v = p / LA.norm(p,2)
        theta = LA.norm(p,2)
    else:
        
        omega = 1 / (2*np.sin(theta)) * VECTORIZE( R-R.transpose() )
        omega_hat = SKEW3(omega)
        G = np.eye(3)*theta + (1- np.cos(theta))*omega_hat + (theta-np.sin(theta))*(omega_hat@omega_hat)
        G_inv = inv(G)
        v = G_inv @ p

    xi = np.zeros( (6,1) )
    v = v.reshape(3,1)
    omega = omega.reshape(3,1)
    xi[0:3] = v
    xi[3:6] = omega
    xi = xi*theta
    return xi


