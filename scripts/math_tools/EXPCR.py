import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm
from math_tools.SKEW3 import *
def EXPCR(X):

    theta = LA.norm(X)
    # print("X: ", X)
    # print("theta: ", theta)
    rotation = None

    if abs( theta ) < 1e-5:
        rotation = np.eye(3,3)
    else:
        n = X / theta
        n_hat = SKEW3(n)
        rotation = expm(n_hat * theta)
        # print("rotation: ", rotation)

    return rotation
