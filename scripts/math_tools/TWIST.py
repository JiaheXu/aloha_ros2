import numpy as np
from math_tools.SKEW3 import *
from math_tools.EXPCR import *
def TWIST(xi,theta):

    w = xi[3:]
    w = w.reshape(3,1)
    v = xi[0:3]
    v = v.reshape(3,1)

    w_hat = SKEW3(w)
    I = np.eye(3)

    Rotation = EXPCR( w * theta)
    Translation = (I-Rotation) @ w_hat @ v + (w @ w.transpose()) @ v * theta
    Translation = Translation.reshape(3,)
    g = np.eye(4)
    g[0:3, 0:3] = Rotation
    g[0:3, 3] = Translation
    return g
