import numpy as np
from SKEW3 import *
from EXPCR import *
def twist(xi,theta):

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
