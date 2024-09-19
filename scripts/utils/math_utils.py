import numpy as np
from scipy.spatial.transform import Rotation
def get_transform( t_7d ):
    t = np.eye(4)
    trans = t_7d[0:3]
    quat = t_7d[3:7]
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t
def get_7D_transform(transf):
    trans = transf[0:3,3]
    trans = trans.reshape(3)
    quat = Rotation.from_matrix( transf[0:3,0:3] ).as_quat()
    quat = quat.reshape(4)
    return np.concatenate( [trans, quat])
