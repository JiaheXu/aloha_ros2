import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped

def get_transform( t_7d ):
    t_7d = t_7d.reshape(-1, 7)
    # print("t_7d: ", t_7d)
    t = np.eye(4)
    trans = t_7d[:,0:3]
    quat = t_7d[:,3:7]
    # print("quat: ", quat)
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
    
def load_yaml(file_dir):
    # Load the YAML file
    with open(file_dir, "r") as file:
        data = yaml.safe_load(file)

    return data

def transform_to_numpy(ros_transformation):
    x = ros_transformation.transform.translation.x
    y = ros_transformation.transform.translation.y
    z = ros_transformation.transform.translation.z
    
    qx = ros_transformation.transform.rotation.x
    qy = ros_transformation.transform.rotation.y
    qz = ros_transformation.transform.rotation.z
    qw = ros_transformation.transform.rotation.w

    return np.array( [x, y, z, qx, qy, qz, qw] )

