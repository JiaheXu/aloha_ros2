import cv2
import numpy as np
import open3d as o3d
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

def get_all_valid_depth( depth , xyz):
    for x in range( depth.shape[0] ):
        for y in range(depth.shape[1]):
            if( depth[x][y] > 0 ): # valid
                if( y + 1 < depth.shape[1] ):
                    if( depth[x][y+1] == 0 ):
                        depth[x][y+1] = depth[x][y]
                        xyz[x][y+1] = xyz[x][y]

    for x in range( depth.shape[0] ):
        for y in reversed( range(depth.shape[1]) ):
            if( depth[x][y] > 0 ): # valid
                if( y -1 >= 0 ):
                    if( depth[x][y-1]==0 ):
                        depth[x][y-1] = depth[x][y]
                        xyz[x][y-1] = xyz[x][y]

    for y in range( depth.shape[1] ):
        for x in range(depth.shape[0] ):
            if( depth[x][y] > 0 ): # valid
                if( x + 1 < depth.shape[0] ):
                    if( depth[x+1][y]==0 ):
                        depth[x+1][y] = depth[x][y]
                        xyz[x+1][y] = xyz[x][y]

    for y in range( depth.shape[1] ):
        for x in reversed( range(depth.shape[0] ) ):
            if( depth[x][y] > 0 ): # valid
                if( x - 1 >= 0 ):
                    if( depth[x-1][y]==0 ):
                        depth[x-1][y] = depth[x][y]
                        xyz[x-1][y] = xyz[x][y]
    return depth, xyz

def transfer_camera_param( bgr, depth, intrinsic_np, original_img_size, resized_intrinsic_np, resized_img_size):
    
    cx = intrinsic_np[0,2]
    cy = intrinsic_np[1,2]

    fx_factor = resized_intrinsic_np[0,0] / intrinsic_np[0,0]
    fy_factor = resized_intrinsic_np[1,1] / intrinsic_np[1,1]

    raw_fx = resized_intrinsic_np[0,0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
    raw_fy = resized_intrinsic_np[1,1] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
    raw_cx = resized_intrinsic_np[0,2] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
    raw_cy = resized_intrinsic_np[1,2] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]

    width = resized_img_size[0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
    height = resized_img_size[0] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
    
    half_width = int( width / 2.0 )
    half_height = int( height / 2.0 )

    cropped_bgr = bgr[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width), :]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.resize(cropped_rgb, resized_img_size)

    cropped_depth = depth[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width)]
    processed_depth = cv2.resize(cropped_depth, resized_img_size, interpolation =cv2.INTER_NEAREST)

    return processed_rgb, processed_depth

def xyz_from_depth(depth_image, depth_intrinsic, depth_extrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    ones = np.ones( ( depth_image.shape[0], depth_image.shape[1], 1) )
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = 0.0

    # print("xyz: ", xyz.shape)
    # print("ones: ", ones.shape)
    # print("depth_extrinsic: ", depth_extrinsic.shape)
    xyz = np.concatenate([xyz, ones], axis=2)
    xyz =  xyz @ np.transpose( depth_extrinsic)
    xyz = xyz[:,:,0:3]
    return xyz

def xyz_rgb_validation(rgb, xyz):
    # verify xyz and depth value
    valid_pcd = o3d.geometry.PointCloud()
    xyz = xyz.reshape(-1,3)
    rgb = (rgb/255.0).reshape(-1,3)
    valid_pcd.points = o3d.utility.Vector3dVector( xyz )
    valid_pcd.colors = o3d.utility.Vector3dVector( rgb )
    # visualize_pcd(valid_pcd)
