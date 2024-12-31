import trimesh
import numpy as np
from PIL import Image
import os
import argparse
import cv2
import open3d as o3d
import copy
#import torch
import plyfile

# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv, det
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
from utils import *

def mesh2points(mesh, num_points = 1000):
    """
    Convert a mesh to a list of points.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def numpy_2_pcd(pcd_np, color = None ):

    # pcd_np = np.array(pcd_np)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    if(color is not None):
        color_np = np.zeros(pcd_np.shape)
        color_np[:,] = color
        pcd.colors = o3d.utility.Vector3dVector(color_np)
    return pcd

def pcd_2_numpy(pcd):

    pcd_np = np.asarray(pcd.points)

    return pcd_np


def align_meshes_pcd(mesh, obj_pcd, bgd_pcd, max_iters=100, threshold=1e-6):
    """
    Align a mesh to a point cloud.
    """
    # Sample points on the surface of the mesh
    
    original_obj_pcd = obj_pcd

    obj_np = pcd_2_numpy(obj_pcd)

    mesh_points = mesh2points(mesh, obj_np.shape[0]*2)
    mesh_center = mesh_points.mean(axis=0)
    mesh_points -= mesh_center

    obj_points = pcd_2_numpy(obj_pcd)
    obj_center = obj_points.mean(axis=0)
    obj_points -= obj_center

    print("mesh_points: ", mesh_points.shape)
    print("obj_points: ", obj_points.shape)

    mesh_scale = np.linalg.norm(mesh_points, axis=0).max()
    pcd_scale = np.linalg.norm(obj_points, axis=0).max()

    rough_scale = pcd_scale / mesh_scale

    print("pcd_scale: ", pcd_scale)
    print("mesh_scale: ", mesh_scale)
    scaled_mesh_points = (mesh_points)*rough_scale


    # visualize_pcds( 
    #     [
    #         numpy_2_pcd(scaled_mesh_points), 
    #         numpy_2_pcd( obj_points) , 
    #         # bgd_pcd
    #     ]
    #     )

    # Compute the transformation matrix
    T, transformed, cost = trimesh.registration.icp(scaled_mesh_points, 
                                                    obj_points,
                                                    max_iterations=max_iters,
                                                    reflection=False,
                                                    scale=True,
                                                    threshold=threshold)
    final_pcd = numpy_2_pcd(scaled_mesh_points)
    final_pcd.transform( T )
    
    
    print("Transform: ", T)
    det_T = det(T[0:3, 0:3])
    T_scale = det_T ** (1/3)
    T[0:3, 0:3] /= T_scale
    T[0:3, 3] += obj_center
    final_scale = rough_scale * T_scale
    # print("scale: ", det_T )
    # print("normed: ", det( T[0:3, 0:3]) )

    scaled_mesh_points = (mesh_points)*final_scale
    scaled_mesh_pcd = numpy_2_pcd( scaled_mesh_points, np.array( [1., 0., 0.] ) )
    scaled_mesh_pcd.transform( T )
    # visualize_pcds( 
    #     [
    #         final_pcd,
    #         numpy_2_pcd( obj_points) , 
    #     ]
    #     )
    print("final result: ")
    print("mesh scale: ", final_scale)
    print("transform: ", T)
    trans_7d = get_7D_transform( T )
    print("quat: ", trans_7d[3:])
    visualize_pcds( 
        [
            scaled_mesh_pcd,
            # original_obj_pcd,
            bgd_pcd,
            # numpy_2_pcd( obj_points) , 
        ]
        )

def get_depth(depth_image, mask):
    depth_mask = np.where(mask > 0, 1, 0)
    depth_image = depth_image * depth_mask
    return depth_image

def depth_to_pcl(depth_image, intrinsics):
    """
    Convert a depth image to a point cloud.
    """
    depth_image = depth_image / 1000
    h, w = depth_image.shape
    u0, v0 = intrinsics[0, 2], intrinsics[1, 2]
    fu, fv = intrinsics[0, 0], intrinsics[1, 1]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_flat = u.flatten() - u0
    v_flat = v.flatten() - v0
    z_flat = depth_image.flatten()
    u_flat = u_flat[z_flat > 0]
    v_flat = v_flat[z_flat > 0]
    z_flat = z_flat[z_flat > 0]
    unporj_u = u_flat * z_flat / fu
    unporj_v = v_flat * z_flat / fv
    return np.vstack((unporj_u, unporj_v, z_flat)).T 

def write_ply(ply_path, points, colors= np.array([])):
        """Write mesh, point cloud, or oriented point cloud to ply file.
        Args:
            ply_path (str): Output ply path.
            points (float): Nx3 x,y,z locations for each point
            colors (uchar): Nx3 r,g,b color for each point
        We provide this function for you.
        """
        with open(ply_path, 'w') as f:
            # Write header.
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(len(points)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            if len(colors) != 0:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            f.write('end_header\n')
            # Write points.
            for i in range(len(points)):
                f.write('{0} {1} {2}'.format(
                    points[i][0],
                    points[i][1],
                    points[i][2]))
                if len(colors) != 0:
                    f.write(' {0} {1} {2}'.format(
                        int(colors[i][0]),
                        int(colors[i][1]),
                        int(colors[i][2])))

def main():
    parser = argparse.ArgumentParser(description="Align a mesh to a point cloud.")
    parser.add_argument("-m", "--mesh_file", default="./red_cube/red_cube/Shaded/base.obj",  help="Provide mesh file.")    
    parser.add_argument("--depth_image", default="./512_depth.npy",  help="Provide depth image.")
    parser.add_argument("--mask", default="./1_msk.png",  help="Provide mask file.")
    
    args = parser.parse_args()
    mesh_file = args.mesh_file
    depth_image = args.depth_image
    mask_file = args.mask

    mesh = trimesh.load_mesh(mesh_file)
    # read depth image
    # depth_image = Image.open(depth_image)
    depth_image = np.load(depth_image)
    # read mask
    mask = Image.open(mask_file)
    mask = np.array(mask)

    source_depth = depth_image
    depth = get_depth(depth_image, mask)
    # depth = depth_image

    img_size = 512
    fxfy = float(img_size)

    intrinsic_np = np.array([
        [fxfy, 0., img_size/2],
        [0. ,fxfy, img_size/2],
        [0., 0., 1.0]
    ])
    # get point cloud
    obj_xyz = depth_to_pcl(depth, intrinsic_np)
    bgd_xyz = depth_to_pcl(source_depth, intrinsic_np)

    # print("PCD: ", pcd.shape)
    obj_np = np.array(obj_xyz)
    obj_pcd = numpy_2_pcd( obj_np , np.array([1.,0.,0.]))
    cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044 , -0.63127772, 0.64917582, -0.31329509, 0.28619116])
    obj_pcd.transform( cam_extrinsic )

    bgd_np = np.array(bgd_xyz)
    # bgd_np[:,2] = -0.05
    bgd_pcd = numpy_2_pcd( bgd_np )
    bgd_pcd.transform( cam_extrinsic )
    # visualize_pcd(obj_pcd)

    align_meshes_pcd(mesh, obj_pcd, bgd_pcd)
    # mesh_points = mesh2points(aligned_mesh)
    # write_ply("aligned_mesh.ply", mesh_points)





if __name__ == "__main__":
    main()
