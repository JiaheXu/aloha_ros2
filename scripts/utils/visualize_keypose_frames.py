import os
import pickle
import glob

import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['figure.dpi'] = 128
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import blosc
import torch
import torch.nn.functional as F
import PIL.Image as Image
# import moviepy.video.io.ImageSequenceClip
import cv2

import utils.pytorch3d_transforms as pytorch3d_transforms



def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def normalize_vector(v, return_mag=False):
    device = v.device
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out  # batch*3


############# Utility functions and Layers #############
# Offset from the gripper center to three gripper points before any action
GRIPPER_DELTAS = torch.tensor([
    #[0, 0, 0,],
    #[0, -0.04, 0.00514],
    #[0, 0.04, 0.00514],
    [0, 0, 0,],
    [0, -0.01, 0.00125],
    [0, 0.01, 0.00125],
])
GRIPPER_DELTAS_FOR_VIS = torch.tensor([
    [0, 0, 0,],
    [0, -0.06, 0.03],
    [0, 0.06, 0.03],
])

# Visualize the gripper as polygons
def build_rectangle_points(center, axis_h, axis_w, axis_d, h, w, d):
    def _helper(cur_points, axis, size):
        points = []
        for p in cur_points:
            points.append(p + axis * size / 2)
        for p in cur_points:
            points.append(p - axis * size / 2)
        return points

    points = _helper([center], axis_h, h)
    points = _helper(points, axis_w, w)
    points = _helper(points, axis_d, d)

    return points


def make_polygons(points):
    """Make polygons from 8 side points of a rectangle
    """
    def _helper(four_points):
        center = four_points.mean(axis=0, keepdims=True)
        five_points = np.concatenate([four_points, center], axis=0)
        return [five_points[[0,1,-1]],
                five_points[[0,2,-1]],
                five_points[[0,3,-1]],
                five_points[[1,2,-1]],
                five_points[[1,3,-1]],
                five_points[[2,3,-1]]]

    polygons = (
        _helper(points[:4])
        + _helper(points[-4:])
        + _helper(points[[0,1,4,5]])
        + _helper(points[[2,3,6,7]])
        + _helper(points[[0,2,4,6]])
        + _helper(points[[1,3,5,7]])
    )
    return polygons


def compute_rectangle_polygons(points):
    p1, p2, p3 = points.chunk(3, 0)

    line12 = p2 - p1
    line13 = p3 - p1

    axis_d = F.normalize(cross_product(line12, line13))
    axis_w = F.normalize(p3 - p2)
    axis_h = F.normalize(cross_product(axis_d, axis_w))
    
    length23 = torch.norm(p3 - p2, dim=-1)
    length13 = (line13 * axis_h).sum(-1).abs()
    rectangle1 = build_rectangle_points(p1, axis_d, axis_w, axis_h,
                                        0.03, length23, length13 / 2)
    rectangle2 = build_rectangle_points(p2, axis_d, axis_w, axis_h,
                                        0.03, length23 / 4, length13 * 2)
    rectangle3 = build_rectangle_points(p3, axis_d, axis_w, axis_h,
                                        0.03, length23 / 4 , length13 * 2)

    rectangle1 = torch.cat(rectangle1, dim=0).data.cpu().numpy()
    rectangle2 = torch.cat(rectangle2, dim=0).data.cpu().numpy()
    rectangle3 = torch.cat(rectangle3, dim=0).data.cpu().numpy()

    polygon1 = make_polygons(rectangle1)
    polygon2 = make_polygons(rectangle2)
    polygon3 = make_polygons(rectangle3)

    return polygon1, polygon2, polygon3


def get_gripper_matrix_from_action(action: torch.Tensor,
                                   rotation_param="quat_from_query"):
    """Converts an action to a transformation matrix.

    Args:
        action: A N-D tensor of shape (batch_size, ..., 8) if rotation is
                parameterized as quaternion.  Otherwise, we assume to have
                a 9D rotation vector (3x3 flattened).

    """
    dtype = action.dtype
    device = action.device

    position = action[..., :3]

    if "quat" in rotation_param:
        quaternion = action[..., 3:7]
        rotation = pytorch3d_transforms.quaternion_to_matrix(quaternion)
    else:
        rotation = compute_rotation_matrix_from_ortho6d(action[..., 3:9])

    shape = list(action.shape[:-1]) + [4, 4]
    gripper_matrix = torch.zeros(shape, dtype=dtype, device=device)
    gripper_matrix[..., :3, :3] = rotation
    gripper_matrix[..., :3, 3] = position
    gripper_matrix[..., 3, 3] = 1

    return gripper_matrix


def get_three_points_from_curr_action(gripper: torch.Tensor,
                                      rotation_param="quat_from_query",
                                      for_vis=False):
    gripper_matrices = get_gripper_matrix_from_action(gripper, rotation_param)
    bs = gripper.shape[0]
    if for_vis:
        pcd = GRIPPER_DELTAS_FOR_VIS.unsqueeze(0).repeat(bs, 1, 1).to(gripper.device)
    else:
        pcd = GRIPPER_DELTAS.unsqueeze(0).repeat(bs, 1, 1).to(gripper.device)

    pcd = torch.cat([pcd, torch.ones_like(pcd[..., :1])], dim=-1)
    pcd = pcd.permute(0, 2, 1)

    pcd = (gripper_matrices @ pcd).permute(0, 2, 1)
    pcd = pcd[..., :3]

    return pcd


def inverse_transform_pcd_with_action(pcd: torch.Tensor, action: torch.Tensor,
                                      rotation_param: str = "quat_from_query"):
    mat = get_gripper_matrix_from_action(action, rotation_param).inverse()

    pcd = torch.cat([pcd, torch.ones_like(pcd[..., :1])], dim=-1)
    pcd = pcd.permute(0, 2, 1)

    pcd = (mat @ pcd).permute(0, 2, 1)
    pcd = pcd[..., :3]

    return pcd


############# Visualization utility functions #############
def visualize_actions_and_point_clouds(visible_pcd, visible_rgb,
                                       gripper_pose_trajs, legends=[], markers=[],
                                       save=True, rotation_param="quat_from_query",
                                       rand_inds=None):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gripper_pose_trajs: A list of tensors of shape (B, 8)
        legends: A list of strings indicating the legend for each trajectory
    """
    gripper_pose_trajs = [t.data.cpu() for t in gripper_pose_trajs]

    cur_vis_pcd = visible_pcd[0].permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    cur_vis_rgb = visible_rgb[0].permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    if rand_inds is None:
        rand_inds = torch.randperm(cur_vis_pcd.shape[0]).data.cpu().numpy()[:50000]
        mask = (
            # (cur_vis_pcd[rand_inds, 2] >= 0.2) &
            # (cur_vis_pcd[rand_inds, 1] <= 2) &
            # (cur_vis_pcd[rand_inds, 0] <= 1.5)
            (cur_vis_pcd[rand_inds, 2] >= -0.2)
        )
        rand_inds = rand_inds[mask]
    fig = plt.figure()
    canvas = fig.canvas
    # ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(cur_vis_pcd[rand_inds, 0],
               cur_vis_pcd[rand_inds, 1],
               cur_vis_pcd[rand_inds, 2],
               c=cur_vis_rgb[rand_inds], s=1)

    # predicted gripper pose
    cont_range_inds = np.linspace(0, 1, len(gripper_pose_trajs)).astype(np.float32)
    cm = plt.get_cmap('brg')
    colors = cm(cont_range_inds)
    legends = (legends if len(legends) == len(gripper_pose_trajs)
               else [""] * len(gripper_pose_trajs))
    markers = (markers if len(markers) == len(gripper_pose_trajs)
                else ["*"] * len(gripper_pose_trajs))
    for gripper_pose, color, legend, marker in (
        zip(gripper_pose_trajs, colors, legends, markers)
        ):
        gripper_pcd = get_three_points_from_curr_action(
            gripper_pose, rotation_param=rotation_param, for_vis=True
        )
        ax.plot(gripper_pcd[0, [1, 0, 2], 0],
                gripper_pcd[0, [1, 0, 2], 1],
                gripper_pcd[0, [1, 0, 2], 2],
                c=color,
                markersize=1, marker=marker,
                linestyle='--', linewidth=1,
                label=legend)
        polygons = compute_rectangle_polygons(gripper_pcd[0])
        for poly_ind, polygon in enumerate(polygons):
            polygon = Poly3DCollection(polygon, facecolors=color)
            alpha = 0.5 if poly_ind == 0 else 1.3
            polygon.set_edgecolor([min(c * alpha, 1.0) for c in color])
            ax.add_collection3d(polygon)

    fig.tight_layout()
    ax.legend(loc="lower center", ncol=len(gripper_pose_trajs))
    # ax.set_xlim(-0.75, 0.75)
    # ax.set_ylim(-0.75, 0.75)
    # ax.set_zlim(0.5, 1.5)
    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                          [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110] # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1)
    ], axis=0)
    if save:
        Image.fromarray(images, mode='RGB').save('diff_traj.png')

    plt.close()

    return images, rand_inds


def visualize_actions_and_point_clouds_video(visible_pcd, visible_rgb,
                                             gt_pose_l, gt_pose_r, curr_pose_l, curr_pose_r,
                                             save=True, rotation_param="quat_from_query"):
    """Visualize by plotting the point clouds and gripper pose as video.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gt_pose: A tensor of shape (B, 8)
        curr_pose: A tensor of shape (B, 8)
    """
    images, rand_inds = [], None
    for i in range(visible_pcd.shape[0]):
        # `visualize_actions_and_point_clouds` only visualize the first
        # point cloud and gripper in the batch.
        # To overlap two scenes in the same visualization, you can
        # 1) concatenate one scene to another at the `width` dimension.
        #    visible_pcd, visible_rgb becomes (B, ncam, 3, H, W * 2)
        # 2) for the gt_pose and curr_pose argument in the function,
        #    you can set the former to the gt_action of the first scene
        #    and the later to the gt_action of the second scene
        image, rand_inds = visualize_actions_and_point_clouds(
            visible_pcd[i:], visible_rgb[i:],
            [gt_pose_l[i:], gt_pose_r[i:], curr_pose_l[i:], curr_pose_r[i:]], 
            ["gt-l", "gt-r", "curr-l", "curr-r"], # add legened label to the imagined gripper
            ["d", "d", "o", "o"], # some dummy matplotlib marker
            save=False,
            rotation_param=rotation_param,
            rand_inds=rand_inds,
        )
        # add denoising progress bar
        images.append(image)
    pil_images = []
    for img in images:
        pil_images.extend([Image.fromarray(img)] * 2)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=1)
    clip.write_videofile("keypose_frames.mp4")


def visualize_gt_trajectories_and_points_clouds(
        visible_pcd, visible_rgb,
        history_trajectories, target_trajectories,
        save=True, save_name="trajs.gif", rotation_param="quat_from_query"
    ):
    """Visualize by plotting the point clouds and gripper pose as video.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        history_trajectories: A tensor of shape (B, nhist, 8)
        target_trajectories: A tensor of shape (B, traj_len, 8)
    """
    images, rand_inds = [], None

    nhist = history_trajectories.shape[1]
    trajs = [m.squeeze(1) for m in history_trajectories.chunk(nhist, 1)]
    trajs.append(None)
    traj_names = [f"hist{i}" for i in range(nhist)]
    traj_names.append(None)
    traj_markers = ["o"] * nhist + ["*"]

    for i in range(target_trajectories.shape[1]):
        trajs[-1] = target_trajectories[:, i]
        traj_names[-1] = "target"
        image, rand_inds = visualize_actions_and_point_clouds(
            visible_pcd, visible_rgb,
            trajs,
            traj_names,
            traj_markers,
            save=False,
            rotation_param=rotation_param,
            rand_inds=rand_inds,
        )
        images.append(image)

    if save:
        pil_images = []
        for img in images:
            pil_images.extend([Image.fromarray(img)] * 2)
        pil_images[0].save(save_name, save_all=True,
                           append_images=pil_images[1:], duration=1, loop=0)


def visualize_denoising_diffusion(visible_pcd, visible_rgb,
                                  gt_pose, noisy_poses, pred_poses,
                                  save=True, rotation_param="quat_from_query",
                                  save_name="diff_trajs.gif"):
    """Visualize by plotting the point clouds and gripper pose as video.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gt_pose: A tensor of shape (B, 8)
        noisy_poses: A list of tensors of shape (B, 8)
        pred_poses: A list of tensors of shape (B, 8)
    """
    images, rand_inds = [], None
    for i, (noisy_pose, pred_pose) in enumerate(zip(noisy_poses, pred_poses)):
        image, rand_inds = visualize_actions_and_point_clouds(
            visible_pcd, visible_rgb,
            [gt_pose, noisy_pose, pred_pose],
            ["gt", "noisy", "pred"],
            ["d", "o", "*"],
            save=False,
            rotation_param=rotation_param,
            rand_inds=rand_inds,
        )
        # add denoising progress bar
        progress_bar = np.zeros((32, image.shape[1], 3), dtype=np.uint8)
        progress = int(image.shape[1] // len(noisy_poses) * i)
        progress_bar[:, :progress] = 255
        progress_bar = progress_bar[:, ::-1]
        image = np.concatenate([progress_bar, image], axis=0)
        images.append(image)
    if save:
        pil_images = []
        for img in images:
            pil_images.extend([Image.fromarray(img)] * 2)
        pil_images[0].save(save_name, save_all=True,
                           append_images=pil_images[1:], duration=1, loop=0)
    video = np.stack(images, axis=0)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
    return video


def visualize_keyposes_and_point_clouds(visible_pcd, visible_rgb,
                                        gripper_pose_trajs,
                                        keypose_gripper_pose_trajs,
                                        save=True,
                                        rotation_param="quat_from_query",
                                        rand_inds=None):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: A tensor of shape (B, ncam, 3, H, W)
        visible_rgb: A tensor of shape (B, ncam, 3, H, W)
        gripper_pose_trajs: A list of tensors of shape (B, 8)
        legends: A list of strings indicating the legend for each trajectory
    """
    gripper_pose_trajs = [t.data.cpu() for t in gripper_pose_trajs]
    keypose_gripper_pose_trajs = [t.data.cpu() for t in keypose_gripper_pose_trajs]

    cur_vis_pcd = visible_pcd[0].permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    cur_vis_rgb = visible_rgb[0].permute(0, 2, 3, 1).flatten(0, -2).data.cpu().numpy()
    if rand_inds is None:
        rand_inds = torch.randperm(cur_vis_pcd.shape[0]).data.cpu().numpy()[:50000]
        mask = (
            # (cur_vis_pcd[rand_inds, 2] >= 0.2) &
            # (cur_vis_pcd[rand_inds, 1] <= 2) &
            # (cur_vis_pcd[rand_inds, 0] <= 1.5)
            (cur_vis_pcd[rand_inds, 2] >= -0.2)
        )
        rand_inds = rand_inds[mask]
    fig = plt.figure()
    canvas = fig.canvas
    # ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(cur_vis_pcd[rand_inds, 0],
               cur_vis_pcd[rand_inds, 1],
               cur_vis_pcd[rand_inds, 2],
               c=cur_vis_rgb[rand_inds], s=1)

    # predicted gripper pose
    colors = [
        np.array([1., 0, 0], dtype=np.float32),
        np.array([0, 0, 1.], dtype=np.float32)
    ]
    for pose_trajs, color in zip(
        [keypose_gripper_pose_trajs, gripper_pose_trajs],
        colors
    ):
        for gripper_pose in pose_trajs:
            gripper_pcd = get_three_points_from_curr_action(
                gripper_pose, rotation_param=rotation_param, for_vis=False
            )
            polygons = compute_rectangle_polygons(gripper_pcd[0])
            for poly_ind, polygon in enumerate(polygons):
                polygon = Poly3DCollection(polygon, facecolors=color)
                alpha = 0.5 if poly_ind == 0 else 1.3
                polygon.set_edgecolor([min(c * alpha, 1.0) for c in color])
                ax.add_collection3d(polygon)

    fig.tight_layout()
    ax.legend(loc="lower center", ncol=len(gripper_pose_trajs))
    # ax.set_xlim(-0.75, 0.75)
    # ax.set_ylim(-0.75, 0.75)
    # ax.set_zlim(0.5, 1.5)
    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                          [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
        ax.view_init(elev=elev, azim=azim)
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110] # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1)
    ], axis=0)
    if save:
        Image.fromarray(images, mode='RGB').save('diff_traj.png')

    plt.close()

    return images, rand_inds

