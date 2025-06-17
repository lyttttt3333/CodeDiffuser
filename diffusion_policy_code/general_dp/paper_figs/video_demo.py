import time
import os

import dill
import hydra
import torch
from omegaconf import open_dict
import diffusers
import cv2
import matplotlib
from matplotlib import cm
import numpy as np
import pickle
import transforms3d
import open3d as o3d
from pytorch3d.transforms import rotation_6d_to_matrix
import matplotlib.pyplot as plt

from diffusion_policy.dataset.real_dataset import RealDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.data_utils import load_dict_from_hdf5, d3fields_proc, d3fields_proc_for_vis
from diffusion_policy.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import np2o3d, aggr_point_cloud_from_data, o3dVisualizer, draw_keypoints, poseInterp, viewCtrlInterp
from d3fields.utils.grounded_sam import grounded_instance_sam_new_ver
from d3fields.fusion import Fusion

def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def extract_pcd_from_data_dict(data_dict, t, fusion, shape_meta, kin_helper):
    # vis raw pcd
    view_keys = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
    color_seq = np.stack([data_dict['observations']['images'][f'{k}_color'][t] for k in view_keys], axis=0) # (V, H ,W, C)
    # color_seq = np.stack([white_balance_loops(color_seq[i]) for i in range(color_seq.shape[0])], axis=0)
    depth_seq = np.stack([data_dict['observations']['images'][f'{k}_depth'][t] for k in view_keys], axis=0) / 1000. # (V, H ,W)
    extri_seq = np.stack([data_dict['observations']['images'][f'{k}_extrinsics'][t] for k in view_keys], axis=0) # (V, 4, 4)
    intri_seq = np.stack([data_dict['observations']['images'][f'{k}_intrinsics'][t] for k in view_keys], axis=0) # (V, 3, 3)
    qpos_seq = data_dict['observations']['full_joint_pos'][t] # (8*num_bots)
    
    if 'd3fields' in shape_meta['obs']:
        boundaries = shape_meta['obs']['d3fields']['info']['boundaries']
        d3fields_shape_meta = shape_meta['obs']['d3fields']
    else:
        boundaries = {
            'x_lower': -0.3,
            'x_upper': 0.1,
            'y_lower': -0.28,
            'y_upper': 0.05,
            'z_lower': 0.01,
            'z_upper': 0.2,
        }
        d3fields_shape_meta = {
            'shape': [5, 1000],
            'type': 'spatial',
            'info': {
                'feat_type': 'no_feats',
                'reference_frame': 'robot',
                'use_seg': False,
                'use_dino': False,
                'distill_dino': False,
                'distill_obj': 'knife',
                'rob_pcd': True,
                'view_keys': ['camera_0', 'camera_1', 'camera_2', 'camera_3'],
                'query_texts': ['cup', 'pad'],
                'query_thresholds': [0.3],
                'N_per_inst': 100,
                'boundaries': boundaries,
                'resize_ratio': 0.5,
            }
        }
    
    # compute robot pcd
    num_bots = qpos_seq.shape[0] // 8
    curr_qpos = qpos_seq
    qpos_dim = curr_qpos.shape[0] // num_bots
    
    # compute robot pcd
    robot_pcd_ls = []
    for rob_i in range(num_bots):
        robot_pcd = kin_helper.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], num_pts=[1000 for _ in range(len(kin_helper.meshes.keys()))], pcd_name=f'robot_pcd_{rob_i}')
        robot_base_pose_in_world = data_dict['observations']['robot_base_pose_in_world'][t, rob_i]
        robot_pcd = (robot_base_pose_in_world @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
        robot_pcd_ls.append(robot_pcd)
    robot_pcd = np.concatenate(robot_pcd_ls, axis=0)
    
    pcd, pcd_colors = aggr_point_cloud_from_data(color_seq, depth_seq, intri_seq, extri_seq, downsample=False, boundaries=boundaries, out_o3d=False, excluded_pts=robot_pcd, exclude_threshold=0.02) # (N, 3), (N, 3)
    robot_base_pose_in_world = data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
    pcd = np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).transpose() # (4, N)
    pcd = pcd[:3].transpose() # (N, 3)
    
    # vis d3fields
    aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feat_ls, rob_mesh_ls = d3fields_proc_for_vis(
        fusion=fusion,
        shape_meta=d3fields_shape_meta,
        color_seq=color_seq[None],
        depth_seq=depth_seq[None],
        extri_seq=extri_seq[None],
        intri_seq=intri_seq[None],
        robot_base_pose_in_world_seq=data_dict['observations']['robot_base_pose_in_world'][t:t+1],
        teleop_robot=kin_helper,
        qpos_seq=qpos_seq[None],
        exclude_threshold=0.02,
        return_raw_feats=True,
    )
    
    if len(aggr_feats_ls) == 0:
        return pcd, pcd_colors, aggr_src_pts_ls[0], None, None, rob_mesh_ls[0]
    else:
        return pcd, pcd_colors, aggr_src_pts_ls[0], aggr_feats_ls[0], aggr_raw_feat_ls[0], rob_mesh_ls[0]

ckpt_name = "our"
data_root = "/media/yixuan_2T/diffusion_policy"

# # knife_1
# ckpt_map_dict = {
#     'our': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_distill_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt',
#     'raw_pcd': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_no_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt',
#     'dp': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_original/checkpoints/latest.ckpt',
#     'dp_rgbd': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.25_knife_Guang_knife_v4_original_rgbd/checkpoints/latest.ckpt',
# }
# dataset_dir = f"{data_root}/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_distill_dino_N_1000_pn2.1_r_0.04/video_demo"
# epi_path = f"{dataset_dir}/episode_0.hdf5"

# # knife_2
# ckpt_map_dict = {
#     'our': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_distill_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt',
#     'raw_pcd': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_no_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt',
#     'dp': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_original/checkpoints/latest.ckpt',
#     'dp_rgbd': '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.25_knife_Guang_knife_v4_original_rgbd/checkpoints/latest.ckpt',
# }
# dataset_dir = f"{data_root}/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_distill_dino_N_1000_pn2.1_r_0.04/eval_more_hhh/dir_0"
# epi_path = f"{dataset_dir}/episode_0.hdf5"

# # knife task
# if ckpt_name == "our":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_distill_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/knife_our_2"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"
# elif ckpt_name == "raw_pcd":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_no_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/knife_raw_pcd"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"
# elif ckpt_name == "dp":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_original/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/knife_dp"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"
# elif ckpt_name == "dp_rgbd":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.25_knife_Guang_knife_v4_original_rgbd/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/knife_dp_rgbd"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"

# open pen
ckpt_map_dict = {
    'our': "/media/yixuan_2T/diffusion_policy/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_no_seg_distill_dino_N_4000_pn2.1_r_0.04/checkpoints/latest.ckpt",
    'raw_pcd': "/media/yixuan_2T/diffusion_policy/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_no_seg_no_dino_N_4000_pn2.1_r_0.04/checkpoints/latest.ckpt",
    'dp': "/media/yixuan_2T/diffusion_policy/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_original/checkpoints/latest.ckpt",
}
# dataset_dir = f"{data_root}/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_no_seg_distill_dino_N_4000_pn2.1_r_0.04/video_demo_1"
# epi_path = f"{dataset_dir}/episode_2.hdf5"
dataset_dir = f"{data_root}/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_no_seg_distill_dino_N_4000_pn2.1_r_0.04/video_demo_2"
epi_path = f"{dataset_dir}/episode_3.hdf5"

# # toothbrush
# ckpt_map_dict = {
#     'our': "/media/yixuan_2T/diffusion_policy/data/outputs/toothbrush/2024.01.31_toothbrush_Guang_toothbrush_no_seg_distill_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt",
# }
# dataset_dir = f"{data_root}/data/outputs/toothbrush/2024.01.31_toothbrush_Guang_toothbrush_no_seg_distill_dino_N_1000_pn2.1_r_0.04/video_demo_1"
# epi_path = f"{dataset_dir}/episode_8.hdf5"
# # dataset_dir = f"{data_root}/data/outputs/toothbrush/2024.01.31_toothbrush_Guang_toothbrush_no_seg_distill_dino_N_1000_pn2.1_r_0.04/video_demo_2"
# # epi_path = f"{dataset_dir}/episode_9.hdf5"

# # place can
# ckpt_map_dict = {
#     'our': "/media/yixuan_2T/diffusion_policy/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_no_seg_distill_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt",
#     'raw_pcd': "/media/yixuan_2T/diffusion_policy/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_no_seg_no_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt",
#     'dp': "/media/yixuan_2T/diffusion_policy/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_original/checkpoints/latest.ckpt",
# }
# # dataset_dir = f"{data_root}/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_no_seg_distill_dino_N_1000_pn2.1_r_0.04/extra"
# dataset_dir = f"{data_root}/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_no_seg_distill_dino_N_1000_pn2.1_r_0.04/extra_2"
# epi_path = f"{dataset_dir}/episode_8.hdf5"

ckpt_path = ckpt_map_dict[ckpt_name]
out_dir = f"{dataset_dir}/teaser_vis"
d3fields_root = '/home/yixuan/bdai/general_dp/d3fields_dev/d3fields'

# load checkpoint
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# hacks for method-specific setup.
action_offset = 0
delta_action = False
if 'diffusion' in cfg.name:
    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # set inference params
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    policy.num_inference_steps = 16 # DDIM inference iterations
    noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )
    # if 'd3fields' in cfg.task.dataset.shape_meta['obs']:
    #     cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.02 # for knife only
    policy.noise_scheduler = noise_scheduler

else:
    raise RuntimeError("Unsupported policy type: ", cfg.name)

cfg.data_root = data_root
cfg.task.dataset.dataset_dir = dataset_dir
cfg.task.dataset.use_cache = True
with open_dict(cfg.task.dataset):
    cfg.task.dataset.vis_input = False
dataset : RealDataset = hydra.utils.instantiate(cfg.task.dataset)
raw_data_dict, _ = load_dict_from_hdf5(epi_path)

# # knife_1
# view_ctrl_info = {
#     "front" : [ 0.40999574872898714, 0.53468751019305216, 0.73892675717401934 ],
#     "lookat" : [ 0.39242082138073253, -0.047167632701400203, 0.028114805637039261 ],
#     "up" : [ -0.25633940739364364, -0.70994573217651635, 0.65594753264375183 ],
#     "zoom" : 0.41999999999999971
# }

# # knife_2
# view_ctrl_info = {
#     "front" : [ 0.73954575634948339, 0.64721601198369272, 0.18488782571439152 ],
#     "lookat" : [ 0.35476861825316802, -0.040310761893000618, 0.04917861507634113 ],
#     "up" : [ -0.19042193752496933, -0.062287343840040026, 0.9797243349568231 ],
#     "zoom" : 0.47999999999999976
# }


# view_ctrl_info = {
#     "front" : [ 0.64874319238291767, 0.4685328123874628, 0.59967430664746291 ],
#     "lookat" : [ 0.46809503105643996, -0.032377395771860003, -0.048422308702472994 ],
#     "up" : [ -0.4883648694840762, -0.34802049374837091, 0.80023839584522649 ],
#     "zoom" : 0.37999999999999967
# }

view_ctrl_info = {
    "front" : [ 0.28617469058768846, 0.69480134183290854, 0.65981447532941451 ],
    "lookat" : [ 0.48442274810915614, -0.03465853350964225, -0.03939080594022408 ],
    "up" : [ -0.21253386675545208, -0.62543190014704308, 0.75077579460211341 ],
    "zoom" : 0.39999999999999969
}

o3d_vis = o3dVisualizer(view_ctrl_info=view_ctrl_info, save_path=f'{out_dir}')
o3d_vis.start()

fusion = Fusion(num_cam=4, dtype=torch.float16)
kin_helper = KinHelper(robot_name='trossen_vx300s_v3')

is_first = True
# rotate_steps = [24] # knife_1
# vis_action_steps = [24, 31, 40, 50, 56, 62, 70, 95, 104, 115, 123, 130, 141, 149]
# rotate_steps = [-1] # knife_1
# rotate_steps = [21] # knife_1 dir_0
# rotate_steps = [13] # knife_1 dir_1
# rotate_steps = [11] # place can
# rotate_steps = [20] # place can
# rotate_steps = [101] # toothbrush demo 1
# vis_action_steps = [101] # toothbrush demo 1
# rotate_steps = [129] # toothbrush demo 2
# vis_action_steps = [129] # toothbrush demo 2
# rotate_steps = [24] # open pen demo 1
# vis_action_steps = [24] # open pen demo 1
rotate_steps = [51] # open pen demo 2
vis_action_steps = [51] # open pen demo 2
save_data_for_rotate_vis = {}

precomp_mesh_seq = []
num_bots = raw_data_dict['observations']['full_joint_pos'][0].shape[0] // 8
# for i in range(20, 50):
# for i in range(10, 25):
for i in range(40, 60):
    qpos_dim = 8
    curr_qpos = raw_data_dict['observations']['full_joint_pos'][i] # (8*num_bots)
    next_qpos = raw_data_dict['observations']['full_joint_pos'][i+1] # (8*num_bots)
    if np.linalg.norm(curr_qpos - next_qpos) < 1e-3:
        continue
    for interp_i in range(10):
        robot_meshes_ls = []
        inter_qpos = curr_qpos * (1 - interp_i/9) + next_qpos * (interp_i/9)
        for rob_i in range(num_bots):
            # compute robot pcd
            robot_meshes = kin_helper.gen_robot_meshes(inter_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], link_names=['vx300s/left_finger_link', 'vx300s/right_finger_link', 'vx300s/gripper_bar_link', 'vx300s/gripper_prop_link', 'vx300s/gripper_link'])
        
            # transform robot pcd to world frame    
            robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][i, rob_i]
            # transform mesh to the frame of first robot
            for mesh in robot_meshes:
                first_robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][i, 0] # (4, 4)
                mesh.transform(np.linalg.inv(first_robot_base_pose_in_world) @ robot_base_pose_in_world)
            robot_meshes_ls = robot_meshes_ls + robot_meshes
        precomp_mesh_seq.append(robot_meshes_ls)

# for i in range(20, 50):
# for i in range(10, 25):
for i in range(40, 60):
    start_time = time.time()
    data = dataset[i]
    obs_dict = data['obs']
    gt_action = data['action'][None].to(device)
    for key in obs_dict.keys():
        obs_dict[key] = obs_dict[key].unsqueeze(0).to(device)
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        pred_action = result['action_pred'] # (N, T, 10 * num_bots)
        num_bots = pred_action.shape[-1] // 10
        
        # use pred action to offset
        N, T = pred_action.shape[:2]
        pred_action = pred_action.reshape(N*T, num_bots, 10)
        pred_action_rot = pred_action[:, :, 3:9] # (N, T, 6)
        pred_action_mat = rotation_6d_to_matrix(pred_action_rot).cpu().numpy() # (N, T, 3, 3)
        theta = -0.2
        ee_rot_adj = np.array([[np.cos(theta), 0, -np.sin(theta)],
                               [0, 1, 0],
                               [np.sin(theta), 0, np.cos(theta)]]) # (3, 3)
        ee_rot_adj = np.tile(ee_rot_adj[None, None], (pred_action_mat.shape[0], pred_action_mat.shape[1], 1, 1)) # (N, T, 3, 3)
        pred_action_mat = pred_action_mat @ ee_rot_adj # (N, T, 3, 3)
        ee_axis = pred_action_mat[:, :, :, 0] # (N, T, 3)
        
        # use eef pose to offset
        # ee_pos = raw_data_dict['observations']['ee_pos'][i]
        # ee_rot_mat = transforms3d.euler.euler2mat(ee_pos[3], ee_pos[4], ee_pos[5], 'sxyz')
        # theta = -0.2
        # ee_rot_adj = np.array([[np.cos(theta), 0, -np.sin(theta)],
        #                        [0, 1, 0],
        #                        [np.sin(theta), 0, np.cos(theta)]])
        # ee_rot_mat = ee_rot_mat @ ee_rot_adj
        # ee_axis = ee_rot_mat[:, 0][None, None]
        
        pred_action_pos = pred_action[:, :, :3].cpu().numpy() # (N, T, 3)
        offset_val = 0.10
        pred_action_pos = pred_action_pos + offset_val * ee_axis # (N, T, 3)
        pred_action_pos = pred_action_pos.reshape(N, T, num_bots*3) # (N, T, 3*num_bots)
        
    # visualize pcd
    start_time = time.time()
    vis_shape_meta = cfg.task.dataset.shape_meta.copy()
    if 'd3fields' in vis_shape_meta['obs']:
        vis_shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.01
        vis_shape_meta['obs']['d3fields']['info']['boundaries']['z_upper'] = 0.2
        vis_shape_meta['obs']['d3fields']['info']['boundaries']['x_lower'] = -0.3
        vis_shape_meta['obs']['d3fields']['info']['boundaries']['x_upper'] = 0.1
        vis_shape_meta['obs']['d3fields']['shape'][1] = 40000
    pcd, pcd_colors, feats_pcd, distilled_feats, raw_feats, rob_mesh = extract_pcd_from_data_dict(raw_data_dict, i, fusion, vis_shape_meta, kin_helper)
    
    # save data for rotate vis
    if i >= vis_action_steps[0] and i <= vis_action_steps[-1]:
        qpos_dim = 8
        curr_qpos = raw_data_dict['observations']['full_joint_pos'][i] # (8*num_bots)
        next_qpos = raw_data_dict['observations']['full_joint_pos'][i+1] # (8*num_bots)
        robot_mesh_seq = []
        for interp_i in range(10):
            robot_meshes_ls = []
            inter_qpos = curr_qpos * (1 - interp_i/9) + next_qpos * (interp_i/9)
            for rob_i in range(num_bots):
                # compute robot pcd
                robot_meshes = kin_helper.gen_robot_meshes(inter_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], link_names=['vx300s/left_finger_link', 'vx300s/right_finger_link', 'vx300s/gripper_bar_link', 'vx300s/gripper_prop_link', 'vx300s/gripper_link'])
            
                # transform robot pcd to world frame    
                robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][i, rob_i]
                # transform mesh to the frame of first robot
                for mesh in robot_meshes:
                    first_robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][i, 0] # (4, 4)
                    mesh.transform(np.linalg.inv(first_robot_base_pose_in_world) @ robot_base_pose_in_world)
                robot_meshes_ls = robot_meshes_ls + robot_meshes
            robot_mesh_seq.append(robot_meshes_ls)
        if 'rob_mesh_seq' not in save_data_for_rotate_vis:
            save_data_for_rotate_vis['rob_mesh_seq'] = robot_mesh_seq
        else:
            save_data_for_rotate_vis['rob_mesh_seq'] = save_data_for_rotate_vis['rob_mesh_seq'] + robot_mesh_seq
                
    raw_pcd = np2o3d(pcd, color=pcd_colors)
    o3d_vis.update_pcd(raw_pcd, mesh_name='pcd')
    
    for mesh_i, mesh in enumerate(rob_mesh):
        o3d_vis.update_custom_mesh(mesh, mesh_name=f'rob_mesh_{mesh_i}')
    
    if ckpt_name == 'our':
        # visualize d3fields
        cmap = matplotlib.colormaps.get_cmap('viridis')
        d3fields = obs_dict['d3fields'][0, -1].cpu().numpy().transpose() # (N, 3+D)
        pred_act_pos = pred_action[0, :, :3].cpu().numpy()
        d3fields_geo = d3fields[:, :3]
        d3fields_sim = d3fields[:, 3]
        d3fields_sim_color = cmap(d3fields_sim)[:,:3]
        input_d3fields_o3d = np2o3d(d3fields_geo, color=d3fields_sim_color)
        o3d_vis.update_pcd(input_d3fields_o3d, mesh_name='input_fields')
        
        d3fields_sim = distilled_feats[:, 0]
        vis_boundaries = cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries'].copy()
        vis_boundaries['z_lower'] = 0.022
        robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
        feats_pcd_in_rob = robot_base_pose_in_world @ np.concatenate([feats_pcd, np.ones((feats_pcd.shape[0], 1))], axis=1).transpose() # (4, N)
        feats_pcd_in_rob = feats_pcd_in_rob[:3].transpose() # (N, 3)
        d3fields_bound_mask = (feats_pcd_in_rob[:, 0] > vis_boundaries['x_lower']) & (feats_pcd_in_rob[:, 0] < vis_boundaries['x_upper']) \
                            & (feats_pcd_in_rob[:, 1] > vis_boundaries['y_lower']) & (feats_pcd_in_rob[:, 1] < vis_boundaries['y_upper']) \
                            & (feats_pcd_in_rob[:, 2] > vis_boundaries['z_lower']) & (feats_pcd_in_rob[:, 2] < vis_boundaries['z_upper'])
        d3fields_sim[~d3fields_bound_mask] = 0
        d3fields_sim_color = cmap(d3fields_sim)[:,:3]
        high_sim_mask = d3fields_sim > 0.6
        d3fields_o3d = np2o3d(feats_pcd, color=d3fields_sim_color)
        partial_d3fields_o3d = np2o3d(feats_pcd[high_sim_mask], color=d3fields_sim_color[high_sim_mask])
        o3d_vis.update_pcd(d3fields_o3d, mesh_name='d3fields')
        o3d_vis.update_pcd(partial_d3fields_o3d, mesh_name='partial_d3fields')
        
    # visualize action
    pred_act_pos = pred_action_pos[0] # (T, 3*num_bots)
    pred_act_pos = pred_act_pos.reshape(T * num_bots, 3) # (T*num_bots, 3)
    action_cmap = cm.get_cmap('plasma')
    act_len = pred_act_pos.shape[0] // num_bots
    pred_act_pos_color = action_cmap(np.arange(act_len)/(act_len-1))[:,:3]
    pred_act_pos_color = np.repeat(pred_act_pos_color, num_bots, axis=0)
    for act_i in range(act_len):
        for bot_i in range(num_bots):
            if is_first:
                o3d_vis.add_triangle_mesh('sphere', f'sphere_{act_i*num_bots+bot_i}', color=pred_act_pos_color[act_i*num_bots+bot_i], radius=0.005)
            tf = np.eye(4)
            tf[:3, 3] = pred_act_pos[act_i * num_bots + bot_i]
            o3d_vis.update_triangle_mesh(f'sphere_{act_i*num_bots+bot_i}', tf=tf)
    if is_first:
        is_first = False
    
    # render and save
    action_names_ls = [f'sphere_{act_i}' for act_i in range(act_len * num_bots)]
    rob_mesh_names_ls = [f'rob_mesh_{mesh_i}' for mesh_i in range(len(rob_mesh))]
    o3d_vis.render(['pcd'] + rob_mesh_names_ls, save_name=f'pcd_{i}')
    o3d_vis.render(['pcd'] + rob_mesh_names_ls + action_names_ls, save_name=f'pcd_{i}_with_action')
    o3d_vis.render(action_names_ls, save_name=f'action_{i}')
    o3d_vis.render(rob_mesh_names_ls, save_name=f'rob_mesh_{i}')
    if ckpt_name == 'our':
        d3fields_img = o3d_vis.render(['d3fields'], save_name=f'd3fields_{i}')
        partial_d3fields_img = o3d_vis.render(['partial_d3fields'], save_name=f'partial_d3fields_{i}')
        # blended_img = cv2.addWeighted(pcd_img, 0.6, partial_d3fields_img, 0.4, 0)
        # cv2.imwrite(f'{out_dir}/blended_{i}.png', blended_img)
        o3d_vis.render(['input_fields'], save_name=f'input_fields_{i}')
    
    # stop and render
    if i in vis_action_steps:
        rot_i = vis_action_steps.index(i)
        view_ctrl_info_0 = view_ctrl_info.copy()
        # # knife_1
        # view_ctrl_info_1 = {
        #     "front" : [ -0.43637831763928309, 0.58176606360051442, 0.68638342865855317 ],
        #     "lookat" : [ 0.39242082138073253, -0.047167632701400203, 0.028114805637039261 ],
        #     "up" : [ 0.2972050866364121, -0.62683183097483874, 0.72024370330754994 ],
        #     "zoom" : 0.41999999999999971
        # }
        
        # # knife_2
        # view_ctrl_info_1 = {
		# 	"front" : [ -0.34838707346757847, 0.91494496087080635, 0.20372080310516835 ],
		# 	"lookat" : [ 0.35476861825316802, -0.040310761893000618, 0.04917861507634113 ],
		# 	"up" : [ 0.022897012949680995, -0.20896517795766978, 0.97765499088333907 ],
		# 	"zoom" : 0.47999999999999976
        # }
        
        # # place can
        # view_ctrl_info_1 = {
		# 	"front" : [ 0.81386324100636387, -0.43769610921523516, 0.38216323857281509 ],
		# 	"lookat" : [ 0.35476861825316802, -0.040310761893000618, 0.04917861507634113 ],
		# 	"up" : [ -0.3187285147842609, 0.21363986820769934, 0.92345554336720292 ],
		# 	"zoom" : 0.47999999999999976
        # }
        
        # # toothbrush
        # view_ctrl_info_1 = {
        #     "front" : [ -0.40620100552729027, 0.63030258297252906, 0.66160365552706601 ],
		# 	"lookat" : [ 0.46809503105643996, -0.032377395771860003, -0.048422308702472994 ],
		# 	"up" : [ 0.29840083367597714, -0.59283777940656557, 0.74799753326449836 ],
		# 	"zoom" : 0.37999999999999967
		# }
        
        # open pen
        view_ctrl_info_1 = {
			"front" : [ -0.47674200844867959, 0.70540682479568229, 0.52451717694656386 ],
			"lookat" : [ 0.48442274810915614, -0.03465853350964225, -0.03939080594022408 ],
			"up" : [ 0.23785337526715902, -0.47091549554212903, 0.84950830951340972 ],
			"zoom" : 0.39999999999999969
		}
        
        if i in rotate_steps:
            inter_view_ctrls = viewCtrlInterp(view_ctrl_info_0, view_ctrl_info_1, 100, interp_type='sine')
            for pose_i, inter_view_ctrl in enumerate(inter_view_ctrls):
                # knife_1
                # o3d_vis.render(['pcd'] + rob_mesh_names_ls, save_name=f'rotate_pcd_{pose_i}_rot_i_{rot_i}', curr_view_ctrl_info=inter_view_ctrl)
                
                # knife_2
                o3d_vis.render(['pcd'] + rob_mesh_names_ls + action_names_ls, save_name=f'rotate_pcd_{pose_i}_rot_i_{rot_i}', curr_view_ctrl_info=inter_view_ctrl)
            os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/rotate_pcd_%d_rot_i_{rot_i}.png {o3d_vis.save_path}/rotate_pcd_rot_i_{rot_i}.mp4')
            
            
            # # knife_1
            # view_ctrl_info_2 = {
            #     "front" : [ -0.28650773155424097, 0.76678418947418858, 0.57441738007484999 ],
            # 	"lookat" : [ 0.40772432385412311, -0.02735813073867014, -0.0014968913523370397 ],
            # 	"up" : [ 0.17981659631168259, -0.5458608273985045, 0.81835319318904198 ],
            # 	"zoom" : 0.23999999999999957
            # }
            # knife_2
            view_ctrl_info_2 = view_ctrl_info_1.copy()
            zoom_inter_view_ctrls = viewCtrlInterp(view_ctrl_info_0, view_ctrl_info_2, 100, interp_type='sine')
            for pose_i, inter_view_ctrl in enumerate(zoom_inter_view_ctrls):
                # # knife_1
                # pcd_pose_i = o3d_vis.render(['pcd'] + rob_mesh_names_ls, save_name=f'zoom_pcd_{pose_i}', curr_view_ctrl_info=inter_view_ctrl)
                # partial_d3fields_pose_i = o3d_vis.render(['partial_d3fields'], save_name=f'zoom_partial_d3fields_{pose_i}', curr_view_ctrl_info=inter_view_ctrl)
                
                # # knife_2
                pcd_pose_i = o3d_vis.render(['pcd'] + rob_mesh_names_ls + action_names_ls, save_name=f'zoom_pcd_{pose_i}', curr_view_ctrl_info=inter_view_ctrl)
                partial_d3fields_pose_i = o3d_vis.render(['partial_d3fields'] + action_names_ls, save_name=f'zoom_partial_d3fields_{pose_i}', curr_view_ctrl_info=inter_view_ctrl)
                partial_d3fields_pose_i_gray = cv2.cvtColor(partial_d3fields_pose_i, cv2.COLOR_BGR2GRAY)
                overlay_mask = np.ones_like(partial_d3fields_pose_i_gray) * 0.5
                overlay_mask[partial_d3fields_pose_i_gray < 200] = 1
                blended_img = pcd_pose_i * (1 - overlay_mask[:, :, None]) + partial_d3fields_pose_i * overlay_mask[:, :, None]
                cv2.imwrite(f'{out_dir}/zoom_blended_{pose_i}_rot_i_{rot_i}.png', blended_img.astype(np.uint8))
            os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/zoom_blended_%d_rot_i_{rot_i}.png {o3d_vis.save_path}/zoom_blended_rot_i_{rot_i}.mp4')
        
        # visualize robot actions
        for act_i in range(8):
            curr_action_name_ls = [f'sphere_{s_i}' for s_i in range(num_bots * act_i)]
            for rep_i in range(10):
                o3d_vis.render(['pcd'] + curr_action_name_ls + rob_mesh_names_ls, save_name=f'extend_action_{act_i * 10 + rep_i}_rot_i_{rot_i}')
        os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/extend_action_%d_rot_i_{rot_i}.png {o3d_vis.save_path}/extend_action_rot_i_{rot_i}.mp4')
        
        # visualize robot mesh
        for seq_i, rob_mesh_ls in enumerate(precomp_mesh_seq):
            for mesh_i, mesh in enumerate(rob_mesh_ls):
                o3d_vis.update_custom_mesh(mesh, mesh_name=f'rob_mesh_{mesh_i}')
            o3d_vis.render(['pcd'] + rob_mesh_names_ls, save_name=f'smooth_rob_mesh_rot_i_{rot_i}_seq_{seq_i}')
        os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/smooth_rob_mesh_rot_i_{rot_i}_seq_%d.png {o3d_vis.save_path}/smooth_rob_mesh_rot_i_{rot_i}.mp4')
        
        # # visualize save data for rotate vis
        # if i != rotate_steps[0]:
        #     last_pcd = save_data_for_rotate_vis['pcd']
        #     last_pcd_colors = save_data_for_rotate_vis['pcd_colors']
        #     last_action = save_data_for_rotate_vis['pred_action_pos']
        #     last_rob_mesh = save_data_for_rotate_vis['rob_mesh_seq']
            
        #     # visualize action pred
        #     o3d_vis.update_pcd(np2o3d(last_pcd, color=last_pcd_colors), mesh_name='pcd')
        #     rob_mesh_ls = last_rob_mesh[0]
        #     for mesh_i, mesh in enumerate(rob_mesh_ls):
        #         o3d_vis.update_custom_mesh(mesh, mesh_name=f'rob_mesh_{mesh_i}')
        #     for act_i in range(act_len):
        #         for bot_i in range(num_bots):
        #             tf = np.eye(4)
        #             tf[:3, 3] = last_action[act_i * num_bots + bot_i]
        #             o3d_vis.update_triangle_mesh(f'sphere_{act_i*num_bots+bot_i}', tf=tf)
        #     for act_i in range(act_len):
        #         curr_action_name_ls = [f'sphere_{s_i}' for s_i in range(num_bots * act_i)]
        #         for rep_i in range(10):
        #             o3d_vis.render(['pcd'] + curr_action_name_ls + rob_mesh_names_ls, save_name=f'extend_action_{act_i * 10 + rep_i}_rot_i_{rot_i-1}')
        #     os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/extend_action_%d_rot_i_{rot_i-1}.png {o3d_vis.save_path}/extend_action_rot_i_{rot_i-1}.mp4')
            
        #     # visualize robot mesh
        #     for seq_i, rob_mesh_ls in enumerate(last_rob_mesh):
        #         for mesh_i, mesh in enumerate(rob_mesh_ls):
        #             o3d_vis.update_custom_mesh(mesh, mesh_name=f'rob_mesh_{mesh_i}')
        #         o3d_vis.render(['pcd'] + rob_mesh_names_ls, save_name=f'smooth_rob_mesh_rot_i_{rot_i-1}_seq_{seq_i}')
        #     os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/smooth_rob_mesh_rot_i_{rot_i-1}_seq_%d.png {o3d_vis.save_path}/smooth_rob_mesh_rot_i_{rot_i-1}.mp4')
        
        # save_data_for_rotate_vis['pcd'] = pcd
        # save_data_for_rotate_vis['pcd_colors'] = pcd_colors
        # save_data_for_rotate_vis['rob_mesh_seq'] = []
        # save_data_for_rotate_vis['pred_action_pos'] = pred_action_pos[0]
    
    # # visualize actions on 2D image
    # for view_idx in range(4):
    #     img = raw_data_dict['observations']['images'][f'camera_{view_idx}_color'][i]
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     extrinsic = raw_data_dict['observations']['images'][f'camera_{view_idx}_extrinsics'][i]
    #     intrinsic = raw_data_dict['observations']['images'][f'camera_{view_idx}_intrinsics'][i]
    #     robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
    #     pred_act_pos_in_world = robot_base_pose_in_world @ np.concatenate([pred_act_pos, np.ones((pred_act_pos.shape[0], 1))], axis=1).transpose() # (4, N)
    #     pred_act_pos_in_cam = extrinsic @ pred_act_pos_in_world # (4, N)
    #     pred_act_pos_in_cam = pred_act_pos_in_cam[:3].transpose() # (N, 3)
    #     pred_act_pos_in_pix = np.zeros((pred_act_pos_in_cam.shape[0], 2)) # (N, 2)
    #     pred_act_pos_in_pix[:, 0] = pred_act_pos_in_cam[:, 0] / pred_act_pos_in_cam[:, 2] * intrinsic[0, 0] + intrinsic[0, 2]
    #     pred_act_pos_in_pix[:, 1] = pred_act_pos_in_cam[:, 1] / pred_act_pos_in_cam[:, 2] * intrinsic[1, 1] + intrinsic[1, 2]
    #     pred_act_pos_color_cv2 = (pred_act_pos_color * 255).astype(np.uint8)[:,::-1]
    #     in_bound_mask = (pred_act_pos_in_pix[:, 0] >= 0) & (pred_act_pos_in_pix[:, 0] < img.shape[0]) \
    #                 & (pred_act_pos_in_pix[:, 1] >= 0) & (pred_act_pos_in_pix[:, 1] < img.shape[1])
    #     pred_act_pos_in_pix = pred_act_pos_in_pix[in_bound_mask].astype(np.int32)
    #     pred_act_pos_color_cv2 = pred_act_pos_color_cv2[in_bound_mask]
    #     img = draw_keypoints(img, pred_act_pos_in_pix, colors=pred_act_pos_color_cv2, radius=6)
    #     cv2.imwrite(f'{out_dir}/action_view_{view_idx}_{i}.png', img)
    
    # measure frequency
    end_time = time.time()
    print(f"freq: {1/(end_time-start_time)}")
# # create vid for pcd_%d
# os.system(f'ffmpeg -framerate 30 -i {o3d_vis.save_path}/pcd_%d.png {o3d_vis.save_path}/pcd.mp4')
