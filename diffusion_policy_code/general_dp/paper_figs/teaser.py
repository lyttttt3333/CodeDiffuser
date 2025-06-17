import time

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

from diffusion_policy.dataset.real_dataset import RealDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.data_utils import load_dict_from_hdf5, d3fields_proc, d3fields_proc_for_vis
from diffusion_policy.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import np2o3d, aggr_point_cloud_from_data, o3dVisualizer, draw_keypoints
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
            'x_upper': 0.0,
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


# open pen task
if ckpt_name == "our":
    ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_no_seg_distill_dino_N_4000_pn2.1_r_0.04/checkpoints/latest.ckpt"
    dataset_dir = f"{data_root}/data/real_aloha_demo/teaser"
    epi_path = f"{dataset_dir}/episode_0.hdf5"
elif ckpt_name == "raw_pcd":
    ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_no_seg_no_dino_N_4000_pn2.1_r_0.04/checkpoints/latest.ckpt"
    dataset_dir = f"{data_root}/data/paper_fig_demo/open_pen_raw_pcd"
    epi_path = f"{dataset_dir}/episode_0.hdf5"
elif ckpt_name == "dp":
    ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/open_pen/2024.01.28_pen_Guang_open_pen_original/checkpoints/latest.ckpt"
    dataset_dir = f"{data_root}/data/paper_fig_demo/open_pen_dp"
    epi_path = f"{dataset_dir}/episode_0.hdf5"

# # place can task
# if ckpt_name == "our":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_no_seg_distill_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/place_can_our"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"
# elif ckpt_name == "raw_pcd":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_no_seg_no_dino_N_1000_pn2.1_r_0.04/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/place_can_raw_pcd"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"
# elif ckpt_name == "dp":
#     ckpt_path = "/media/yixuan_2T/diffusion_policy/data/outputs/place_can/2024.01.29_place_can_Guang_place_can_original/checkpoints/latest.ckpt"
#     dataset_dir = f"{data_root}/data/paper_fig_demo/place_can_dp"
#     epi_path = f"{dataset_dir}/episode_0.hdf5"

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
    if 'd3fields' in cfg.task.dataset.shape_meta['obs']:
        cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.02 # for knife only
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

# view_ctrl_info = {
#     'front': [-0.41297139921984272, 0.76069613603421393, 0.50079537942059937],
#     'lookat': [0.33009816660393027, 0.019792663057997747, 0.085635952780270794],
#     'up': [0.21467481777633354, -0.45307550013602943, 0.86523829884557524],
#     'zoom': 0.49999999999999978,
# }
view_ctrl_info = {
    "front" : [ 0.74499635772659722, 0.52783812652277207, 0.40788152589083287 ],
    "lookat" : [ 0.40477802261694829, -0.063561775394286382, 0.009329832705745526 ],
    "up" : [ -0.36785814797644378, -0.18499450825722094, 0.91129436236707273 ],
    "zoom" : 0.49999999999999978
}
o3d_vis = o3dVisualizer(view_ctrl_info=view_ctrl_info, save_path=f'{out_dir}')
o3d_vis.start()

fusion = Fusion(num_cam=4, dtype=torch.float16)
kin_helper = KinHelper(robot_name='trossen_vx300s_v3')

is_first = True
for i in range(40, 70):
# for i in range(len(dataset)):
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
        vis_shape_meta['obs']['d3fields']['info']['boundaries']['x_upper'] = 0.0
        vis_shape_meta['obs']['d3fields']['shape'][1] = 40000
    pcd, pcd_colors, feats_pcd, distilled_feats, raw_feats, rob_mesh = extract_pcd_from_data_dict(raw_data_dict, i, fusion, vis_shape_meta, kin_helper)
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
        vis_boundaries['z_lower'] = 0.023
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
    o3d_vis.render(['pcd'], save_name=f'pcd_{i}')
    o3d_vis.render(action_names_ls, save_name=f'action_{i}')
    o3d_vis.render(rob_mesh_names_ls, save_name=f'rob_mesh_{i}')
    if ckpt_name == 'our':
        d3fields_img = o3d_vis.render(['d3fields'], save_name=f'd3fields_{i}')
        partial_d3fields_img = o3d_vis.render(['partial_d3fields'], save_name=f'partial_d3fields_{i}')
        # blended_img = cv2.addWeighted(pcd_img, 0.6, partial_d3fields_img, 0.4, 0)
        # cv2.imwrite(f'{out_dir}/blended_{i}.png', blended_img)
        o3d_vis.render(['input_fields'], save_name=f'input_fields_{i}')
    
    # visualize actions on 2D image
    for view_idx in range(4):
        img = raw_data_dict['observations']['images'][f'camera_{view_idx}_color'][i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        extrinsic = raw_data_dict['observations']['images'][f'camera_{view_idx}_extrinsics'][i]
        intrinsic = raw_data_dict['observations']['images'][f'camera_{view_idx}_intrinsics'][i]
        robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
        pred_act_pos_in_world = robot_base_pose_in_world @ np.concatenate([pred_act_pos, np.ones((pred_act_pos.shape[0], 1))], axis=1).transpose() # (4, N)
        pred_act_pos_in_cam = extrinsic @ pred_act_pos_in_world # (4, N)
        pred_act_pos_in_cam = pred_act_pos_in_cam[:3].transpose() # (N, 3)
        pred_act_pos_in_pix = np.zeros((pred_act_pos_in_cam.shape[0], 2)) # (N, 2)
        pred_act_pos_in_pix[:, 0] = pred_act_pos_in_cam[:, 0] / pred_act_pos_in_cam[:, 2] * intrinsic[0, 0] + intrinsic[0, 2]
        pred_act_pos_in_pix[:, 1] = pred_act_pos_in_cam[:, 1] / pred_act_pos_in_cam[:, 2] * intrinsic[1, 1] + intrinsic[1, 2]
        pred_act_pos_color_cv2 = (pred_act_pos_color * 255).astype(np.uint8)[:,::-1]
        in_bound_mask = (pred_act_pos_in_pix[:, 0] >= 0) & (pred_act_pos_in_pix[:, 0] < img.shape[0]) \
                    & (pred_act_pos_in_pix[:, 1] >= 0) & (pred_act_pos_in_pix[:, 1] < img.shape[1])
        pred_act_pos_in_pix = pred_act_pos_in_pix[in_bound_mask].astype(np.int32)
        pred_act_pos_color_cv2 = pred_act_pos_color_cv2[in_bound_mask]
        img = draw_keypoints(img, pred_act_pos_in_pix, colors=pred_act_pos_color_cv2, radius=6)
        cv2.imwrite(f'{out_dir}/action_view_{view_idx}_{i}.png', img)
    
    # measure frequency
    end_time = time.time()
    print(f"freq: {1/(end_time-start_time)}")
