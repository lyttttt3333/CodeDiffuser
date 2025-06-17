import time

import dill
import hydra
import torch
from omegaconf import open_dict, OmegaConf
import diffusers
import cv2
from matplotlib import cm
import numpy as np
import pickle

from diffusion_policy.dataset.real_dataset import RealDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.data_utils import load_dict_from_hdf5, d3fields_proc, d3fields_proc_for_vis
from d3fields.utils.draw_utils import np2o3d, aggr_point_cloud_from_data, o3dVisualizer
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

def extract_pcd_from_data_dict(data_dict, t, fusion, shape_meta, auto_wb_path=None):
    # vis raw pcd
    view_keys = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
    if auto_wb_path is None:
        color_seq = np.stack([data_dict['observations']['images'][f'{k}_color'][t] for k in view_keys], axis=0) # (V, H ,W, C)
        color_seq = np.stack([white_balance_loops(color_seq[i]) for i in range(color_seq.shape[0])], axis=0)
    else:
        color_seq = np.stack([cv2.imread(f'{auto_wb_path}/episode_0_{k}_color_{t}_AWB.png') for k in view_keys], axis=0)[..., ::-1] # (V, H ,W, C)
    depth_seq = np.stack([data_dict['observations']['images'][f'{k}_depth'][t] for k in view_keys], axis=0) / 1000. # (V, H ,W)
    extri_seq = np.stack([data_dict['observations']['images'][f'{k}_extrinsics'][t] for k in view_keys], axis=0) # (V, 4, 4)
    intri_seq = np.stack([data_dict['observations']['images'][f'{k}_intrinsics'][t] for k in view_keys], axis=0) # (V, 3, 3)
    pcd, pcd_colors = aggr_point_cloud_from_data(color_seq, depth_seq, intri_seq, extri_seq, downsample=False, boundaries=shape_meta['obs']['d3fields']['info']['boundaries'], out_o3d=False)
    robot_base_pose_in_world = data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
    pcd = np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).transpose() # (4, N)
    pcd = pcd[:3].transpose() # (N, 3)
    
    # vis d3fields
    aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feat_ls, _ = d3fields_proc_for_vis(
        fusion=fusion,
        shape_meta=shape_meta['obs']['d3fields'],
        color_seq=color_seq[None],
        depth_seq=depth_seq[None],
        extri_seq=extri_seq[None],
        intri_seq=intri_seq[None],
        robot_base_pose_in_world_seq=data_dict['observations']['robot_base_pose_in_world'][t:t+1],
        return_raw_feats=True,
    )
    
    return pcd, pcd_colors, aggr_src_pts_ls[0], aggr_feats_ls[0], aggr_raw_feat_ls[0]

cfg_path = "/home/yixuan/bdai/general_dp/general_dp/paper_figs/heatmap_vis_cfg.yaml"
cfg = OmegaConf.load(cfg_path)
data_root = cfg.data_root
dataset_dir = cfg.task.dataset_dir
epi_path = f"{dataset_dir}/episode_0.hdf5"
d3fields_root = '/home/yixuan/bdai/general_dp/d3fields_dev/d3fields'
device = 'cuda'

raw_data_dict, _ = load_dict_from_hdf5(epi_path)

view_ctrl_info = {
    'front': [-0.29207358497430574, 0.55611300190168145, 0.778094692229781],
    'lookat': [0.35064170643336739, 0.06367969456844233, 0.035913447487742001],
    'up': [0.25833110885334704, -0.73745617290948928, 0.62403800464097514],
    'zoom': 0.49999999999999978,
}
o3d_vis = o3dVisualizer(view_ctrl_info=view_ctrl_info, save_path=f'{dataset_dir}/vis')
o3d_vis.start()

fusion = Fusion(num_cam=4, dtype=torch.float16)

# # generate pca images
# pca_name = cfg.task.dataset.shape_meta['obs']['d3fields']['info']['distill_obj']
# pca = pickle.load(open(f'{d3fields_root}/pca_model/{pca_name}.pkl', 'rb'))
# example_img_path = f'{d3fields_root}/data/wild/{pca_name}/0.jpg'
# example_img = cv2.imread(example_img_path)
# img_h, img_w = example_img.shape[:2]
# example_params = {
#     'patch_h': 40,
#     'patch_w': 40,
# }
# example_feats = fusion.extract_dinov2_features(example_img[None], example_params)[0].detach().cpu().numpy() # (H, W, D)
# example_pca_feats = pca.transform(example_feats.reshape(-1, example_feats.shape[-1])).reshape(example_feats.shape[:-1] + (3,)) # (H, W, 3)
# pca_norm_min = np.array([-20, -20, -20])
# pca_norm_max = np.array([25, 20, 25])
# example_pca_norm = (example_pca_feats - pca_norm_min) / (pca_norm_max - pca_norm_min)
# example_pca_norm = np.clip(example_pca_norm, 0, 1)
# example_pca_norm = (example_pca_norm * 255).astype(np.uint8)
# example_pca_norm = cv2.resize(example_pca_norm, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
# example_mask = grounded_instance_sam_new_ver(example_img, [pca_name], fusion.ground_dino_model, fusion.sam_model, box_thresholds=[0.3], merge_all=True)[0][1]
# example_pca_norm[~example_mask] = 0
# cv2.imwrite(f'{dataset_dir}/vis/pca_example.png', example_pca_norm)
# cv2.imwrite(f'{dataset_dir}/vis/example.png', example_img)

is_first = True
for i in range(raw_data_dict['observations']['images']['camera_0_color'].shape[0]):
    start_time = time.time()
        
    # visualize pcd
    start_time = time.time()
    vis_shape_meta = cfg.task.dataset.shape_meta.copy()
    vis_shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.01
    vis_shape_meta['obs']['d3fields']['shape'][1] = 40000
    pcd, pcd_colors, feats_pcd, distilled_feats, raw_feats = extract_pcd_from_data_dict(raw_data_dict, i, fusion, vis_shape_meta) # , auto_wb_path=f'{dataset_dir}/auto_wb')
    raw_pcd = np2o3d(pcd, color=pcd_colors)
    o3d_vis.update_pcd(raw_pcd, mesh_name='pcd')
    
    
    # visualize d3fields
    cmap = cm.get_cmap('viridis')
    # d3fields = obs_dict['d3fields'][0, -1].cpu().numpy().transpose() # (N, 3+D)
    # pred_act_pos = pred_action[0, :, :3].cpu().numpy()
    # d3fields_geo = d3fields[:, :3]
    # d3fields_sim = d3fields[:, 3]
    # d3fields_sim_color = cmap(d3fields_sim)[:,:3]
    # d3fields_o3d = np2o3d(d3fields_geo, color=d3fields_sim_color)
    
    vis_boundaries = cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries']
    vis_boundaries['z_lower'] = 0.03
    for ch_i in range(distilled_feats.shape[1]):
        d3fields_sim = distilled_feats[:, ch_i]
        robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
        feats_pcd_in_rob = robot_base_pose_in_world @ np.concatenate([feats_pcd, np.ones((feats_pcd.shape[0], 1))], axis=1).transpose() # (4, N)
        feats_pcd_in_rob = feats_pcd_in_rob[:3].transpose() # (N, 3)
        d3fields_bound_mask = (feats_pcd_in_rob[:, 0] > vis_boundaries['x_lower']) & (feats_pcd_in_rob[:, 0] < vis_boundaries['x_upper']) \
                            & (feats_pcd_in_rob[:, 1] > vis_boundaries['y_lower']) & (feats_pcd_in_rob[:, 1] < vis_boundaries['y_upper']) \
                            & (feats_pcd_in_rob[:, 2] > vis_boundaries['z_lower']) & (feats_pcd_in_rob[:, 2] < vis_boundaries['z_upper'])
        d3fields_sim[~d3fields_bound_mask] = 0
        d3fields_sim_color = cmap(d3fields_sim)[:,:3]
        high_sim_mask = d3fields_sim > 0.5
        d3fields_o3d = np2o3d(feats_pcd, color=d3fields_sim_color)
        partial_d3fields_o3d = np2o3d(feats_pcd[high_sim_mask], color=d3fields_sim_color[high_sim_mask])
        o3d_vis.update_pcd(d3fields_o3d, mesh_name=f'd3fields_{ch_i}')
        o3d_vis.update_pcd(partial_d3fields_o3d, mesh_name=f'partial_d3fields_{ch_i}')
    
    # # visualize pca features
    # pca_feats = pca.transform(raw_feats) # (N, 3)
    # pca_feats_norm = (pca_feats - pca_norm_min) / (pca_norm_max - pca_norm_min)
    # pca_feats_norm = np.clip(pca_feats_norm, 0, 1)[:,::-1]
    # pca_feats_norm[~d3fields_bound_mask] = 0.8
    # pca_feats_o3d = np2o3d(feats_pcd, color=pca_feats_norm)
    # o3d_vis.update_pcd(pca_feats_o3d, mesh_name='pca_feats')
    
    # render and save
    pcd_img = o3d_vis.render(['pcd'], save_name=f'raw_pcd_{i}')
    for ch_i in range(distilled_feats.shape[1]):
        d3fields_img = o3d_vis.render([f'd3fields_{ch_i}'], save_name=f'd3fields_ch_{ch_i}_{i}')
        # pca_img = o3d_vis.render(['pca_feats'], save_name=f'pca_feats_{i}')
        partial_d3fields_img = o3d_vis.render([f'partial_d3fields_{ch_i}'], save_name=f'partial_d3fields_ch_{ch_i}_{i}')
        blended_img = cv2.addWeighted(pcd_img, 0.4, partial_d3fields_img, 0.6, 0)
        cv2.imwrite(f'{dataset_dir}/vis/blended_ch_{ch_i}_{i}.png', blended_img)
    # o3d_vis.render(['pcd', 'd3fields'], save_name=f'pcd_d3fields_{i}')
    # o3d_vis.render()
    
    # measure frequency
    end_time = time.time()
    print(f"freq: {1/(end_time-start_time)}")
