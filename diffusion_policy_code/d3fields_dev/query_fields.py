import os
import pickle

import cv2
import numpy as np
import torch
from matplotlib import colormaps as cm

from d3fields.fusion import Fusion, create_init_grid
from diffusion_policy.common.data_utils import load_dict_from_hdf5
from d3fields.utils.draw_utils import (
    aggr_point_cloud_from_data,
    o3dVisualizer,
)


def vis_depth(depth: np.ndarray) -> np.ndarray:
    cmap = cm.get_cmap("viridis")
    depth = depth.copy()
    depth_max = 1.0
    depth_min = 0.0
    depth = np.clip(depth, depth_min, depth_max)
    depth = (depth - depth_min) / (depth_max - depth_min)
    depth = cmap(depth)[:, :, :3]
    return (depth * 255).astype(np.uint8)


def vis_main(scene):

    hyper_param_dict = {
        "hang_mug": {
            "data_path": "/home/sim/general_dp-neo-attention_map/data/data/sapien_demo/hang_mug_demo_150_multi/episode_1.hdf5",
            "pca_path": "/home/sim/general_dp-neo-attention_map/diffusion_policy_code/d3fields_dev/d3fields/pca_model/mug.pkl",
            "prompt": ['red mug', 'gray mug'],
        }
    }

    # define the path
    data_path = hyper_param_dict[scene]["data_path"]
    result_path = f"/home/sim/general_dp-neo-attention_map/fields/{scene}"
    os.system(f"mkdir -p {result_path}")
    query_texts = hyper_param_dict[scene]["prompt"]
    query_thresholds = [0.25]
    device = "cuda"

    # hyper-parameter
    t = 10
    dtype = torch.float16

    step = 0.002

    x_upper = 0.20
    x_lower = -0.20
    y_upper = 0.20
    y_lower = -0.20
    z_upper = 0.6
    z_lower = -0.02

    boundaries = {
        "x_lower": x_lower,
        "x_upper": x_upper,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "z_lower": z_lower,
        "z_upper": z_upper,
    }

    # load data 
    data_dict, _ = load_dict_from_hdf5(data_path)

    cam_name_list = [
        "left_bottom_view",
        "right_bottom_view",
        "left_top_view",
        "right_top_view",
    ]

    colors = []
    for i in cam_name_list:
        img = data_dict["observations"]["images"][f"{i}_color"][t]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        colors.append(img)
    colors = np.stack(colors, axis=0)

    depths = (
        np.stack(
            [
                data_dict["observations"]["images"][f"{i}_depth"][t]
                for i in cam_name_list
            ],
            axis=0,
        )
        / 1000.0
    )
    extrinsics = np.stack(
        [
            data_dict["observations"]["images"][f"{i}_extrinsic"][t]
            for i in cam_name_list
        ]
    )
    intrinsics = np.stack(
        [
            data_dict["observations"]["images"][f"{i}_intrinsic"][t]
            for i in cam_name_list
        ]
    )

    obs = {
        "color": colors,
        "depth": depths,
        "pose": extrinsics[:, :3],  # (N, 3, 4)
        "K": intrinsics,
    }

    fusion = Fusion(num_cam=4, feat_backbone="dinov2", dtype=dtype, device=device)

    fusion.update(obs)
    fusion.text_queries_for_inst_mask_no_track(
        query_texts, query_thresholds, boundaries=boundaries, merge_iou=0.05
    )

    # visualize mesh
    init_grid, grid_shape = create_init_grid(boundaries, step)
    init_grid = init_grid.to(device=device, dtype=fusion.dtype)

    print("eval init grid")
    with torch.no_grad():
        out = fusion.batch_eval(init_grid, return_names=[])

    # extract mesh
    print("extract mesh")
    vertices, _ = fusion.extract_mesh(init_grid, out, grid_shape)

    # eval mask and feature of vertices
    vertices_tensor = torch.from_numpy(vertices).to(device, dtype=dtype)
    print("eval mesh vertices")
    with torch.no_grad():
        out = fusion.batch_eval(
            vertices_tensor, return_names=["dino_feats", "mask", "color_tensor"]
        )

    np.save(f"{result_path}/vertices.npy", vertices)
    np.save(f"{result_path}/vertices_feats.npy", out["dino_feats"].cpu().numpy())


if __name__ == "__main__":
    # scene_list = ["mug", "spoon", "hammer", "shoe", "toothpaste"]
    scene_list = ["hang_mug"]
    for scene in scene_list:
        vis_main(scene)
