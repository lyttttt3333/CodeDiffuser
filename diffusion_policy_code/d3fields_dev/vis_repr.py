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
    depth_max = 3
    depth_min = 0.0
    depth = np.clip(depth, depth_min, depth_max)
    depth = (depth - depth_min) / (depth_max - depth_min)
    depth = cmap(depth)[:, :, :3]
    return (depth * 255).astype(np.uint8)


def vis_main(scene):

    hyper_param_dict = {
        "pack": {
            "data_path": "/home/yitong/diffusion/data_train/data_delta_test/episode_0.hdf5",
            "pca_path": "/home/yitong/diffusion/d3fields_dev/d3fields/pca_model/mug.pkl",
            "prompt": ['battery', 'crate'],
        }
    }

    # define the path
    data_path = hyper_param_dict[scene]["data_path"]
    pca_path = hyper_param_dict[scene]["pca_path"]
    result_path = f"/home/yitong/diffusion/data_train/data_realworld/fields/{scene}"
    os.system(f"mkdir -p {result_path}")
    query_texts = hyper_param_dict[scene]["prompt"]
    query_thresholds = [0.25]
    device = "cuda"

    # hyper-parameter
    t = 10
    dtype = torch.float16

    step = 0.002

    x_lower = -0.3
    x_upper =  0.3
    y_lower =  -0.3
    y_upper =  0.25
    z_lower =  -0.3
    z_upper =  0.5

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
        "direct_up_view",
    ]

    # cam_name_list = [
    #     "camera_0",
    #     "camera_1",
    #     "camera_2",
    #     "camera_3",
    #     "camera_4",
    # ]

    colors = []
    for i in cam_name_list:
        img = data_dict["observations"]["images"][f"{i}_color"][t]
        # cv2.imwrite(f"{result_path}/{i}_color.png", img)
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
    )  # [N, H, W]
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

    H, W = colors.shape[1:3]

    cv2.imwrite(f"{result_path}/color_0.png", colors[0][..., ::-1])
    cv2.imwrite(f"{result_path}/color_1.png", colors[1][..., ::-1])
    cv2.imwrite(f"{result_path}/color_2.png", colors[2][..., ::-1])
    cv2.imwrite(f"{result_path}/color_3.png", colors[3][..., ::-1])
    cv2.imwrite(f"{result_path}/color_4.png", colors[4][..., ::-1])
    cv2.imwrite(f"{result_path}/depth_0.png", vis_depth(depths[0])[..., ::-1])
    cv2.imwrite(f"{result_path}/depth_1.png", vis_depth(depths[1])[..., ::-1])
    cv2.imwrite(f"{result_path}/depth_2.png", vis_depth(depths[2])[..., ::-1])
    cv2.imwrite(f"{result_path}/depth_3.png", vis_depth(depths[3])[..., ::-1])
    np.save(f"{result_path}/pose.npy", extrinsics[0])
    np.save(f"{result_path}/intrinsics.npy", intrinsics[0])

    # load pca and fusion
    # pca = pickle.load(open(pca_path, "rb"))
    fusion = Fusion(num_cam=4, feat_backbone="dinov2", dtype=dtype, device=device)

    # define visualizer
    view_ctrl_info = {
        "front": [-0.4, -0.7, 0.7],
        "lookat": [0, 0, -0.1],
        "up": [1, 1.8, 0],
        "zoom": 0.75,
    }
    o3d_vis = o3dVisualizer(view_ctrl_info=view_ctrl_info, save_path=result_path)
    o3d_vis.start()

    obs = {
        "color": colors,
        "depth": depths,
        "pose": extrinsics[:, :3],  # (N, 3, 4)
        "K": intrinsics,
    }

    pcd = aggr_point_cloud_from_data(
        colors[..., ::-1],
        depths,
        intrinsics,
        extrinsics,
        downsample=True,
        boundaries=boundaries,
    )
    pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)

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
    vertices, triangles = fusion.extract_mesh(init_grid, out, grid_shape)

    # eval mask and feature of vertices
    vertices_tensor = torch.from_numpy(vertices).to(device, dtype=dtype)
    print("eval mesh vertices")
    with torch.no_grad():
        out = fusion.batch_eval(
            vertices_tensor, return_names=["dino_feats", "mask", "color_tensor"]
        )

    # create mask mesh
    mask_meshes = fusion.create_instance_mask_mesh(vertices, triangles, out, out_o3d=True)
    for m_i, mask_mesh in enumerate(mask_meshes):
        o3d_vis.update_custom_mesh(mask_mesh, f"mask_{m_i}")

    # create feature mesh
    # feature_mesh, pca_max, pca_min = fusion.create_descriptor_mesh(
    #     vertices,
    #     triangles,
    #     out,
    #     {"pca": pca},
    #     mask_out_bg=True,
    #     out_o3d=True,
    #     z_lower=0.01,
    # )
    # o3d_vis.update_custom_mesh(feature_mesh, "feature")

    # create color mesh
    color_mesh = fusion.create_color_mesh(vertices, triangles, out, out_o3d=True)
    o3d_vis.update_custom_mesh(color_mesh, "color")

    np.save(f"{result_path}/vertices.npy", vertices)
    np.savetxt(f"{result_path}/vertices.txt", vertices)
    np.save(f"{result_path}/triangles.npy", triangles)
    np.save(f"{result_path}/vertices_feats.npy", out["dino_feats"].cpu().numpy())
    np.save(f"{result_path}/vertices_color.npy", out["color_tensor"].cpu().numpy())

    # render them
    for m_i, mask_mesh in enumerate(mask_meshes):
        o3d_vis.render([f"mask_{m_i}"], save_name=f"d3f_mask_{m_i}", blocking=False)
    o3d_vis.render(["feature"], save_name="d3f_feature", blocking=False)
    o3d_vis.render(["color"], save_name="d3f_color", blocking=False)


if __name__ == "__main__":
    scene_list = ["pack"]
    for scene in scene_list:
        vis_main(scene)
