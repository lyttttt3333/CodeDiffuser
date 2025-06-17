import os

import cv2
import numpy as np
import open3d as o3d
import pyrender
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import trimesh
from matplotlib import colormaps as cm
from PIL import Image
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.data_utils import load_dict_from_hdf5
from d3fields.utils.draw_utils import (
    draw_keypoints,
    o3dVisualizer,
)


def compute_similarity_tensor(src_feat_map, tgt_feat):
    # :param src_feat_map: [B, C, **dim] torch tensor
    # :param tgt_feat: [C] torch tensor
    # :param scale: float
    # :return: [B, **dim] torch tensor

    assert src_feat_map.shape[1] == tgt_feat.shape[0]

    similarity = torch.nn.functional.cosine_similarity(
        src_feat_map, tgt_feat[None], dim=1
    )
    # np.savetxt("/home/yitong/diffusion/data_train/data_realworld/tgt_feat_slot.txt", tgt_feat.cpu().numpy())
    similarity[similarity<0.7]=0

    assert similarity.shape[0] == src_feat_map.shape[0]
    return similarity


def vis_corr(
    src_info,
    tgt_info,
    x,
    y,
    scene,
    vertices_feats_tensor,
    mesh,
    o3d_mesh,
    H,
    W,
    renderer,
    bbox=None,
):
    # x = 396
    # y = 270
    cmap = cm.get_cmap("viridis")

    num_tgt = len(tgt_info["color"])
    src_color_render_curr = draw_keypoints(
        src_info["color"],
        np.array([[x, y]]),
        colors=[(255, 0, 0)],
        radius=int(5 * src_info["color"].shape[1] / 360),
    )
    cv2.imshow("src", src_color_render_curr[..., ::-1])
    feats_h, feats_w = src_info["dino_feats"].shape[:2]
    img_h, img_w = src_info["color"].shape[:2]
    src_feat_tensor = src_info["dino_feats"][
        int(y * feats_h / img_h), int(x * feats_w / img_w)
    ]
    # src_feat_tensor = torch.from_numpy(np.load("/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/tg_ft_crate.npy")).to(src_feat_tensor.device).to(src_feat_tensor.dtype)
    tgt_feat_sims_tensor = compute_similarity_tensor(
        vertices_feats_tensor, src_feat_tensor
    )  # [N]
    tgt_feat_sims_norm = tgt_feat_sims_tensor.detach().cpu().numpy()  # [N]
    # indices = (tgt_feat_sims_norm >= 0.85)

    vertices_color = (cmap(tgt_feat_sims_norm)[:, :3] * 255).astype(np.uint8)
    if bbox is not None:
        x_lower = bbox["x_lower"]
        x_upper = bbox["x_upper"]
        y_lower = bbox["y_lower"]
        y_upper = bbox["y_upper"]
        z_lower = bbox["z_lower"]
        z_upper = bbox["z_upper"]
        vertices = np.array(mesh.vertices)
        mask = (
            (vertices[:, 0] > x_lower)
            & (vertices[:, 0] < x_upper)
            & (vertices[:, 1] > y_lower)
            & (vertices[:, 1] < y_upper)
            & (vertices[:, 2] > z_lower)
            & (vertices[:, 2] < z_upper)
        )
        vertices_color[~mask] = (cmap(np.zeros(1))[0, :3] * 255).astype(np.uint8)
    mesh.visual.vertex_colors = vertices_color
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_color / 255.0)

    src_img_h, src_img_w = src_info["color"].shape[:2]
    src_img_w_scale = int(src_img_w / src_img_h * H)
    src_color_render_curr = cv2.resize(
        src_color_render_curr,
        (src_img_w_scale, H),
        interpolation=cv2.INTER_NEAREST,
    )

    final_img = np.zeros(
        (H * num_tgt, int(W * 2 + src_img_w_scale), 3),
        dtype=np.uint8,
    )
    final_img[:H, :src_img_w_scale] = src_color_render_curr[..., ::-1]

    # update scene
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    nodes = scene.get_nodes()
    for node in nodes:
        if node.mesh is not None:
            scene.remove_node(node)
            break
    scene.add(render_mesh, pose=np.eye(4))

    heatmap_ls = []
    tgt_imshow_ls = []
    for idx in range(num_tgt):
        # update camera
        opencv_T_opengl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        tgt_pose = tgt_info["pose"][idx].detach().cpu().numpy()
        tgt_pose = np.linalg.inv(tgt_pose)
        opengl_pose = np.matmul(tgt_pose, opencv_T_opengl)

        nodes = scene.get_nodes()
        for node in nodes:
            if node.camera is not None:
                node.translation = opengl_pose[:3, 3]
                node.rotation = R.from_matrix(opengl_pose[:3, :3]).as_quat()

        # render scene
        heatmap, _ = renderer.render(scene)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255
        heatmap = heatmap.astype(np.uint8)

        heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        min_val, max_val = np.percentile(
            heatmap_gray, [1, 99]
        )  # Adjusting to a wider percentile range
        heatmap_val_scaled = (heatmap_gray - min_val) / (max_val - min_val + 1e-6)
        heatmap_val = np.clip(heatmap_val_scaled, 0, 1)

        # Prepare the colored heatmap (retain original colors)
        colored_heatmap_for_blending = heatmap * heatmap_val[..., None]

        # Blend the heatmap with the original image
        original_image = tgt_info["color"][idx]  # Assuming this is the original image
        blended_image = (
            original_image * (1 - heatmap_val[..., None]) + colored_heatmap_for_blending
        )

        threshold = 190
        mask_below_threshold = heatmap_gray < threshold

        overlay_mask = np.ones_like(heatmap_val)
        overlay_mask[mask_below_threshold] = 0
        blended_image = (
            original_image * (1 - overlay_mask[..., None])
            + colored_heatmap_for_blending * overlay_mask[..., None]
        )
        # Ensure the final image is in the correct format
        blended_image_normalized = np.clip(blended_image, 0, 255).astype(np.uint8)

        tgt_imshow_curr = blended_image_normalized
        # tgt_imshow_curr = blended_image
        final_img[H * idx : H * (idx + 1), src_img_w_scale + W :] = heatmap[..., ::-1]
        final_img[
            H * idx : H * (idx + 1),
            src_img_w_scale : src_img_w_scale + W,
        ] = tgt_imshow_curr[..., ::-1]
        heatmap_ls.append(heatmap[..., ::-1])
        tgt_imshow_ls.append(tgt_imshow_curr[..., ::-1])
    return {
        "final_img": final_img,
        "heatmap": np.stack(heatmap_ls),
        "tgt_imshow_curr": np.stack(tgt_imshow_ls),
        "src_color_render_curr": src_color_render_curr[..., ::-1],
    }


def interactive_corr(
    src_info,
    tgt_info,
    vertices,
    traingles,
    vertices_color,
    vertices_feats,
    o3d_vis,
    output_dir,
    bbox=None,
):
    # :param src_info: dict contains:
    #                  - 'color': (H, W, 3) np array, color image
    #                  - 'dino_feats': (H, W, f) torch tensor, dino features
    # :param tgt_info: dict contains:
    #                  - 'color': (K, H, W, 3) np array, color image
    #                  - 'pose': (K, 3, 4) torch tensor, pose of the camera
    #                  - 'K': (K, 3, 3) torch tensor, camera intrinsics
    # :param vertices: (N, 3) numpy array in world frame
    # :param traingles: (M, 3) numpy array, the indices of the vertices
    # :param vertices_color: (N, 3) numpy array, the color of the vertices
    # :param vertices_feats: (N, f) numpy array, the features of the vertices
    os.system(f"mkdir -p {output_dir}")
    H = 480
    W = 640
    num_tgt = len(tgt_info["color"])
    imshow_scale = 0.5
    device = "cuda"
    vertices_feats_tensor = torch.from_numpy(vertices_feats).to(device)

    def drawHeatmap(event, x, y, flags, param):
        mesh = param["mesh"]
        o3d_mesh = param["o3d_mesh"]
        scene = param["scene"]
        renderer = param["renderer"]
        vid = param["vid"]
        click = param["click"]
        # if event == cv2.EVENT_LBUTTONDOWN:
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            if event == cv2.EVENT_LBUTTONDOWN:
                param["click"] = param["click"] + 1
                param["pixels"] = np.concatenate(
                    [param["pixels"], np.array([[x, y]])], axis=0
                )
            vis_out = vis_corr(
                src_info,
                tgt_info,
                x,
                y,
                scene,
                vertices_feats_tensor,
                mesh,
                o3d_mesh,
                H,
                W,
                renderer,
                bbox,
            )

            final_img = vis_out["final_img"]
            if event == cv2.EVENT_LBUTTONDOWN:
                o3d_vis.update_custom_mesh(o3d_mesh, "mesh")
                o3d_vis.render()
                o3d.io.write_triangle_mesh(f"{output_dir}/{click}_tgt.ply", o3d_mesh)
                cv2.imwrite(
                    f"{output_dir}/{click}_src.png", vis_out["src_color_render_curr"]
                )
                for idx, heatmap in enumerate(vis_out["heatmap"]):
                    cv2.imwrite(f"{output_dir}/{click}_tgt_{idx}.png", heatmap)
                    cv2.imwrite(
                        f"{output_dir}/{click}_final_{idx}.png",
                        vis_out["tgt_imshow_curr"][idx],
                    )
                cv2.imwrite(f"{output_dir}/{click}_final.png", final_img)

            vid.write(final_img)
            final_imshow_img = cv2.resize(
                final_img.astype(np.uint8),
                (
                    int(final_img.shape[1] * imshow_scale),
                    int(final_img.shape[0] * imshow_scale),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("final", final_imshow_img)

    cv2.imshow("src", src_info["color"][..., ::-1])

    mesh = trimesh.Trimesh(vertices=vertices, faces=traingles[..., ::-1], process=False)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(traingles[..., ::-1])
    # o3d_mesh.visual.vertex_colors = o3d.utility.Vector3dVector(vertices_color / 255.)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_color[:, :3])
    # save the mesh
    o3d.io.write_triangle_mesh(f"{output_dir}/mesh_src.ply", o3d_mesh)
    # o3d_mesh.compute_vertex_normals()
    # o3d_mesh.compute_triangle_normals()
    # o3d_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw_geometries([o3d_mesh])
    o3d_vis.update_custom_mesh(o3d_mesh, "mesh")
    o3d_vis.render()

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0, 0, 0])
    camera = pyrender.IntrinsicsCamera(
        fx=tgt_info["K"][0, 0, 0].item(),
        fy=tgt_info["K"][0, 1, 1].item(),
        cx=tgt_info["K"][0, 0, 2].item(),
        cy=tgt_info["K"][0, 1, 2].item(),
    )
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.7)
    light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5.0], [0, 0, 0, 1]])
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    scene.add(render_mesh, pose=np.eye(4))
    scene.add(light, pose=light_pose)

    opencv_T_opengl = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    tgt_pose = tgt_info["pose"][0].cpu().numpy()
    tgt_pose = np.linalg.inv(tgt_pose)
    opengl_pose = np.matmul(tgt_pose, opencv_T_opengl)
    scene.add(camera, pose=opengl_pose)

    src_img_h, src_img_w = src_info["color"].shape[:2]
    src_img_w_scale = int(src_img_w / src_img_h * H)
    vid = cv2.VideoWriter(
        f"{output_dir}/corr.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (src_img_w_scale + 2 * W, H * num_tgt),
    )

    param = {
        "mesh": mesh,
        "o3d_mesh": o3d_mesh,
        "scene": scene,
        "renderer": pyrender.OffscreenRenderer(W, H),
        # 'renderer': pyrender.Renderer(W, H),
        "vid": vid,
        "click": 0,
        "pixels": np.zeros((0, 2)),
    }

    cv2.setMouseCallback("src", drawHeatmap, param)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vid.release()
    return param["pixels"]


def extract_dinov2_feats(imgs, model):
    device = "cuda"
    dtype = torch.float16
    K, H, W, _ = imgs.shape

    patch_h = H // 20
    patch_w = W // 20
    # feat_dim = 384 # vits14
    # feat_dim = 768 # vitb14
    feat_dim = 1024  # vitl14
    # feat_dim = 1536 # vitg14

    transform = T.Compose(
        [
            # T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    imgs_tensor = torch.zeros(
        (K, 3, patch_h * 14, patch_w * 14), device=device, dtype=dtype
    )
    for j in range(K):
        img = Image.fromarray(imgs[j])
        imgs_tensor[j] = transform(img)[:3]
    with torch.no_grad():
        features_dict = model.forward_features(imgs_tensor.to(dtype=dtype))
        features = features_dict["x_norm_patchtokens"]
        features = features.reshape((K, patch_h, patch_w, feat_dim))
    features = features.permute(0, 3, 1, 2)
    features = F.interpolate(
        features, size=(H, W), mode="bilinear", align_corners=False
    )
    features = features.permute(0, 2, 3, 1)
    return features


def main():
    o3d_vis = o3dVisualizer()
    o3d_vis.start()
    device = "cuda"

    hyper_param_dict = {
        "pack": {
            # "src_path": "/home/neo/Documents/general_dp/d3fields_dev/data/blue_can.jpg",
            # "src_path": "/home/neo/Documents/general_dp/d3fields_dev/data/red_can.jpg",
            # "src_path": "/home/neo/Documents/general_dp/d3fields_dev/data/blue_can_small.png",
            # "src_path": "/home/neo/Documents/general_dp/d3fields_dev/data/red_can_small.png",
            "src_path": "/home/yitong/diffusion/data_train/data_realworld/fields/pack/color_2.png",
            "tgt_hdf5": "/home/yitong/diffusion/data_train/data_realworld/battery.hdf5",
        },
        # "mug": {
        #     "src_path": "/home/ywang/d3fields_dev/data/wild/mug/0.png",
        #     "tgt_hdf5": "/home/ywang/d3fields_dev/data/camera_only/episode_1.hdf5",
        # },
        # "hammer": {
        #     "src_path": "/home/ywang/d3fields_dev/data/wild/hammer/0.jpg",
        #     "tgt_hdf5": "/home/ywang/d3fields_dev/data/camera_only/episode_2.hdf5",
        # },
        # "shoe": {
        #     "src_path": "/home/ywang/d3fields_dev/data/wild/shoe/0.jpg",
        #     "tgt_hdf5": "/home/ywang/d3fields_dev/data/camera_only/episode_3.hdf5",
        # },
        # "drill": {
        #     "src_path": "/home/ywang/d3fields_dev/data/wild/drill/1.jpg",
        #     "tgt_hdf5": "/home/ywang/d3fields_dev/data/camera_only/episode_7.hdf5",
        # },
    }

    scene = "pack"
    tgt_dir = f"/home/yitong/diffusion/data_train/data_realworld/fields/{scene}"
    src_path = hyper_param_dict[scene]["src_path"]
    tgt_hdf5 = hyper_param_dict[scene]["tgt_hdf5"]
    bbox = {
        "x_upper": 0.3,
        "x_lower": -0.3,
        "y_upper": 0.3,
        "y_lower": -0.4,
        "z_upper": 0.5,
        "z_lower": -0.3,
    }

    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dinov2_model = dinov2_model.to(device=device, dtype=torch.float16)

    src_img = cv2.imread(src_path)[..., ::-1]
    src_feats = extract_dinov2_feats(src_img[None], dinov2_model)[0]

    src_info = {"color": cv2.imread(src_path)[..., ::-1], "dino_feats": src_feats}

    tgt_data_dict, _ = load_dict_from_hdf5(tgt_hdf5)
    cam_name_list = [
        "left_bottom_view",
        "right_bottom_view",
        "left_top_view",
        "right_top_view",
    ]
    cam_name_list = [
        "camera_0",
        "camera_1",
        "camera_2",
        "camera_3",
        "camera_4",
    ]
    camera_name = cam_name_list[2]
    pose = tgt_data_dict["observations"]["images"][f"{camera_name}_extrinsics"][0]
    K = tgt_data_dict["observations"]["images"][f"{camera_name}_intrinsics"][0]
    img = tgt_data_dict["observations"]["images"][f"{camera_name}_color"][0:1][0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tgt_info = {
        "color": [img],
        "pose": torch.from_numpy(pose).to(device)[None],
        "K": torch.from_numpy(K).to(device)[None],
    }

    vertices = np.load(f"{tgt_dir}/vertices.npy")
    triangles = np.load(f"{tgt_dir}/triangles.npy")
    vertices_color = np.load(f"{tgt_dir}/vertices_color.npy")
    vertices_feats = np.load(f"{tgt_dir}/vertices_feats.npy")
    pixels = interactive_corr(
        src_info,
        tgt_info,
        vertices,
        triangles,
        vertices_color,
        vertices_feats,
        o3d_vis,
        output_dir=f"d3fields_dev/data/corr/{scene}",
        bbox=bbox,
    )


if __name__ == "__main__":
    main()
