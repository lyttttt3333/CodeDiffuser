import glob
import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from d3fields.fusion import Fusion


def load_ref(root_path, name_list):
    ref_dict = {}
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(curr_dir, "attn_features", root_path)
    bg_paths = glob.glob(os.path.join(ref_path, "bg_*.txt"))
    bg_feats = []
    for path in bg_paths:
        feat = np.loadtxt(path).reshape(1, -1)
        bg_feats.append(feat)
    bg_feats = np.concatenate(bg_feats, axis=0)
    ref_dict["background"] = bg_feats
    for name in name_list:
        path = os.path.join(ref_path, name + ".txt")
        if os.path.exists(path):
            feat = np.loadtxt(path).reshape(1, -1)
            ref_dict[name] = feat
        else:
            list_feats = []
            path_list = glob.glob(os.path.join(ref_path, f"{name}_*.txt"))
            for path in path_list:
                feat = np.loadtxt(path).reshape(1, -1)
                list_feats.append(feat)
            list_feats = np.concatenate(list_feats, axis=0)
            ref_dict[name] = list_feats
    return ref_dict


def local_image_to_data_url(np_array):
    import base64
    import io

    from PIL import Image

    if np_array.dtype != np.uint8:
        np_array = np_array.astype(np.uint8)
    image = Image.fromarray(np_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    base64_url = f"data:image/png;base64,{img_base64}"
    return base64_url


def crop_pts_feats(pts, feats, root_path, num=6000, crop=None):
    if crop is not None:
        x_upper = crop["x_upper"]
        x_lower = crop["x_lower"]
        y_upper = crop["y_upper"]
        y_lower = crop["y_lower"]
        z_upper = crop["z_upper"]
        z_lower = crop["z_lower"]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        select_index = (
            (z > z_lower)
            * (z < z_upper)
            * (x < x_upper)
            * (x > x_lower)
            * (y > y_lower)
            * (y < y_upper)
        )
        pts = pts[select_index, :]
        feats = feats[select_index, :]

    vis_pcd(np2o3d(pts))

    pts, idx, _ = fps_np(pts, num)
    vis_pcd(np2o3d(pts))
    feats = feats[idx, :]
    np.save(os.path.join(root_path, "src/pts.npy"), pts)
    np.save(os.path.join(root_path, "src/feats.npy"), feats)
    return pts, feats


def fps_np(pcd, particle_num, init_idx=-1, seed=0):
    np.random.seed(seed)
    fps_idx = []
    assert pcd.shape[0] > 0
    if init_idx == -1:
        rand_idx = np.random.randint(pcd.shape[0])
    else:
        rand_idx = init_idx
    fps_idx.append(rand_idx)
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while len(pcd_fps_lst) < particle_num:
        fps_idx.append(dist.argmax())
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps, fps_idx, dist.max()


def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    if pcd.shape[0] == 0:
        return pcd_o3d
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        color = color.reshape(-1, 3)
        # color = color - color.min()
        # color = color / color.max()
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def vis_pcd(pcd_o3d):
    import copy

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    curr_pcd = copy.deepcopy(pcd_o3d)
    visualizer.add_geometry(curr_pcd)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    visualizer.add_geometry(origin)
    visualizer.run()


def text_from_path(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


class Vision:
    def __init__(
        self,
        fusion: Fusion,
        boundaries=None,
        query_threshold=None,
        N_per_inst=None,
        dynamics_dict: Optional[dict] = None,
    ):
        self.fusion = fusion
        self.boundaries = boundaries
        self.query_threshold = query_threshold
        self.N_per_inst = N_per_inst
        self.dynamics_dict = dynamics_dict
        self.return_feature = False

        self.single_detection_mode = False
        self.main_image = None
        self.label = []
        self.num_inst = -1
        self.bbox = []
        self.mask = None
        self.masked_image = []
        self.masked_pcd = []
        self.main_camera = -1
        self.attn_flag = []

        #self.vis_flag = True
        self.vis_flag = False

    def update_obs(self, obs):
        self.obs = obs

    def updata_ee_pcd(self, ee_pcd):
        self.ee_pcd = ee_pcd

    def clear_xmem(self):
        self.fusion.xmem_first_mask_loaded = False

    def detection(self, query_texts, cam_idx=0):
        assert 0 <= cam_idx < 4
        self.single_detection_mode = True
        self.main_camera = cam_idx
        self.fusion.main_camera = cam_idx
        self.fusion.update(self.obs, update_dino=False)
        self.fusion.detection(
            query_texts,
            self.boundaries,
            voxel_size=0.012,
            merge_iou=1,
        )
        self.main_image = self.fusion.curr_obs_torch["color"][cam_idx]
        self.label = self.fusion.curr_obs_torch["consensus_mask_label"]
        self.num_inst = len(self.label)
        self.attn_flag = [False] * self.num_inst
        self.mask = (
            self.fusion.curr_obs_torch["mask"][cam_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
            .transpose((2, 0, 1))
        )
        for obj_idx in range(self.num_inst):
            mask = self.mask[obj_idx]
            mask_idx = np.where(mask != 0)
            bbox = (
                np.min(mask_idx[0]),
                np.max(mask_idx[0]),
                np.min(mask_idx[1]),
                np.max(mask_idx[1]),
            )
            invert_mask = np.logical_not(mask)
            image_with_mask = self.main_image.copy()
            image_with_mask[invert_mask] = [255, 255, 255]
            image_with_mask = image_with_mask[bbox[0] : bbox[1], bbox[2] : bbox[3]]
            self.masked_image.append(image_with_mask)
            self.bbox.append(bbox)

        # NOTE: maybe, change the logic of LMP, it should output a inst_idx instead of pcd and then check through num of points
        # NOTE: in d3f_proc, the first time, detect-->LMP-->mask, rest of the time, xmem update mask

        obj_pcd, _ = self.fusion.extract_pcd_in_box(
            boundaries=self.boundaries,
            downsample=True,
            downsample_r=0.002,
        )
        self.masked_pcd = self.fusion.select_pcd_from_mask(obj_pcd)
        # for pcd in self.masked_pcd:
        #     self.center.append(np.mean(pcd, axis=0))

    def update(self,src_dict, th):
        self.src_dict = src_dict
        self.pts = src_dict["pts"]
        self.pts_feat = src_dict["feat"]
        self.cam_num = src_dict["img"].shape[0]
        self.cluster_threshold = th

    def get_one_instance(self, idx, frame="world"):
        pcd = self.masked_pcd[idx]
        if frame != "world":
            gpt_t_world = np.linalg.inv(self.world_t_gpt)
            pcd = tf_pts(gpt_t_world, pcd)
        return pcd

    def get_all_instance(self, key, frame="world"):
        if key not in self.label:
            return []
        all_instance = []
        for i in range(self.num_inst):
            if key == self.label[i]:
                all_instance.append(self.masked_pcd[i])
        if frame != "world":
            gpt_t_world = np.linalg.inv(self.world_t_gpt)
            all_instance = [tf_pts(gpt_t_world, pcd) for pcd in all_instance]
        return all_instance

    # def get_obj(self, key, frame="world"):
    #     if key not in self.attn_group:
    #         return []
    #     if frame == "world":
    #         return self.attn_group[key]
    #     else:
    #         pcd_ls = self.attn_group[key]
    #         gpt_t_world = np.linalg.inv(self.world_t_gpt)
    #         pcd_ls = [tf_pts(gpt_t_world, pcd) for pcd in pcd_ls]
    #         return pcd_ls

    def get_label_img(self, key, img_idx = 0):
        import matplotlib.pyplot as plt 
        bbox_list = self.bbox[key][img_idx]
        image_with_label = self.draw_bounding_boxes(self.src_dict["img"][img_idx], bbox_list)
        if self.vis_flag:
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB))
            plt.show()
        string = self.image_to_base64(image_with_label)
        return string

    def image_to_base64(self, image, image_format="png"):
        import base64

        success, encoded_image = cv2.imencode(f".{image_format}", image)
        byte_data = encoded_image.tobytes()
        base64_encoded_data = base64.b64encode(byte_data).decode("utf-8")
        base64_string = f"data:image/{image_format};base64,{base64_encoded_data}"
        return base64_string

    def draw_selection(
        self,
        image,
        inst_num,
        marker_color=(255, 0, 0),
        marker_radius=15,
        font_scale=0.7,
        font_color=(255, 255, 255),
        font_thickness=2,
    ):
        padding = np.full((30, image.shape[1], 3), 255, dtype=np.uint8)
        image = np.vstack((padding, image))
        padding = np.full((image.shape[0], 30, 3), 255, dtype=np.uint8)
        image = np.hstack((padding, image, padding))

        center = (image.shape[1] // 2, 15)
        cv2.circle(image, center, marker_radius, marker_color, -1)  # -1 填充圆圈

        text = str(inst_num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=1)
        text_width, text_height = text_size
        text_x = center[0] - text_width // 2
        text_y = center[1] + text_height // 2

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )
        return image

    def draw_bounding_boxes(
        self,
        image_np,
        bounding_boxes,
        box_color=(255, 0, 0),
        box_thickness=2,
        marker_color=(255, 0, 0),
        marker_radius=15,
        font_scale=0.7,
        font_color=(255, 255, 255),
        font_thickness=2,
    ):
        image = image_np.astype(np.uint8)

        for idx, box in enumerate(bounding_boxes):
            if len(box) != 4:
                print(f"无效的边界框: {box}, 跳过.")
                continue
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(
                image, (x_min, y_min), (x_max, y_max), box_color, box_thickness
            )
            center = (x_min, y_min)
            # center = (
            #     int(x_min + (x_max - x_min) / 2),
            #     int(y_min + (y_max - y_min) / 2),
            # )

            # 绘制圆圈
            cv2.circle(image, center, marker_radius, marker_color, -1)  # -1 填充圆圈

            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=1)
            text_width, text_height = text_size
            text_x = center[0] - text_width // 2
            text_y = center[1] + text_height // 2

            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )
        return image

    def draw_bbox(self, bounding_box, img):
        import matplotlib.pyplot as plt

        H, W, _ = img.shape
        y_min, x_min, y_max, x_max = bounding_box[0]
        # y_min, y_max =  H - y_max, H - y_min
        fig, ax = plt.subplots()
        ax.imshow(img)  # 显示图片
        ax.add_patch(
            plt.Rectangle(
                (y_min, x_min),
                y_max - y_min,
                x_max - x_min,
                edgecolor="red",
                facecolor="none",
            )
        )
        plt.axis("off")
        plt.show()

    def bounding_box(self, src_dict):
        self.extrinsic = src_dict["extrinsic"]
        self.intrinsic = src_dict["intrinsic"]
        self.bbox = dict()
        for key in self.attn_group.keys():
            bbox_list = []
            for i in range(self.cam_num):
                mask = self.pts_to_mask(self.attn_group[key], img = src_dict["img"][i], cam_idx=i)
                # self.show_mask(mask[0])
                bbox = self.mask_to_bbox(mask)
                bbox_list.append(bbox)
                # self.draw_bbox(bbox, src_dict["img"][i])
            self.bbox[key] = bbox_list
        self.bbox_inst = {}
        for key in self.bbox.keys():
            if key == "background":
                self.bbox_inst[key] = {"inst_0":self.bbox[key]}
                continue
            else:
                cate_dict = {}
                inst_num = len(self.bbox[key][0])
                for inst_idx in range(inst_num):
                    inst_bbox_list = []
                    for cam_idx in range(self.cam_num):
                        xyxy = self.bbox[key][cam_idx][inst_idx]
                        inst_bbox_list.append(xyxy)
                    cate_dict[f"inst_{inst_idx}"] = inst_bbox_list
                self.bbox_inst[key] = cate_dict

    def pts_to_mask(self, pts_list, img, cam_idx):
        mask_list = list()
        K = torch.from_numpy(self.intrinsic)[cam_idx][None, ...]
        Rt = torch.from_numpy(self.extrinsic)[cam_idx, :3, :][None, ...]
        device = K.device
        dtype = K.dtype
        for idx, pts in enumerate(pts_list):
            pts = torch.from_numpy(pts).to(device).to(dtype)
            pn = pts.shape[0]
            hpts = torch.cat(
                [pts, torch.ones([pn, 1], device=pts.device, dtype=pts.dtype)], 1
            )
            srn = Rt.shape[0]
            KRt = K @ Rt  # rfn,3,4
            last_row = torch.zeros([srn, 1, 4], device=pts.device, dtype=pts.dtype)
            last_row[:, :, 3] = 1.0
            H = torch.cat([KRt, last_row], 1)  # rfn,4,4
            pts_cam = H[:, None, :, :] @ hpts[None, :, :, None]
            pts_cam = pts_cam[:, :, :3, 0]
            depth = pts_cam[:, :, 2:]
            invalid_mask = torch.abs(depth) < 1e-4
            depth[invalid_mask] = 1e-3
            pts_2d = pts_cam[:, :, :2] / depth
            pts_2d = pts_2d[0].cpu().numpy().astype(np.int16)

            mask = np.zeros([img.shape[0], img.shape[1]])
            pts_2d[:, 1][pts_2d[:, 1] >= mask.shape[0]] = mask.shape[0] - 1
            pts_2d[:, 0][pts_2d[:, 0] >= mask.shape[1]] = mask.shape[1] - 1
            mask[pts_2d[:, 1], pts_2d[:, 0]] = 1
            mask_list.append(mask[None, ...])

        return np.concatenate(mask_list, axis=0)

    def mask_to_bbox(self, mask_mat):
        bbox_list = list()
        for i in range(mask_mat.shape[0]):
            mask = mask_mat[i]
            indices = np.argwhere(mask)
            x_min, y_min = indices.min(axis=0)
            x_max, y_max = indices.max(axis=0) + 1
            x_min = x_min - 10
            y_min = y_min - 10
            x_max = x_max + 10
            y_max = y_max + 10
            bbox = np.array([y_min, x_min, y_max, x_max])
            bbox_list.append(bbox[None, :])
        return np.concatenate(bbox_list, axis=0)

    def show_mask(self, mask, rgb=False):
        import matplotlib.pyplot as plt

        if rgb == False:
            plt.imshow(mask, cmap="binary", interpolation="nearest")
            plt.axis("off")
            plt.show()
        else:
            plt.imshow(mask)
            plt.axis("off")
            plt.show()

    def semantic_cluster(self, ref_dict, vis):
        self.attn_group={}
        for key in ref_dict.keys():
            attn_pts = self.filter_out_attn(self.pts_feat, ref_feat=ref_dict[key], pcd = self.pts)
            # attn_pts = density_based_downsampling_numpy(attn_pts, 0.008, 6) hang mug
            attn_pts = density_based_downsampling_numpy(attn_pts, 0.008, 6)
            # if vis:
            #     vis_pcd(np2o3d(attn_pts))
            if key != "background":
                attn_pts_list, _ = self.cluster(attn_pts, None, vis=vis)
                print(f"cluster after filtering: {len(attn_pts_list)}")
                # attn_pts_list = [attn[None,...] for attn in attn_pts_list]
                # attn_pts_list = np.concatenate(attn_pts_list,axis= -1)
                self.attn_group[key] = attn_pts_list
                print(key, len(attn_pts_list))
                # for attn in attn_pts_list:
                #     vis_pcd(np2o3d(attn))
            else:
                self.attn_group[key] = [attn_pts]

    def cluster(self, attention_pts, type="dbscan", vis=True):
        X = attention_pts
        labels = custom_cluster(X, self.cluster_threshold)
        if vis:
            n_clusters_found = len(np.unique(labels))
            # print(f"cluster: {n_clusters_found}")
            # YX: visualize the clustering result
            visualize_clusters(
                X,
                labels,
                self.src_dict["pts"],
                title="Clustering Result on 3D Point Cloud",
            )

        unique_labels = set(labels)
        # print(unique_labels)
        attention_pts_list = list()
        attention_centroid_list = list()
        pts_num_list = []
        for label in unique_labels:
            if label != -1 or len(unique_labels) == 1:
                indice = labels == label
                attention_group = attention_pts[indice]
                pts_num = attention_group.shape[0]
                pts_num_list.append(pts_num)
                attention_pts_list.append(attention_group)
                attention_mean = np.mean(attention_group, axis=0)
                attention_centroid_list.append(attention_mean[None, :])
        max_pts_num = max(pts_num_list)
        filter_pts_list = []
        filter_centroid_list = []
        for idx, num in enumerate(pts_num_list):
            if num / max_pts_num < 0.05:
                continue
            else:
                filter_pts_list.append(attention_pts_list[idx])
                filter_centroid_list.append(attention_centroid_list[idx])

        return filter_pts_list, filter_centroid_list

    def compute_max_dist(self, np_points):
        points = torch.from_numpy(np_points)
        distance_matrix = torch.cdist(points, points, p=2)
        max_distance = distance_matrix.max()
        return max_distance.numpy().astype(np_points.dtype)

    def bbox_to_pcd(self, inst_name, inst_idx):
        mask_list = self.multi_view_bbox_to_mask(inst_name, inst_idx)
        cam_params = []
        for cam_idx in range(self.cam_num):
            single_cam = []
            single_cam.append(self.intrinsic[cam_idx])
            single_cam.append(self.extrinsic[cam_idx, :3, :])
            cam_params.append(single_cam)
        pts = apply_masks(points_3d=self.pts, masks=mask_list, cameras=cam_params)
        vis_pcd(np2o3d(pts))

    def multi_view_bbox_to_mask(self, inst_name, inst_index=0):
        multi_view_bbox = self.bbox_inst[inst_name][f"inst_{inst_index}"]
        mask_list = []
        for cam_idx, bbox in enumerate(multi_view_bbox):
            mask = self.single_view_bbox_to_mask(bbox, cam_idx)
            mask_list.append(mask)
        mask_list = np.concatenate(mask_list, axis=0)
        return mask_list

    def single_view_bbox_to_mask(self, bbox, cam_idx):
        image = self.src_dict["img"][cam_idx]
        real_mask = self.segment(self.sam_model, image, [bbox])
        # self.show_mask(real_mask[0][...,None]*image,rgb=True)
        return real_mask

    def filter_out_attn(self, src_feat_map, ref_feat, pcd):
        src_feat_map = torch.from_numpy(src_feat_map).to(torch.float32)
        ref_feat = torch.from_numpy(ref_feat).to(torch.float32)

        similarity_list = []
        for i in range(ref_feat.shape[0]):
            tgt_feat = ref_feat[i]
            assert src_feat_map.shape[1] == tgt_feat.shape[0]
            similarity = torch.nn.functional.cosine_similarity(
                src_feat_map, tgt_feat[None], dim=1
            )
            similarity = torch.nan_to_num(similarity, nan=0.0)
            similarity = (similarity - torch.min(similarity)) / (
                torch.max(similarity) - torch.min(similarity)
            )
            similarity_list.append(similarity[None, :])

        similarity_list = torch.cat(similarity_list, dim=0)
        similarity = torch.max(similarity_list, dim=0)[0]

        # 0.8
        attn_indices = (similarity > 0.85).tolist()
        pcd = pcd[attn_indices, :]
        return pcd

    def fps_np(self, pcd, particle_num, init_idx=-1, seed=0):
        np.random.seed(seed)
        # pcd: (n, c) numpy array
        # pcd_fps: (particle_num, c) numpy array
        # radius: float
        fps_idx = []
        assert pcd.shape[0] > 0
        if init_idx == -1:
            rand_idx = np.random.randint(pcd.shape[0])
            # rand_idx = findClosestPoint(pcd, pcd.mean(axis=0))
        else:
            rand_idx = init_idx
        fps_idx.append(rand_idx)
        pcd_fps_lst = [pcd[rand_idx]]
        dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
        while len(pcd_fps_lst) < particle_num:
            fps_idx.append(dist.argmax())
            pcd_fps_lst.append(pcd[dist.argmax()])
            dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
        pcd_fps = np.stack(pcd_fps_lst, axis=0)
        return pcd_fps, fps_idx, dist.max()

    def color_pts(self, backbone_pts, attn_pts):
        backbone_pts = torch.from_numpy(backbone_pts)
        attn_pts = np.concatenate(attn_pts, axis=0)
        attn_pts = torch.from_numpy(attn_pts)
        dist_mat = torch.cdist(backbone_pts, attn_pts)
        min_dist = torch.min(dist_mat, dim=-1)[0]
        index = (min_dist < 0.008).numpy().astype(np.float64).reshape(-1, 1)
        backbone_pts = backbone_pts.numpy().astype(np.float64)
        return np.concatenate([backbone_pts, index], axis=-1)

    def decouple_into_bbox(self, attn_dict):
        cam_view_bbox = []
        attn_flag_list = []
    
        for cam_idx in range(self.cam_num):
            bbox_list = []
            for key in self.attn_group.keys():
                if key in self.dynamics_dict["movable"]:
                    inst_num = self.bbox[key][cam_idx].shape[0]
                    for inst_idx in range(inst_num):
                        bbox_list.append(self.bbox[key][cam_idx][inst_idx][None, ...])
            bbox_list = np.concatenate(bbox_list, axis=0)
            cam_view_bbox.append(bbox_list[None, ...])
    
        cam_view_bbox = np.concatenate(cam_view_bbox, axis=0)
    
        for key in self.attn_group.keys():
            if key in self.dynamics_dict["movable"]:
                inst_num = self.bbox[key][cam_idx].shape[0]
                for inst_idx in range(inst_num):
                    if key in attn_dict.keys():
                        inst_pts = self.attn_group[key][inst_idx]
                        tgt_pts = attn_dict[key]
                        # NOTE: is this the correct way to match?
                        if inst_pts.shape[0] == tgt_pts.shape[0]:
                            attn_flag_list.append(1)
                        else:
                            attn_flag_list.append(0)
                    else:
                        attn_flag_list.append(0)
        return cam_view_bbox, attn_flag_list
    
    def get_attn_for_training(self, ground_truth_dict):
        attn_dict = {}
        for key in ground_truth_dict.keys():
            all_clusters = self.attn_group[key]
            anchor = ground_truth_dict[key]
            dist = [np.linalg.norm(np.mean(pts, axis=0) - anchor) for pts in all_clusters]
            sel_idx = dist.index(min(dist))
            attn_dict[key] = self.attn_group[key][sel_idx]
        return attn_dict

    def find_match_mask(self, attn_list):
        for attn_pts in attn_list:
            for idx in range(self.num_inst):
                inst_pts = self.masked_pcd[idx]
                # NOTE: is this the correct way to match?
                if inst_pts.shape[0] == attn_pts.shape[0]:
                    self.attn_flag[idx] = True
        if self.vis_flag:
            attn_mask = self.mask[self.attn_flag]
            attn_image = self.fusion.annotate_cv2(self.main_image, attn_mask)
            cv2.imshow("attn_image", attn_image)
            cv2.waitKey(0)

        return self.mask[self.attn_flag], self.attn_flag
    
    def get_obj(self,key):
        return self.attn_group[key]


def density_based_downsampling_numpy(points, radius, threshold):
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True)
    keep_indices = np.where(counts >= threshold)[0]
    downsampled_points = points[keep_indices]
    return downsampled_points


def project_point(K, extrinsic, point_3D):
    """
    将3D点投影到2D相机平面。

    参数:
    - K: 相机内参矩阵 (3x3)
    - R: 旋转矩阵 (3x3)
    - t: 平移向量 (3x1)
    - point_3D: 3D点坐标 (3x1)

    返回:
    - point_2D: 2D点像素坐标 (2x1)
    """
    # 将3D点转换为齐次坐标
    point_3D_hom = np.concatenate(
        [point_3D, np.ones([point_3D.shape[0], 1])], axis=-1
    )  # (4,)

    # 计算相机坐标系中的点
    points_cam = point_3D_hom @ extrinsic.T  # (3,)

    # 投影到图像平面 (透视投影)
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]

    # 将相机坐标系中的点转换到像素坐标系
    points_image_hom = (K @ np.vstack((x, y, np.ones_like(x)))).T  # (N, 3)

    # 归一化以获得像素坐标
    points_2D = points_image_hom[:, :2] / points_image_hom[:, 2, np.newaxis]

    return points_2D


def apply_masks(points_3d, masks, cameras):
    filtered_indices = np.ones(points_3d.shape[0], dtype=bool)

    for mask, (camera_matrix, extrinsic) in zip(masks, cameras):
        points_2d = project_point(camera_matrix, extrinsic, points_3d[filtered_indices])
        x, y = points_2d[:, 0].astype(int), points_2d[:, 1].astype(int)
        valid_indices = (
            (x >= 0)
            & (x < mask.shape[1])
            & (y >= 0)
            & (y < mask.shape[0])
            & (mask[y, x] > 0)
        )
        filtered_indices[filtered_indices] &= valid_indices

    return points_3d[filtered_indices]


def custom_cluster(X, r):
    from scipy.spatial.distance import cdist

    n_points = X.shape[0]
    labels = np.full(n_points, -1)
    current_label = 0

    distance_matrix = cdist(X, X, "euclidean")
    adjacency_matrix = distance_matrix <= r

    def dfs(start_idx, label):
        stack = [start_idx]
        labels[start_idx] = label
        while stack:
            idx = stack.pop()
            neighbors = np.where(adjacency_matrix[idx])[0]
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = label
                    stack.append(neighbor)

    for idx in range(n_points):
        if labels[idx] == -1:
            dfs(idx, current_label)
            current_label += 1

    return labels


def visualize_clusters(X, labels, full_pcd, title="Clustering Result"):
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # unique_labels = np.unique(labels)
    # colors = plt.cm.get_cmap('rainbow', len(unique_labels))

    # for label in unique_labels:
    #     cluster_points = X[labels == label]
    #     ax.scatter(
    #         cluster_points[:, 0],
    #         cluster_points[:, 1],
    #         cluster_points[:, 2],
    #         s=50,
    #         color=colors(label),
    #         label=f'Cluster {label}'
    #     )

    # ax.set_aspect('equal')
    # ax.set_title(title)
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # ax.legend()
    # plt.show()

    colors = plt.cm.get_cmap("rainbow", len(np.unique(labels)))
    pts_colors = np.zeros((X.shape[0], 3))
    for label in np.unique(labels):
        pts_colors[labels == label] = colors(label)[:3]
    pts_o3d = np2o3d(X, pts_colors)
    gray_colors = 0.5 * np.ones((full_pcd.shape[0], 3))
    full_pcd_o3d = np2o3d(full_pcd, gray_colors)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([full_pcd_o3d, origin, pts_o3d])