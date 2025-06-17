import concurrent.futures
import copy
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import sapien.core as sapien
import torch
import transforms3d
import zarr
from d3fields.fusion import Fusion
from d3fields.utils.my_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms
from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.data_utils import _convert_actions, d3fields_proc, d3fields_proc_2d
from diffusion_policy.common.kinematics_utils import KinHelper
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from filelock import FileLock
from tqdm import tqdm

register_codecs()


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training


def _convert_sapien_to_dp_replay(
    store,
    shape_meta,
    dataset_dir,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    fusion: Optional[Fusion] = None,
    robot_name="panda",
    expected_labels=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    spatial_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)
        elif type == "spatial":
            spatial_keys.append(key)
            max_pts_num = obs_shape_meta[key]["shape"][1]

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)
    kin_helper = KinHelper(robot_name=robot_name)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict = dict()
    rgb_data_dict = dict()
    depth_data_dict = dict()
    spatial_data_dict = dict()
    for epi_idx in tqdm(
        episodes_idx,
        desc="Loading episodes",
        colour="green",
        ascii=" 123456789>",
    ):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        feats_per_epi = list()  # save it separately to avoid OOM
        if True:
            with h5py.File(dataset_path) as file:
                # count total steps
                episode_length = file["cartesian_action"].shape[0]
                attention_config = file["attention_config"]
                # save lowdim data to lowedim_data_dict
                if len(lowdim_keys + ["action"]) != 0:
                    low_dim_dict = {}
                for key in lowdim_keys + ["action"]:
                    data_key = "observations/" + key
                    if key == "action":
                        data_key = (
                            "cartesian_action"
                            if "key" not in shape_meta["action"]
                            else shape_meta["action"]["key"]
                        )
                    if key not in lowdim_data_dict:
                        lowdim_data_dict[key] = list()
                    if key == "action":
                        lowdim_data = file[data_key][()]
                        lowdim_data = _convert_actions(
                            raw_actions=lowdim_data,
                            rotation_transformer=rotation_transformer,
                            action_key=data_key,
                        )
                        assert lowdim_data.shape == (episode_length,) + tuple(
                            shape_meta["action"]["shape"]
                        )
                    elif key == "ee_pos":
                        lowdim_data = file[data_key][()]
                        lowdim_data = _convert_actions(
                            raw_actions=lowdim_data,
                            rotation_transformer=rotation_transformer,
                            action_key=data_key,
                        )
                        assert lowdim_data.shape == (episode_length,) + tuple(
                            shape_meta["obs"][key]["shape"]
                        )
                    elif key == "embedding":
                        data_key = "attention_config/" + key
                        lowdim_data = file[data_key][()]
                        lowdim_data = lowdim_data[None, :]
                        lowdim_data = np.tile(lowdim_data, (episode_length, 1))
                        assert lowdim_data.shape == (episode_length,) + tuple(
                            shape_meta["obs"][key]["shape"]
                        )
                    else:
                        pass
                    low_dim_dict[key] = lowdim_data

                if len(rgb_keys) != 0:
                    rgb_imgs_dict = {}
                
                if not shape_meta["2d_attn"]:
                    for key in rgb_keys:
                        if key not in rgb_data_dict:
                            rgb_data_dict[key] = list()
                        rgb_imgs = file["observations"]["images"][key][()]
                        shape = tuple(shape_meta["obs"][key]["shape"])
                        c, h, w = shape
                        resize_imgs = [
                            cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_AREA)
                            for rgb_img in rgb_imgs
                        ]
                        rgb_imgs = np.stack(resize_imgs, axis=0)
                        assert rgb_imgs[0].shape == (h, w, c)
                        rgb_imgs_dict[key] = rgb_imgs
                else:
                    view_keys = shape_meta["attn"]["info"]["view_keys"]
                    color_seq = np.stack(
                        [
                            file["observations"]["images"][f"{k}_color"][()]
                            for k in view_keys
                        ],
                        axis=1,
                    )  # (T, V, H ,W, C)
                    depth_seq = (
                        np.stack(
                            [
                                file["observations"]["images"][f"{k}_depth"][()]
                                for k in view_keys
                            ],
                            axis=1,
                        )
                        / 1000.0
                    )  # (T, V, H ,W)
                    extri_seq = np.stack(
                        [
                            file["observations"]["images"][f"{k}_extrinsic"][()]
                            for k in view_keys
                        ],
                        axis=1,
                    )  # (T, V, 4, 4)
                    intri_seq = np.stack(
                        [
                            file["observations"]["images"][f"{k}_intrinsic"][()]
                            for k in view_keys
                        ],
                        axis=1,
                    )  # (T, V, 3, 3)
                    qpos_seq = (
                        file["observations"]["full_joint_pos"][()]
                        if "full_joint_pos" in file["observations"]
                        else file["observations"]["joint_pos"][()]
                    )  # (T, -1)
                    if (
                        robot_name == "panda"
                        and "full_joint_pos" not in file["observations"]
                    ):
                        qpos_seq = np.concatenate([qpos_seq, qpos_seq[:, -1:]], axis=-1)
                    if "robot_base_pose_in_world" in file["observations"]:
                        # (T, 4, 4)
                        robot_base_pose_in_world_seq = file["observations"][
                            "robot_base_pose_in_world"
                        ][()]
                    else:
                        init_pos = np.array([0.0, -0.5, 0.0])
                        init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
                        init_pose = sapien.Pose(
                            init_pos, init_ori
                        ).to_transformation_matrix()
                        robot_base_pose_in_world_seq = np.stack(
                            [init_pose] * qpos_seq.shape[0], axis=0
                        )  # (T, 4, 4)

                    rgb_imgs_multi_view, _, _, _ = d3fields_proc_2d(
                        fusion=fusion,
                        shape_meta=shape_meta["attn"],
                        color_seq=color_seq,
                        depth_seq=depth_seq,
                        extri_seq=extri_seq,
                        intri_seq=intri_seq,
                        robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                        teleop_robot=kin_helper,
                        qpos_seq=qpos_seq,
                        attention_config=attention_config,
                        episode_idx=epi_idx,
                        )
                    del color_seq
                    del depth_seq
                    rgb_imgs_dict = {}
                    print(rgb_keys,view_keys)
                    for idx, key in enumerate(rgb_keys):
                        if key not in rgb_data_dict:
                            rgb_data_dict[key] = list()
                        rgb_imgs = rgb_imgs_multi_view[:,idx]
                        shape = tuple(shape_meta["obs"][key]["shape"])
                        c, h, w = shape
                        resize_imgs = [
                            cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_AREA)
                            for rgb_img in rgb_imgs
                        ]
                        rgb_imgs = np.stack(resize_imgs, axis=0)
                        if c == 1:
                            rgb_imgs = rgb_imgs[...,None]
                        assert rgb_imgs[0].shape == (h, w, c)
                        # print(np.sum((origin_rgb_imgs_dict[key] - rgb_imgs)** 2))
                        rgb_imgs_dict[key] = rgb_imgs


                for key in depth_keys:
                    if key not in depth_data_dict:
                        depth_data_dict[key] = list()
                    depth_imgs = file["observations"]["images"][key][()]
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    resize_imgs = [
                        cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_AREA)
                        for depth_img in depth_imgs
                    ]
                    depth_imgs = np.stack(resize_imgs, axis=0)[..., None]
                    depth_imgs = np.clip(depth_imgs, 0, 1000).astype(np.uint16)
                    assert depth_imgs[0].shape == (h, w, c)

                for key in spatial_keys:
                    assert key == "d3fields"  # only support d3fields for now
                    if fusion is None:
                        raise RuntimeError("fusion must be specified")

                    # construct inputs for d3fields processing
                    view_keys = shape_meta["obs"][key]["info"]["view_keys"]
                    use_dino = (
                        shape_meta["obs"][key]["info"]["use_dino"]
                        if "use_dino" in shape_meta["obs"][key]["info"]
                        else False
                    )
                    distill_dino = (
                        shape_meta["obs"][key]["info"]["distill_dino"]
                        if "distill_dino" in shape_meta["obs"][key]["info"]
                        else False
                    )
                    use_seg = (
                        shape_meta["obs"][key]["info"]["use_seg"]
                        if "use_seg" in shape_meta["obs"][key]["info"]
                        else True
                    )
                    is_joint = ("key" in shape_meta["action"].keys()) and (
                        shape_meta["action"]["key"] == "joint_action"
                    )
                    color_seq = np.stack(
                        [
                            file["observations"]["images"][f"{k}_color"][()]
                            for k in view_keys
                        ],
                        axis=1,
                    )  # (T, V, H ,W, C)
                    depth_seq = (
                        np.stack(
                            [
                                file["observations"]["images"][f"{k}_depth"][()]
                                for k in view_keys
                            ],
                            axis=1,
                        )
                        / 1000.0
                    )  # (T, V, H ,W)
                    extri_seq = np.stack(
                        [
                            file["observations"]["images"][f"{k}_extrinsic"][()]
                            for k in view_keys
                        ],
                        axis=1,
                    )  # (T, V, 4, 4)
                    intri_seq = np.stack(
                        [
                            file["observations"]["images"][f"{k}_intrinsic"][()]
                            for k in view_keys
                        ],
                        axis=1,
                    )  # (T, V, 3, 3)
                    qpos_seq = (
                        file["observations"]["full_joint_pos"][()]
                        if "full_joint_pos" in file["observations"]
                        else file["observations"]["joint_pos"][()]
                    )  # (T, -1)
                    if (
                        robot_name == "panda"
                        and "full_joint_pos" not in file["observations"]
                    ):
                        qpos_seq = np.concatenate([qpos_seq, qpos_seq[:, -1:]], axis=-1)
                    if "robot_base_pose_in_world" in file["observations"]:
                        # (T, 4, 4)
                        robot_base_pose_in_world_seq = file["observations"][
                            "robot_base_pose_in_world"
                        ][()]
                    else:
                        init_pos = np.array([0.0, -0.5, 0.0])
                        init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
                        init_pose = sapien.Pose(
                            init_pos, init_ori
                        ).to_transformation_matrix()
                        robot_base_pose_in_world_seq = np.stack(
                            [init_pose] * qpos_seq.shape[0], axis=0
                        )  # (T, 4, 4)

                    aggr_src_pts_ls, _ = d3fields_proc(
                        fusion=fusion,
                        shape_meta=shape_meta["obs"][key],
                        color_seq=color_seq,
                        depth_seq=depth_seq,
                        extri_seq=extri_seq,
                        intri_seq=intri_seq,
                        robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                        teleop_robot=kin_helper,
                        qpos_seq=qpos_seq,
                        attention_config=attention_config,
                    )

                    vis_pcd = False
                    if vis_pcd:
                        import open3d as o3d
                        from d3fields.utils.draw_utils import np2o3d

                        for pts_idx, aggr_src_pts in enumerate(aggr_src_pts_ls):
                            if aggr_src_pts.shape[-1] == 4:
                                weight = aggr_src_pts[:, 3:].reshape(-1, 1)
                                color = weight * np.array([[1, 0, 0]]) + (
                                    1 - weight
                                ) * np.array([[0, 0, 1]])
                            else:
                                color = aggr_src_pts[:, 3:].reshape(-1, 1)
                            pcd_o3d = np2o3d(aggr_src_pts[:, :3], color=color)
                            if pts_idx == 0:
                                visualizer = o3d.visualization.Visualizer()
                                visualizer.create_window()
                                curr_pcd = copy.deepcopy(pcd_o3d)
                                visualizer.add_geometry(curr_pcd)
                                origin = (
                                    o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.1
                                    )
                                )
                                visualizer.add_geometry(origin)

                            curr_pcd.points = pcd_o3d.points
                            curr_pcd.colors = pcd_o3d.colors

                            visualizer.update_geometry(curr_pcd)
                            visualizer.poll_events()
                            visualizer.update_renderer()

                            if pts_idx == 0:
                                visualizer.run()
                        visualizer.close()

                    if key not in spatial_data_dict:
                        spatial_data_dict[key] = list()

                    spatial_data_dict[key] = spatial_data_dict[key] + aggr_src_pts_ls

                episode_end = prev_end + episode_length
                prev_end = episode_end
                episode_ends.append(episode_end)

                for key in lowdim_keys + ["action"]:
                    lowdim_data_dict[key].append(low_dim_dict[key])

                for key in rgb_keys:
                    rgb_data_dict[key].append(rgb_imgs_dict[key])

                for key in depth_keys:
                    depth_data_dict[key].append(depth_imgs)

            if fusion is not None:
                fusion.clear_xmem_memory()
        else:
            if fusion is not None:
                fusion.clear_xmem_memory()
            print(f"Loading Error: {dataset_path}")

    input("wait")

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data_dict
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    # dump spatial data
    print("Dumping spatial data")
    for key, data in spatial_data_dict.items():
        # pad to max_pts_num
        for d_i, d in enumerate(data):
            if d.shape[0] > max_pts_num:
                data[d_i] = d[:max_pts_num]
            else:
                data[d_i] = np.pad(
                    d, ((0, max_pts_num - d.shape[0]), (0, 0)), mode="constant"
                )
        data = np.stack(data, axis=0)  # (T, N, 1027)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype,
        )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


class SapienDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_dir: str,
        vis_input=False,
        horizon=1,
        pad_before=0,
        pad_after=0,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=True,
        use_cache=True,
        seed=42,
        val_ratio=0.0,
        manual_val_mask=False,
        manual_val_start=-1,
        n_obs_steps=None,
        robot_name="panda",
        expected_labels=None,
        device="cuda:0",
    ):
        super().__init__()

        rotation_transformer = RotationTransformer(
            from_rep="euler_angles", to_rep=rotation_rep, from_convention="xyz"
        )

        replay_buffer = None
        fusion = None
        cache_info_str = ""
        d3fields_feats_type = None
        for key, attr in shape_meta["obs"].items():
            if ("type" in attr) and (attr["type"] == "depth"):
                cache_info_str += "_rgbd"
                break
        if "d3fields" in shape_meta["obs"]:
            use_seg = shape_meta["obs"]["d3fields"]["info"]["use_seg"]
            use_dino = shape_meta["obs"]["d3fields"]["info"]["use_dino"]
            distill_dino = (
                shape_meta["obs"]["d3fields"]["info"]["distill_dino"]
                if "distill_dino" in shape_meta["obs"]["d3fields"]["info"]
                else False
            )
            d3fields_feats_type = shape_meta["obs"]["d3fields"]["info"]["feats_type"]
            if use_seg:
                cache_info_str += "_seg"
            else:
                cache_info_str += "_no_seg"
            if not use_dino and not distill_dino:
                cache_info_str += "_no_dino"
            elif not use_dino and distill_dino:
                cache_info_str += "_distill_dino"
            else:
                cache_info_str += "_dino"
            if (
                "key" in shape_meta["action"]
                and shape_meta["action"]["key"] == "joint_action"
            ):
                cache_info_str += "_joint"
            elif (
                "key" in shape_meta["action"]
                and shape_meta["action"]["key"] == "motion_prim"
            ):
                cache_info_str += "_motion_prim"
            else:
                cache_info_str += "_eef"
        if use_cache:
            cache_zarr_path = os.path.join(
                dataset_dir, f"cache{cache_info_str}.zarr.zip"
            )
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # create fusion if necessary
                    self.fusion_dtype = torch.float16
                    # If get stuck here, try to `export OMP_NUM_THREADS=1`
                    # refer: https://github.com/pytorch/pytorch/issues/21956
                    for key, attr in shape_meta["obs"].items():
                        if ("type" in attr) and (attr["type"] == "spatial"):
                            num_cam = len(attr["info"]["view_keys"])
                            fusion = Fusion(
                                num_cam=num_cam, dtype=self.fusion_dtype, device=device
                            )
                            break
                    if "attn" in shape_meta.keys():
                        fusion = Fusion(
                            num_cam=shape_meta["num_cam"], dtype=self.fusion_dtype, device=device
                        )
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_sapien_to_dp_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_dir=dataset_dir,
                            rotation_transformer=rotation_transformer,
                            fusion=fusion,
                            robot_name=robot_name,
                            expected_labels=expected_labels,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            # create fusion if necessary
            self.fusion_dtype = torch.float16
            # If get stuck here, try to `export OMP_NUM_THREADS=1`
            # refer: https://github.com/pytorch/pytorch/issues/21956
            for key, attr in shape_meta["obs"].items():
                if ("type" in attr) and (attr["type"] == "spatial"):
                    num_cam = len(attr["info"]["view_keys"])
                    fusion = Fusion(num_cam=num_cam, dtype=self.fusion_dtype)
                    break
            replay_buffer = _convert_sapien_to_dp_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_dir=dataset_dir,
                rotation_transformer=rotation_transformer,
                fusion=fusion,
                robot_name=robot_name,
                expected_labels=expected_labels,
            )
        self.replay_buffer = replay_buffer
        if fusion is not None:
            fusion.close()

        if vis_input:
            if (
                "d3fields" in shape_meta["obs"]
                and shape_meta["obs"]["d3fields"]["info"]["distill_dino"]
            ):
                vis_distill_feats = True
            else:
                vis_distill_feats = False
            self.replay_buffer.visualize_data(
                output_dir=os.path.join(
                    dataset_dir,
                    f"replay_buffer_vis_{get_current_YYYY_MM_DD_hh_mm_ss_ms()}",
                ),
                vis_distill_feats=vis_distill_feats,
            )

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        spatial_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)
            elif type == "spatial":
                spatial_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
            )
        else:
            try:
                assert manual_val_start >= 0
                assert manual_val_start < replay_buffer.n_episodes
            except:
                raise RuntimeError("invalid manual_val_start")
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys + spatial_keys:
                key_first_k[key] = n_obs_steps
        key_last_k = {"action": horizon}

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=n_obs_steps + horizon,
            # sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            dataset_dir=dataset_dir,
            key_first_k=key_first_k,
            key_last_k=key_last_k,
            d3fields_feats_type=d3fields_feats_type,
            shape_meta=shape_meta,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.spatial_keys = spatial_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_dir = dataset_dir
        self.d3fields_feats_type = d3fields_feats_type
        self.key_first_k = key_first_k
        self.key_last_k = key_last_k

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.n_obs_steps + self.horizon,
            # sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            dataset_dir=self.dataset_dir,
            key_first_k=self.key_first_k,
            key_last_k=self.key_last_k,
            d3fields_feats_type=self.d3fields_feats_type,
            shape_meta=self.shape_meta,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.use_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
        else:
            raise RuntimeError("unsupported")
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("embedding"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)

            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        # spatial
        is_joint = ("key" in self.shape_meta["action"].keys()) and (
            self.shape_meta["action"]["key"] == "joint_action"
        )
        for key in self.spatial_keys:
            B, N, C = self.replay_buffer[key].shape
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * N, C))
            if self.shape_meta["obs"][key]["shape"][0] == 1027:
                # compute normalizer for feats of top 10 demos
                feats = []
                for demo_i in range(1):
                    if self.shape_meta["obs"][key]["info"]["use_seg"]:
                        feats_prefix = ""
                    else:
                        feats_prefix = "_no_seg"
                    if is_joint:
                        feats_prefix += "_joint"
                    if os.path.exists(
                        os.path.join(
                            self.dataset_dir,
                            f"feats{feats_prefix}",
                            f"episode_{demo_i}.hdf5",
                        )
                    ):
                        with h5py.File(
                            os.path.join(
                                self.dataset_dir,
                                f"feats{feats_prefix}",
                                f"episode_{demo_i}.hdf5",
                            )
                        ) as file:
                            feats.append(file["feats"][()])
                feats = np.concatenate(feats, axis=0)  # (10, T, N, 1024)
                feats = feats.reshape(-1, 1024)  # (10 * T * N, 1024)
                feat_stat = array_to_stats(feats)
                for stat_key in stat:
                    stat[stat_key] = np.concatenate(
                        [stat[stat_key], feat_stat[stat_key]], axis=0
                    )
                normalizer[key] = get_range_normalizer_from_stat(
                    stat, ignore_dim=[0, 1, 2]
                )
            else:
                normalizer[key] = get_identity_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        obs_T_slice = slice(self.n_obs_steps)
        action_T_slice = slice(self.n_obs_steps, self.n_obs_steps + self.horizon)


        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(sample[key][obs_T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = (
                np.moveaxis(sample[key][obs_T_slice], -1, 1).astype(np.float32) / 1000.0
            )
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][obs_T_slice].astype(np.float32)
            del sample[key]
        for key in self.spatial_keys:
            obs_dict[key] = np.moveaxis(sample[key][obs_T_slice], 1, 2).astype(
                np.float32
            )
            del sample[key]

        action = sample["action"][action_T_slice]
        action_is_pad = sample["action_is_pad"][action_T_slice]

        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action.astype(np.float32)),
            "is_pad": torch.from_numpy(action_is_pad.astype(bool)),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data
