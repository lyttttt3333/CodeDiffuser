import os
import time

import cv2
import h5py
import numpy as np
import torch
from d3fields.fusion import Fusion
from d3fields.utils.draw_utils import np2o3d
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, merge_pcd
from d3fields.utils.lmp_exec import Attention
from d3fields.utils.lmp_utils import Vision, local_image_to_data_url, np2o3d, vis_pcd
from diffusion_policy.common.kinematics_utils import KinHelper
from tqdm import tqdm
from utils.my_utils import bcolors



def create_init_grid(boundaries, step_size):
    x_lower, x_upper = boundaries["x_lower"], boundaries["x_upper"]
    y_lower, y_upper = boundaries["y_lower"], boundaries["y_upper"]
    z_lower, z_upper = boundaries["z_lower"], boundaries["z_upper"]
    x = torch.arange(x_lower, x_upper, step_size, dtype=torch.float32) + step_size / 2
    y = torch.arange(y_lower, y_upper, step_size, dtype=torch.float32) + step_size / 2
    z = torch.arange(z_lower, z_upper, step_size, dtype=torch.float32) + step_size / 2
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords, xx.shape


### ALOHA fixed constants
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    + MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    + PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
)

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
    MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
    PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    MASTER_GRIPPER_JOINT_NORMALIZE_FN(x)
)

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)

MASTER_POS2JOINT = (
    lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE)
    / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
)
PUPPET_POS2JOINT = (
    lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE)
    / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
)

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2


def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    """
    ....
    """
    with h5py.File(filename, "w") as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, "/", dic, config_dict)


def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            # h5file[path + key] = item
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(
                path + key, shape=item.shape, **config_dict[key]
            )
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(
                h5file, path + key + "/", item, config_dict[key]
            )
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_hdf5(filename):
    """
    ....
    """
    # with h5py.File(filename, 'r') as h5file:
    #     return recursively_load_dict_contents_from_group(h5file, '/')
    h5file = h5py.File(filename, "r")
    return recursively_load_dict_contents_from_group(h5file, "/"), h5file


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


def modify_hdf5_from_dict(filename, dic):
    """
    Modify hdf5 file from a dictionary
    """
    with h5py.File(filename, "r+") as h5file:
        recursively_modify_hdf5_from_dict(h5file, "/", dic)


def recursively_modify_hdf5_from_dict(h5file, path, dic):
    """
    Modify hdf5 file from a dictionary recursively
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray) and key in h5file[path]:
            h5file[path + key][...] = item
        elif isinstance(item, dict):
            recursively_modify_hdf5_from_dict(h5file, path + key + "/", item)
        else:
            raise ValueError("Cannot modify %s type" % type(item))


def vis_distill_feats(pts, feats):
    """visualize distilled features

    Args:
        pts (np.ndarray): (N, 3)
        feats (np.ndarray): (N, f) ranging in [0, 1]
    """
    import open3d as o3d
    from matplotlib import cm

    cmap = cm.get_cmap("viridis")
    for i in range(pts.shape[1]):
        feats_i = feats[:, i]
        colors = cmap(feats_i)[:, :3]
        pts_o3d = np2o3d(pts, color=colors)
        o3d.visualization.draw_geometries([pts_o3d])

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


def load_from_model(obj, pts_num, position=np.array([0, 0, 0])):
    if obj == "crate":
        src_pts = np.loadtxt("/home/yitong/diffusion/src_pts/src_crate_model.txt")
    elif obj == "battery":
        src_pts = np.loadtxt("/home/yitong/diffusion/src_pts/src_battery_model.txt")
    src_pts = src_pts + position
    if pts_num is not None:
        src_pts, _, _ = fps_np(src_pts, pts_num)
    return src_pts


def generate_robot_pcd(
    robot_base_pose_in_world: np.ndarray,
    teleop_robot: KinHelper,
    curr_qpos: np.ndarray,
    num_bots: int,
    N_per_inst: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # compute robot pcd
    if "panda" in teleop_robot.robot_name:
        finger_names = ["panda_leftfinger", "panda_rightfinger", "panda_hand"]
        dense_num_pts = [50, 50, 400]
        sparse_num_pts = [
            int(0.2 * N_per_inst),
            int(0.2 * N_per_inst),
            int(0.6 * N_per_inst),
        ]
    elif "trossen_vx300s" in teleop_robot.robot_name:
        finger_names = ["vx300s/left_finger_link", "vx300s/right_finger_link"]
        dense_num_pts = [250, 250]
        sparse_num_pts = [int(0.5 * N_per_inst), int(0.5 * N_per_inst)]
    else:
        raise RuntimeError("unsupported")

    qpos_dim = curr_qpos.shape[0] // num_bots

    dense_ee_pcd_ls = []
    ee_pcd_ls = []
    robot_pcd_ls = []
    for rob_i in range(num_bots):
        # compute robot pcd
        dense_ee_pcd = teleop_robot.compute_robot_pcd(
            curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
            finger_names,
            dense_num_pts,
            pcd_name=f"dense_ee_pcd_{rob_i}",
        )  # (N, 3)
        ee_pcd = teleop_robot.compute_robot_pcd(
            curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
            finger_names,
            sparse_num_pts,
            pcd_name=f"ee_pcd_{rob_i}",
        )
        robot_pcd = teleop_robot.compute_robot_pcd(
            curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
            num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))],
            pcd_name=f"robot_pcd_{rob_i}",
        )

        # transform robot pcd to world frame
        robot_i_pose = robot_base_pose_in_world[rob_i]
        dense_ee_pcd = (
            robot_i_pose
            @ np.concatenate(
                [dense_ee_pcd, np.ones((dense_ee_pcd.shape[0], 1))], axis=-1
            ).T
        ).T[:, :3]
        ee_pcd = (
            robot_i_pose
            @ np.concatenate([ee_pcd, np.ones((ee_pcd.shape[0], 1))], axis=-1).T
        ).T[:, :3]
        robot_pcd = (
            robot_i_pose
            @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T
        ).T[:, :3]

        # save to list
        dense_ee_pcd_ls.append(dense_ee_pcd)
        ee_pcd_ls.append(ee_pcd)
        robot_pcd_ls.append(robot_pcd)
    # convert to numpy array
    dense_ee_pcd = np.concatenate(dense_ee_pcd_ls, axis=0)
    ee_pcd = np.concatenate(ee_pcd_ls, axis=0)
    robot_pcd = np.concatenate(robot_pcd_ls, axis=0)
    return dense_ee_pcd, ee_pcd, robot_pcd

def color_pts(full_pts):
    device = "cuda:0"
    dtype = torch.float32
    none_attn_pts = full_pts[full_pts[:, -1] == 0][:, :3]
    attn_pts = full_pts[full_pts[:, -1] == 1][:, :3]
    tensor_none_attn = torch.from_numpy(none_attn_pts).to(device, dtype=dtype)
    tensor_attn = torch.from_numpy(attn_pts).to(device, dtype=dtype)
    if tensor_none_attn.shape[0] == 0:
        return full_pts
    dist = torch.cdist(tensor_none_attn, tensor_attn)
    min_dist = torch.min(dist, dim=-1)[0]
    attn_flag = min_dist < 0.016
    attn_flag = attn_flag.cpu().numpy()
    full_pts[:, -1][full_pts[:, -1] == 0] = attn_flag
    return full_pts

def load_ref(ref_path, name_list):
    import glob
    ref_dict = {}
    bg_paths = glob.glob(os.path.join(ref_path,"bg_*.txt"))
    bg_feats = []
    for path in bg_paths:
        feat = np.loadtxt(path).reshape(1,-1)
        bg_feats.append(feat)
    bg_feats = np.concatenate(bg_feats, axis =0)
    ref_dict["background"] = bg_feats
    for name in name_list:
        path = os.path.join(ref_path,name+".txt")
        if os.path.exists(path):
            feat = np.loadtxt(path).reshape(1,-1)
            ref_dict[name] = feat
        else:
            list_feats = []
            path_list = glob.glob(os.path.join(ref_path,f"{name}_*.txt"))
            for path in path_list:
                feat = np.loadtxt(path).reshape(1,-1)
                list_feats.append(feat)
            list_feats = np.concatenate(list_feats, axis =0)
            ref_dict[name] = list_feats
    return ref_dict

def d3fields_proc_attn(
    fusion: Fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    expected_labels=None,
    tool_names=[None],
    attention_config=None,
    exclude_threshold=0.02,
    prompt_info=None,

):
    keys = attention_config.keys()
    configuration_dict = {}
    for key in keys:
        configuration_dict[key] = attention_config[key][:]
    configuration_dict["tgt_layout"][-1] = np.array([0.0217115, 0.11065497, 0.26605847])
    
    ground_truth_dict = {}
    ground_truth_dict["mug"] = configuration_dict["init_layout"][int(configuration_dict["init"][0])]
    ground_truth_dict["branch"] = configuration_dict["tgt_layout"][int(configuration_dict["tgt"][0])]

    boundaries = shape_meta["info"]["boundaries"]
    use_seg = shape_meta["info"]["use_seg"]
    use_dino = shape_meta["info"]["use_dino"]
    distill_dino = shape_meta["info"]["distill_dino"]
    N_per_inst = shape_meta["info"]["N_per_inst"]
    max_pts_num = shape_meta["shape"][1]
    resize_ratio = shape_meta["info"]["resize_ratio"]
    reference_frame = shape_meta["info"]["reference_frame"]

    num_bots = 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    # color_seq, depth_seq, intri_seq = resize_color_depth(
    #     color_seq, depth_seq, intri_seq, resize_ratio
    # )

    T, V, H, W, C = color_seq.shape
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_src_pts_color_ls = []


    dynamics_dict = {"static": ["bg", "branch"], "movable": ["mug"]}


    for t in range(T):
        rgb = color_seq[t]#[..., ::-1].copy()
        obs = {
            "color": rgb,
            "depth": depth_seq[t],
            "pose": extri_seq[t][:, :3, :],
            "K": intri_seq[t],
        }

        grain = 10
        fusion.update(
            obs,
            update_dino=(use_seg or use_dino or distill_dino)
            and (not fusion.xmem_first_mask_loaded),
            # grain=grain,
        )

        generated_code = None
        if not fusion.xmem_first_mask_loaded:
            url = local_image_to_data_url(color_seq[0, 3])


            vis = Vision(
                fusion=fusion,
                boundaries=None,
                query_threshold=None,
                N_per_inst=None,
                dynamics_dict=dynamics_dict,
            )

            aggr_pcd, _ = aggr_point_cloud_from_data(
                color_seq[0],
                depth_seq[0],
                intri_seq[0],
                extri_seq[0],
                downsample=True,
                downsample_r=0.002,
                boundaries=shape_meta["info"]["boundaries"],
                out_o3d=False,
            )

            aggr_feat_list, aggr_pcd_list, _, _ = fusion.select_features_from_pcd(
                aggr_pcd,
                18000,
                per_instance=True,
                use_seg=False,
                use_dino=(use_dino or distill_dino),
            )
            aggr_pcd = aggr_pcd_list[0]
            aggr_feat = aggr_feat_list[0]

            a = 1

            # aggr_pcd_tensor = torch.from_numpy(aggr_pcd).to(fusion.device, dtype=fusion.dtype)
            # with torch.no_grad():
            #     out = fusion.batch_eval(aggr_pcd_tensor, return_names=["dino_feats"])
            #     feats = out["dino_feats"].cpu().numpy()

            src_dict = {
                "img": rgb,
                "intrinsic": intri_seq[0],
                "extrinsic": extri_seq[0],
            }
            src_dict["pts"] = aggr_pcd
            src_dict["feat"] = aggr_feat.cpu().numpy()


            lib_root = "/home/yitong/diffusion/ref_lib"
            lib_path = os.path.join(lib_root, "hang_mug")

            ref_dict = load_ref(lib_path, ["mug", "branch"])

            vis.update(src_dict)
            vis.semantic_cluster(ref_dict, 0)
            vis.bounding_box(src_dict)


            
            if prompt_info is not None:

                task = prompt_info["task"]
                obj_list = prompt_info["obj_list"]
                prompt = prompt_info["prompt"]

                curr_dir = os.path.dirname(os.path.abspath(__file__))
                prompt_log_path = os.path.join(curr_dir, "prompt_log.py")
                attention = Attention(
                    vis=vis, feat_dict=None, obj_list=obj_list, out_path=prompt_log_path
                )
                attn_dict, generated_code = attention.compose(
                    task=task, obj_list=obj_list, instruction=prompt, url=url
                )

            else:
                attn_dict = vis.get_attn_for_training(ground_truth_dict)
                
            
            attn_dict_ground_truth = vis.get_attn_for_training(ground_truth_dict)

            bbox_list, attn_list = vis.decouple_into_bbox(attn_dict)

            for key in vis.attn_group.keys():
                if key in vis.dynamics_dict["static"] and key in attn_dict.keys():
                    tgt_pcd = attn_dict[key]
                    fix_obs = np.concatenate(
                        [tgt_pcd, np.ones([tgt_pcd.shape[0], 1])], axis=-1
                    )

            fusion.static_pts = fix_obs
            fusion.init_bbox = bbox_list
            fusion.attn_list = attn_list

        _, ee_pcd, robot_pcd = generate_robot_pcd(
            robot_base_pose_in_world_seq[t],
            teleop_robot,
            qpos_seq[t],
            num_bots,
            N_per_inst,
        )

        if use_seg:
            fusion.exclude_camera = None

            fusion.bbox_for_inst_mask(
                fusion.init_bbox,
                boundaries,
                expected_labels=expected_labels,
                voxel_size=0.012,
                merge_iou=0.005,
                pause=False,
            )
            obj_idx_list, obj_idx_dict = fusion.get_mask_idx_dict()

            # select objects with attention
            attention_obj_idx = []
            query_obj_idx = []
            for idx, item in enumerate(
                [f"inst_{i}" for i in range(len(fusion.attn_list))]
            ):
                if fusion.attn_list[idx] == 1:
                    # if item not in obj_idx_dict:
                    #     # if item != "table":
                    #     #     print(f"{item} not in obj_idx_dict")
                    #     continue
                    attention_obj_idx += obj_idx_dict[item]
                query_obj_idx += obj_idx_dict[item]

            obj_pcd, _ = aggr_point_cloud_from_data(
                color_seq[t],
                depth_seq[t],
                intri_seq[t],
                extri_seq[t],
                downsample=True,
                downsample_r=0.001,
                boundaries=boundaries,
                out_o3d=False,
                excluded_pts=None,
                exclude_threshold=exclude_threshold,
            )
            # if fusion.static_pts.shape[0] > 0:
            #     obj_pcd = np.concatenate([obj_pcd, fusion.static_pts[:, :3]], axis=0)

            src_feat_list, src_pts_list, _, _ = fusion.select_features_from_pcd(
                obj_pcd,
                N_per_inst,
                per_instance=True,
                use_seg=True,
                use_dino=(use_dino or distill_dino),
            )
            
            movable_pcd = np.concatenate(src_pts_list, axis =0)
            if fusion.env_pcd is None:
                env_num = max_pts_num - movable_pcd.shape[0] - ee_pcd.shape[0]
                y = obj_pcd[:,1]
                obj_pcd = obj_pcd[y>0.06]
                pcd_exclude = np.concatenate([movable_pcd,robot_pcd],axis=0)
                if env_num <= 0:
                    raise
                _, env_pcd = merge_pcd(env_pcd=obj_pcd, pcd_to_add=pcd_exclude, env_num=env_num)
                fusion.env_pcd = env_pcd
            else:
                pass



            full_pcd = np.concatenate([fusion.env_pcd, movable_pcd])
            full_pcd = resize_array(full_pcd, max_pts_num - ee_pcd.shape[0])

            pcd_attn = fusion.compute_attn(
                full_pcd, attention_obj_idx, fusion.static_pts
            )
            aggr_src_pts = np.concatenate([full_pcd, pcd_attn[:, None]], axis=-1)
            aggr_src_pts = color_pts(aggr_src_pts)

        ee_no_attn = np.zeros((ee_pcd.shape[0], 1))
        aggr_src_pts = np.concatenate(
            [
                aggr_src_pts,
                np.concatenate((ee_pcd, ee_no_attn), -1),
            ],
            axis=0,
        )

        # transform to reference frame
        if reference_frame == "world":
            pass
        elif reference_frame == "robot":
            aggr_src_pts[:, :3] = (
                np.linalg.inv(robot_base_pose_in_world_seq[t, 0])
                @ np.concatenate(
                    [aggr_src_pts[:, :3], np.ones((aggr_src_pts.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]

        # save to list
        aggr_src_pts = aggr_src_pts.astype(np.float32)
        # aggr_src_pts = color_pts(aggr_src_pts)
        assert aggr_src_pts.shape[0] == max_pts_num
        aggr_src_pts_ls.append(aggr_src_pts)
        aggr_feats_ls.append(None)
        aggr_src_pts_color_ls.append(None)

    return aggr_src_pts_ls, generated_code

def d3fields_proc(
    fusion: Fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    expected_labels=None,
    tool_names=[None],
    attention_config=None,
    exclude_threshold=0.02,
    prompt_info=None,

):
    keys = attention_config.keys()
    configuration_dict = {}
    for key in keys:
        configuration_dict[key] = attention_config[key][:]
    init_name = shape_meta["info"]["init_name"]
    tgt_name = shape_meta["info"]["tgt_name"]
    task_name = shape_meta["info"]["task_name"]
    if task_name == "hang_mug":
        configuration_dict["tgt_layout"][-1] = np.array([0.0217115, 0.11065497, 0.26605847])
    
    ground_truth_dict = {}
    ground_truth_dict[init_name] = configuration_dict["init_layout"][int(configuration_dict["init"][0])]
    ground_truth_dict[tgt_name] = configuration_dict["tgt_layout"][int(configuration_dict["tgt"][0])]

    boundaries = shape_meta["info"]["boundaries"]
    use_attn = shape_meta["info"]["use_attn"]
    N_per_inst = shape_meta["info"]["N_per_inst"]
    max_pts_num = shape_meta["shape"][1]
    reference_frame = shape_meta["info"]["reference_frame"]

    num_bots = 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    # color_seq, depth_seq, intri_seq = resize_color_depth(
    #     color_seq, depth_seq, intri_seq, resize_ratio
    # )

    T, V, H, W, C = color_seq.shape
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_src_pts_color_ls = []

    if use_attn:
        cluster_threshold = shape_meta["info"]["cluster_threshold"]
        dynamics_dict = {"static": ["bg", tgt_name], "movable": [init_name]}


    for t in range(T):
        rgb = color_seq[t]#[..., ::-1].copy()
        obs = {
            "color": rgb,
            "depth": depth_seq[t],
            "pose": extri_seq[t][:, :3, :],
            "K": intri_seq[t],
        }

        grain = 10
        fusion.update(
            obs,
            update_dino = use_attn and (not fusion.xmem_first_mask_loaded),
        )

        generated_code = None
        if (not fusion.xmem_first_mask_loaded) and use_attn:
            url = local_image_to_data_url(color_seq[0, 3])


            vis = Vision(
                fusion=fusion,
                boundaries=None,
                query_threshold=None,
                N_per_inst=None,
                dynamics_dict=dynamics_dict,
            )

            aggr_pcd, _ = aggr_point_cloud_from_data(
                color_seq[0],
                depth_seq[0],
                intri_seq[0],
                extri_seq[0],
                downsample=True,
                downsample_r=0.002,
                boundaries=shape_meta["info"]["boundaries"],
                out_o3d=False,
            )

            aggr_feat_list, aggr_pcd_list, _, _ = fusion.select_features_from_pcd(
                aggr_pcd,
                18000,
                per_instance=True,
                use_seg=False,
                use_dino=use_attn,
            )
            aggr_pcd = aggr_pcd_list[0]
            aggr_feat = aggr_feat_list[0]

            src_dict = {
                "img": rgb,
                "intrinsic": intri_seq[0],
                "extrinsic": extri_seq[0],
            }
            src_dict["pts"] = aggr_pcd
            src_dict["feat"] = aggr_feat.cpu().numpy()

            def get_ref_lib_path(current_file_path):
                ref_lib_path = os.path.join(current_file_path, "../../../../ref_lib")
                return os.path.abspath(ref_lib_path)

            current_file_path = os.path.dirname(os.path.abspath(__file__))
            lib_root = get_ref_lib_path(current_file_path)
            lib_path = os.path.join(lib_root, task_name)

            ref_dict = load_ref(lib_path, [init_name, tgt_name])

            vis.update(src_dict, th= cluster_threshold)
            vis.semantic_cluster(ref_dict, 0)
            vis.bounding_box(src_dict)


            
            if prompt_info is not None:

                task = prompt_info["task"]
                obj_list = prompt_info["obj_list"]
                prompt = prompt_info["prompt"]

                curr_dir = os.path.dirname(os.path.abspath(__file__))
                prompt_log_path = os.path.join(curr_dir, "prompt_log.py")
                attention = Attention(
                    vis=vis, feat_dict=None, obj_list=obj_list, out_path=prompt_log_path
                )
                attn_dict, generated_code = attention.compose(
                    task=task, obj_list=obj_list, instruction=prompt, url=url
                )

            else:
                attn_dict = vis.get_attn_for_training(ground_truth_dict)
            

            bbox_list, attn_list = vis.decouple_into_bbox(attn_dict)

            for key in vis.attn_group.keys():
                if key in vis.dynamics_dict["static"] and key in attn_dict.keys():
                    tgt_pcd = attn_dict[key]
                    fix_obs = np.concatenate(
                        [tgt_pcd, np.ones([tgt_pcd.shape[0], 1])], axis=-1
                    )

            fusion.static_pts = fix_obs
            fusion.init_bbox = bbox_list
            fusion.attn_list = attn_list

        _, ee_pcd, robot_pcd = generate_robot_pcd(
            robot_base_pose_in_world_seq[t],
            teleop_robot,
            qpos_seq[t],
            num_bots,
            N_per_inst,
        )

        if use_attn:
            fusion.exclude_camera = None

            fusion.bbox_for_inst_mask(
                fusion.init_bbox,
                boundaries,
                expected_labels=expected_labels,
                voxel_size=0.012,
                merge_iou=0.005,
                pause=False,
            )
            obj_idx_list, obj_idx_dict = fusion.get_mask_idx_dict()

            # select objects with attention
            attention_obj_idx = []
            query_obj_idx = []
            for idx, item in enumerate(
                [f"inst_{i}" for i in range(len(fusion.attn_list))]
            ):
                if fusion.attn_list[idx] == 1:
                    attention_obj_idx += obj_idx_dict[item]
                query_obj_idx += obj_idx_dict[item]

            obj_pcd, _ = aggr_point_cloud_from_data(
                color_seq[t],
                depth_seq[t],
                intri_seq[t],
                extri_seq[t],
                downsample=True,
                downsample_r=0.001,
                boundaries=boundaries,
                out_o3d=False,
                excluded_pts=None,
                exclude_threshold=exclude_threshold,
            )
            # if fusion.static_pts.shape[0] > 0:
            #     obj_pcd = np.concatenate([obj_pcd, fusion.static_pts[:, :3]], axis=0)

            src_feat_list, src_pts_list, _, _ = fusion.select_features_from_pcd(
                obj_pcd,
                N_per_inst,
                per_instance=True,
                use_seg=True,
                use_dino=use_attn,
            )
            
            if task_name == "hang_mug":
                movable_pcd = np.concatenate(src_pts_list, axis =0)
                if fusion.env_pcd is None:
                    env_num = max_pts_num - movable_pcd.shape[0] - ee_pcd.shape[0]
                    y = obj_pcd[:,1]
                    obj_pcd = obj_pcd[y>0.06]
                    pcd_exclude = np.concatenate([movable_pcd,robot_pcd],axis=0)
                    if env_num <= 0:
                        raise
                    _, env_pcd = merge_pcd(env_pcd=obj_pcd, pcd_to_add=pcd_exclude, env_num=env_num)
                    fusion.env_pcd = env_pcd
            else:
                movable_pcd = src_pts_list[attention_obj_idx[0] - 1]
                if fusion.env_pcd is None:
                    env_num = max_pts_num - movable_pcd.shape[0] - ee_pcd.shape[0]
                    y = obj_pcd[:,1]
                    obj_pcd = obj_pcd[y>-0.1]
                    pcd_exclude = np.concatenate([movable_pcd,robot_pcd],axis=0)
                    if env_num <= 0:
                        raise
                    _, env_pcd = merge_pcd(env_pcd=obj_pcd, pcd_to_add=pcd_exclude, env_num=env_num)
                    fusion.env_pcd = env_pcd



            full_pcd = np.concatenate([fusion.env_pcd, movable_pcd])
            full_pcd = resize_array(full_pcd, max_pts_num - ee_pcd.shape[0])

            pcd_attn = fusion.compute_attn(
                full_pcd, attention_obj_idx, fusion.static_pts
            )
            aggr_src_pts = np.concatenate([full_pcd, pcd_attn[:, None]], axis=-1)
            aggr_src_pts = color_pts(aggr_src_pts)

            ee_no_attn = np.zeros((ee_pcd.shape[0], 1))
            aggr_src_pts = np.concatenate(
                [
                    aggr_src_pts,
                    np.concatenate((ee_pcd, ee_no_attn), -1),
                ],
                axis=0,
            )

        else:
            fusion.exclude_camera = None

            obj_pcd, color = aggr_point_cloud_from_data(
                color_seq[t],
                depth_seq[t],
                intri_seq[t],
                extri_seq[t],
                downsample=True,
                downsample_r=0.003,
                boundaries=boundaries,
                out_o3d=False,
                excluded_pts=robot_pcd,
                exclude_threshold=0.015,
            )

            _, src_pts_list, _, color_list = fusion.select_features_from_pcd(
                obj_pcd,
                max_pts_num - ee_pcd.shape[0],
                per_instance=False,
                use_seg=False,
                use_dino=use_attn,
                color=color, 
            )

            pcd = np.concatenate([src_pts_list[0],color_list[0][:,::-1]],axis=-1)

            pcd = resize_array(pcd, max_pts_num - ee_pcd.shape[0])



            ee_attn = np.zeros((ee_pcd.shape[0], 3))
            aggr_src_pts = np.concatenate(
                [
                    # np.concatenate((pcd, pcd_no_attn), -1),
                    pcd,
                    np.concatenate((ee_pcd, ee_attn), -1),
                ],
                axis=0,
            )

        # transform to reference frame
        if reference_frame == "world":
            pass
        elif reference_frame == "robot":
            aggr_src_pts[:, :3] = (
                np.linalg.inv(robot_base_pose_in_world_seq[t, 0])
                @ np.concatenate(
                    [aggr_src_pts[:, :3], np.ones((aggr_src_pts.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]

        # save to list
        aggr_src_pts = aggr_src_pts.astype(np.float32)
        assert aggr_src_pts.shape[0] == max_pts_num
        aggr_src_pts_ls.append(aggr_src_pts)
        aggr_feats_ls.append(None)
        aggr_src_pts_color_ls.append(None)

    return aggr_src_pts_ls, generated_code


def resize_array(arr, num):
    n = arr.shape[0]
    if n > num:
        arr_resized = arr[:num, :]
    elif n < num:
        padding = np.zeros((num - n, 3)) 
        arr_resized = np.vstack((arr, padding)) 
    else:
        arr_resized = arr 
    return arr_resized


def d3fields_proc_infer(
    fusion: Fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    expected_labels=None,
    tool_names=[None],
    exclude_threshold=0.01,
    exclude_colors=[],
    query_texts=None,
    attention_obj=None,
    attention_config=None,
    env_config=None,
    init_crate_pts_4dim=None,
):
    goal_position = attention_config["goal_position"][:]
    target_obj_layout = env_config["target_layout"][:]
    init_position = target_obj_layout[0, :3]
    wait_obj_layout = env_config["wait_layout"][:]
    done_obj_layout = env_config["done_layout"][:]

    if len(wait_obj_layout.shape) == 1 and wait_obj_layout[0] == 0:
        none_wait = True
    else:
        none_wait = False

    if len(done_obj_layout.shape) == 1 and done_obj_layout[0] == 0:
        none_done = True
    else:
        none_done = False

    layout_list = []
    if not none_done:
        layout_list.append(done_obj_layout)
    if not none_wait:
        layout_list.append(wait_obj_layout)

    if len(layout_list) != 0:
        none_layout = False
        layout = np.concatenate(layout_list, axis=0)
    else:
        none_layout = True

    query_thresholds = shape_meta["info"]["query_thresholds"]
    boundaries = shape_meta["info"]["boundaries"]
    use_seg = shape_meta["info"]["use_seg"] if "use_seg" in shape_meta["info"] else True
    use_dino = (
        shape_meta["info"]["use_dino"] if "use_dino" in shape_meta["info"] else False
    )
    use_dino = True
    distill_dino = (
        shape_meta["info"]["distill_dino"]
        if "distill_dino" in shape_meta["info"]
        else False
    )
    distill_obj = (
        shape_meta["info"]["distill_obj"]
        if "distill_obj" in shape_meta["info"]
        else False
    )
    N_per_inst = shape_meta["info"]["N_per_inst"]
    N_per_inst_robo = 600
    N_total = shape_meta["shape"][1]
    max_pts_num = shape_meta["shape"][1]

    resize_ratio = shape_meta["info"]["resize_ratio"]
    resize_ratio = 1
    reference_frame = (
        shape_meta["info"]["reference_frame"]
        if "reference_frame" in shape_meta["info"]
        else "world"
    )

    num_bots = (
        robot_base_pose_in_world_seq.shape[1]
        if len(robot_base_pose_in_world_seq.shape) == 4
        else 1
    )
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)

    new_color_seq = np.zeros(
        (
            color_seq.shape[0],
            color_seq.shape[1],
            resize_H,
            resize_W,
            color_seq.shape[-1],
        ),
        dtype=np.uint8,
    )
    new_depth_seq = np.zeros(
        (depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32
    )
    new_intri_seq = np.zeros(
        (intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32
    )
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t, v] = cv2.resize(
                color_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_depth_seq[t, v] = cv2.resize(
                depth_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_intri_seq[t, v] = intri_seq[t, v] * resize_ratio
            new_intri_seq[t, v, 2, 2] = 1.0
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []

    crate_pts_4dim = None
    for t in range(T):
        obs = {
            "color": color_seq[t][:, ...],
            "depth": depth_seq[t][:, ...],
            "pose": extri_seq[t][:, :3, :][:, ...],
            "K": intri_seq[t][:, ...],
        }

        fusion.update(obs, update_dino=(use_dino or distill_dino))

        # compute robot pcd
        if "panda" in teleop_robot.robot_name:
            finger_names = ["panda_leftfinger", "panda_rightfinger", "panda_hand"]
            dense_num_pts = [50, 50, 400]
            sparse_num_pts = [
                int(0.2 * N_per_inst_robo),
                int(0.2 * N_per_inst_robo),
                int(0.6 * N_per_inst_robo),
            ]
        elif "trossen_vx300s" in teleop_robot.robot_name:
            finger_names = ["vx300s/left_finger_link", "vx300s/right_finger_link"]
            dense_num_pts = [250, 250]
            sparse_num_pts = [int(0.5 * N_per_inst_robo), int(0.5 * N_per_inst_robo)]
        else:
            raise RuntimeError("unsupported")

        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots

        dense_ee_pcd_ls = []
        ee_pcd_ls = []
        robot_pcd_ls = []
        tool_pcd_ls = []
        for rob_i in range(num_bots):
            tool_name = tool_names[rob_i]
            # compute robot pcd
            dense_ee_pcd = teleop_robot.compute_robot_pcd(
                curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                finger_names,
                dense_num_pts,
                pcd_name=f"dense_ee_pcd_{rob_i}",
            )  # (N, 3)
            ee_pcd = teleop_robot.compute_robot_pcd(
                curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                finger_names,
                sparse_num_pts,
                pcd_name=f"ee_pcd_{rob_i}",
            )
            robot_pcd = teleop_robot.compute_robot_pcd(
                curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))],
                pcd_name=f"robot_pcd_{rob_i}",
            )
            if tool_name is not None:
                tool_pcd = teleop_robot.compute_tool_pcd(
                    curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                    tool_name,
                    N_per_inst_robo,
                    pcd_name=f"tool_pcd_{rob_i}",
                )

            # transform robot pcd to world frame
            robot_base_pose_in_world = (
                robot_base_pose_in_world_seq[t, rob_i]
                if robot_base_pose_in_world_seq is not None
                else None
            )
            dense_ee_pcd = (
                robot_base_pose_in_world
                @ np.concatenate(
                    [dense_ee_pcd, np.ones((dense_ee_pcd.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]
            ee_pcd = (
                robot_base_pose_in_world
                @ np.concatenate([ee_pcd, np.ones((ee_pcd.shape[0], 1))], axis=-1).T
            ).T[:, :3]
            robot_pcd = (
                robot_base_pose_in_world
                @ np.concatenate(
                    [robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]
            if tool_name is not None:
                tool_pcd = (
                    robot_base_pose_in_world
                    @ np.concatenate(
                        [tool_pcd, np.ones((tool_pcd.shape[0], 1))], axis=-1
                    ).T
                ).T[:, :3]

            # save to list
            dense_ee_pcd_ls.append(dense_ee_pcd)
            ee_pcd_ls.append(ee_pcd)
            robot_pcd_ls.append(robot_pcd)
            if tool_name is not None:
                tool_pcd_ls.append(tool_pcd)
        # convert to numpy array
        dense_ee_pcd = np.concatenate(dense_ee_pcd_ls + tool_pcd_ls, axis=0)
        ee_pcd = np.concatenate(ee_pcd_ls + tool_pcd_ls, axis=0)
        robot_pcd = np.concatenate(robot_pcd_ls + tool_pcd_ls, axis=0)

        # post process robot pcd
        ee_pcd_tensor = torch.from_numpy(ee_pcd).to(
            device=fusion.device, dtype=fusion.dtype
        )

        if use_dino or distill_dino:
            ee_eval_res = fusion.eval(ee_pcd_tensor, return_names=["dino_feats"])
            ee_feats = ee_eval_res["dino_feats"]

        fusion.text_queries_for_inst_mask(
            [],
            target_obj_layout,
            query_thresholds,
            boundaries,
            expected_labels=expected_labels,
            robot_pcd=dense_ee_pcd,
            merge_iou=0.001,
        )
        obj_idx_list, obj_idx_dict = fusion.get_mask_idx_dict()

        # select objects with attention
        attention_obj = ["battery"]
        query_obj_idx = list()

        for item in attention_obj:
            if item not in obj_idx_dict:
                continue
            query_obj_idx += obj_idx_dict[item]

        obj_pcd = fusion.extract_masked_pcd(query_obj_idx, boundaries=boundaries)

        if obj_pcd.shape[0] != 0:
            _, src_pts_list, _, label_list = fusion.select_features_from_pcd(
                obj_pcd,
                N_per_inst,
                per_instance=True,
                use_seg=use_seg,
                use_dino=False,
            )
            track = True
        else:
            src_pts_list = list()
            track = False

        if not none_layout:
            for i in range(layout.shape[0]):
                pts = load_from_model(
                    obj="battery", pts_num=200, position=layout[i, :3]
                )
                src_pts_list.append(pts)

        crate_pts = load_from_model(obj="crate", pts_num=None)
        src_pts_list.append(crate_pts)

        vertices_tensor = torch.from_numpy(crate_pts).to(
            fusion.device, dtype=fusion.dtype
        )
        with torch.no_grad():
            out = fusion.batch_eval(vertices_tensor, return_names=["dino_feats"])
            crate_feat = out["dino_feats"]

        init_position, src_pts_list, crate_pts_4dim = add_addition_channel_pack(
            src_pts_list,
            init_position,
            goal_position,
            crate_feat,
            track,
            crate_pts_4dim,
        )
        if init_crate_pts_4dim is None:
            init_crate_pts_4dim = crate_pts_4dim
        else:
            src_pts_list[-1] = init_crate_pts_4dim

        aggr_src_pts = np.concatenate(src_pts_list, axis=0)
        aggr_feats = None

        # max_obj_pts_num = max_pts_num - ee_pcd.shape[0]
        # if aggr_src_pts.shape[0] > max_obj_pts_num:
        #     aggr_src_pts = aggr_src_pts[:max_obj_pts_num]
        # elif aggr_src_pts.shape[0] < max_obj_pts_num:
        #     aggr_src_pts = np.pad(
        #         aggr_src_pts,
        #         ((0, max_obj_pts_num - aggr_src_pts.shape[0]), (0, 0)),
        #         mode="constant",
        #     )

        addition_channel = np.zeros([ee_pcd.shape[0], 1])
        ee_pcd = np.concatenate([ee_pcd, addition_channel], axis=-1)
        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0)

        if aggr_src_pts.shape[0] > max_pts_num:
            aggr_src_pts = aggr_src_pts[:max_pts_num]
        elif aggr_src_pts.shape[0] < max_pts_num:
            aggr_src_pts = np.pad(
                aggr_src_pts,
                ((0, max_pts_num - aggr_src_pts.shape[0]), (0, 0)),
                mode="constant",
            )

        if reference_frame == "world":
            pass
        elif reference_frame == "robot":
            aggr_src_pts = (
                np.linalg.inv(robot_base_pose_in_world_seq[t, 0])
                @ np.concatenate(
                    [aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]

        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        aggr_feats_ls.append(aggr_feats)
    return aggr_src_pts_ls, aggr_feats_ls, init_crate_pts_4dim, None


def d3fields_proc_raw(
    fusion: Fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    expected_labels=None,
    tool_names=[None],
    exclude_threshold=0.01,
    exclude_colors=[],
    query_texts=None,
    attention_obj=None,
):
    query_thresholds = shape_meta["info"]["query_thresholds"]
    boundaries = shape_meta["info"]["boundaries"]
    use_seg = shape_meta["info"]["use_seg"] if "use_seg" in shape_meta["info"] else True
    use_dino = (
        shape_meta["info"]["use_dino"] if "use_dino" in shape_meta["info"] else False
    )
    use_dino = True
    use_attn = False
    distill_dino = (
        shape_meta["info"]["distill_dino"]
        if "distill_dino" in shape_meta["info"]
        else False
    )
    distill_obj = (
        shape_meta["info"]["distill_obj"]
        if "distill_obj" in shape_meta["info"]
        else False
    )
    N_per_inst = shape_meta["info"]["N_per_inst"]
    N_total = shape_meta["shape"][1]
    max_pts_num = shape_meta["shape"][1]

    resize_ratio = shape_meta["info"]["resize_ratio"]
    resize_ratio = 1
    reference_frame = (
        shape_meta["info"]["reference_frame"]
        if "reference_frame" in shape_meta["info"]
        else "world"
    )

    # if eval_phase is not None:
    #     use_seg = True

    num_bots = (
        robot_base_pose_in_world_seq.shape[1]
        if len(robot_base_pose_in_world_seq.shape) == 4
        else 1
    )
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)

    new_color_seq = np.zeros(
        (
            color_seq.shape[0],
            color_seq.shape[1],
            resize_H,
            resize_W,
            color_seq.shape[-1],
        ),
        dtype=np.uint8,
    )
    new_depth_seq = np.zeros(
        (depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32
    )
    new_intri_seq = np.zeros(
        (intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32
    )
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t, v] = cv2.resize(
                color_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_depth_seq[t, v] = cv2.resize(
                depth_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_intri_seq[t, v] = intri_seq[t, v] * resize_ratio
            new_intri_seq[t, v, 2, 2] = 1.0
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []

    for t in range(T):
        obs = {
            "color": color_seq[t][0:4, ...],
            "depth": depth_seq[t][0:4, ...],
            "pose": extri_seq[t][:, :3, :][0:4, ...],
            "K": intri_seq[t][0:4, ...],
        }

        fusion.update(obs, update_dino=(use_dino or distill_dino))

        # compute robot pcd
        if "panda" in teleop_robot.robot_name:
            finger_names = ["panda_leftfinger", "panda_rightfinger", "panda_hand"]
            dense_num_pts = [50, 50, 400]
            sparse_num_pts = [
                int(0.2 * N_per_inst),
                int(0.2 * N_per_inst),
                int(0.6 * N_per_inst),
            ]
        elif "trossen_vx300s" in teleop_robot.robot_name:
            finger_names = ["vx300s/left_finger_link", "vx300s/right_finger_link"]
            dense_num_pts = [250, 250]
            sparse_num_pts = [int(0.5 * N_per_inst), int(0.5 * N_per_inst)]
        else:
            raise RuntimeError("unsupported")

        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots

        dense_ee_pcd_ls = []
        ee_pcd_ls = []
        robot_pcd_ls = []
        tool_pcd_ls = []
        for rob_i in range(num_bots):
            tool_name = tool_names[rob_i]
            # compute robot pcd
            dense_ee_pcd = teleop_robot.compute_robot_pcd(
                curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                finger_names,
                dense_num_pts,
                pcd_name=f"dense_ee_pcd_{rob_i}",
            )  # (N, 3)
            ee_pcd = teleop_robot.compute_robot_pcd(
                curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                finger_names,
                sparse_num_pts,
                pcd_name=f"ee_pcd_{rob_i}",
            )
            robot_pcd = teleop_robot.compute_robot_pcd(
                curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))],
                pcd_name=f"robot_pcd_{rob_i}",
            )
            if tool_name is not None:
                tool_pcd = teleop_robot.compute_tool_pcd(
                    curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                    tool_name,
                    N_per_inst,
                    pcd_name=f"tool_pcd_{rob_i}",
                )

            # transform robot pcd to world frame
            robot_base_pose_in_world = (
                robot_base_pose_in_world_seq[t, rob_i]
                if robot_base_pose_in_world_seq is not None
                else None
            )
            dense_ee_pcd = (
                robot_base_pose_in_world
                @ np.concatenate(
                    [dense_ee_pcd, np.ones((dense_ee_pcd.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]
            ee_pcd = (
                robot_base_pose_in_world
                @ np.concatenate([ee_pcd, np.ones((ee_pcd.shape[0], 1))], axis=-1).T
            ).T[:, :3]
            robot_pcd = (
                robot_base_pose_in_world
                @ np.concatenate(
                    [robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]
            if tool_name is not None:
                tool_pcd = (
                    robot_base_pose_in_world
                    @ np.concatenate(
                        [tool_pcd, np.ones((tool_pcd.shape[0], 1))], axis=-1
                    ).T
                ).T[:, :3]

            # save to list
            dense_ee_pcd_ls.append(dense_ee_pcd)
            ee_pcd_ls.append(ee_pcd)
            robot_pcd_ls.append(robot_pcd)
            if tool_name is not None:
                tool_pcd_ls.append(tool_pcd)
        # convert to numpy array
        dense_ee_pcd = np.concatenate(dense_ee_pcd_ls + tool_pcd_ls, axis=0)
        ee_pcd = np.concatenate(ee_pcd_ls + tool_pcd_ls, axis=0)
        robot_pcd = np.concatenate(robot_pcd_ls + tool_pcd_ls, axis=0)

        # post process robot pcd
        ee_pcd_tensor = torch.from_numpy(ee_pcd).to(
            device=fusion.device, dtype=fusion.dtype
        )

        if use_dino or distill_dino:
            ee_eval_res = fusion.eval(ee_pcd_tensor, return_names=["dino_feats"])
            ee_feats = ee_eval_res["dino_feats"]

        if use_seg:
            query_texts = ["crate", "battery"]

            fusion.text_queries_for_inst_mask(
                query_texts,
                # query_texts.tolist(),
                query_thresholds,
                boundaries,
                expected_labels=expected_labels,
                robot_pcd=dense_ee_pcd,
                merge_iou=0.05,
            )
            obj_idx_list, obj_idx_dict = fusion.get_mask_idx_dict()

            # select objects with attention
            attention_obj = ["crate", "battery"]
            query_obj_idx = list()

            for item in attention_obj:
                if item not in obj_idx_dict:
                    continue
                query_obj_idx += obj_idx_dict[item]

            obj_pcd = fusion.extract_masked_pcd(query_obj_idx, boundaries=boundaries)
            src_feat_list, src_pts_list, _, label_list = (
                fusion.select_features_from_pcd(
                    obj_pcd,
                    N_per_inst,
                    per_instance=True,
                    use_seg=use_seg,
                    use_dino=(use_dino or distill_dino),
                )
            )
            if "battery" not in label_list or "crate" not in label_list:
                if t == 0:
                    raise
            # box_pts = np.loadtxt("/root/lyt/branch_src/pts_box.txt")
            # src_feat_list.append(None)
            # src_pts_list.append(box_pts)
            # label_list.append("box")

            # np.loadtxt("/root/lyt/branch_src/pts_battery.txt")

            # init_position, src_pts_list = add_addition_channel_pack(src_pts_list, init_position, t, src_feat_list, label_list, goal_position)

            # for idx, pcd in enumerate(src_pts_list):
            #     np.savetxt(f"/home/sim/general_dp-neo-attention_map/branch_src/pack_{idx}",pcd)

        else:
            obj_pcd = fusion.extract_pcd_in_box(
                boundaries=boundaries,
                downsample=True,
                downsample_r=0.002,
                excluded_pts=robot_pcd,
                exclude_threshold=exclude_threshold,
                exclude_colors=exclude_colors,
            )
            src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(
                obj_pcd,
                N_total - ee_pcd.shape[0],
                per_instance=True,
                use_seg=use_seg,
                use_dino=(use_dino or distill_dino),
            )

        aggr_src_pts = np.concatenate(src_pts_list, axis=0)
        aggr_feats = None

        # max_obj_pts_num = max_pts_num - ee_pcd.shape[0]
        # if aggr_src_pts.shape[0] > max_obj_pts_num:
        #     aggr_src_pts = aggr_src_pts[:max_obj_pts_num]
        # elif aggr_src_pts.shape[0] < max_obj_pts_num:
        #     aggr_src_pts = np.pad(
        #         aggr_src_pts,
        #         ((0, max_obj_pts_num - aggr_src_pts.shape[0]), (0, 0)),
        #         mode="constant",
        #     )

        addition_channel = np.zeros([ee_pcd.shape[0], 1])
        ee_pcd = np.concatenate([ee_pcd, addition_channel], axis=-1)
        addition_channel = np.zeros([aggr_src_pts.shape[0], 1])
        aggr_src_pts = np.concatenate([aggr_src_pts, addition_channel], axis=-1)
        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0)

        if aggr_src_pts.shape[0] > max_pts_num:
            aggr_src_pts = aggr_src_pts[:max_pts_num]
        elif aggr_src_pts.shape[0] < max_pts_num:
            aggr_src_pts = np.pad(
                aggr_src_pts,
                ((0, max_pts_num - aggr_src_pts.shape[0]), (0, 0)),
                mode="constant",
            )

        if reference_frame == "world":
            pass
        elif reference_frame == "robot":
            aggr_src_pts = (
                np.linalg.inv(robot_base_pose_in_world_seq[t, 0])
                @ np.concatenate(
                    [aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]

        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        aggr_feats_ls.append(aggr_feats)
    return aggr_src_pts_ls, aggr_feats_ls, None, None

def d3fields_proc_2d(
    fusion: Fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    expected_labels=None,
    tool_names=[None],
    attention_config=None,
    exclude_threshold=0.02,
    prompt_info=None,
    episode_idx = 0,

):
    init_name = shape_meta["info"]["init_name"]
    tgt_name = shape_meta["info"]["tgt_name"]
    task_name = shape_meta["info"]["task_name"]
    cluster_threshold = shape_meta["info"]["cluster_threshold"]
    
    keys = attention_config.keys()
    configuration_dict = {}
    for key in keys:
        configuration_dict[key] = attention_config[key][:]
    if task_name == "hang_mug":
        configuration_dict["tgt_layout"][-1] = np.array([0.0217115, 0.11065497, 0.26605847])
    
    ground_truth_dict = {}
    ground_truth_dict[init_name] = configuration_dict["init_layout"][int(configuration_dict["init"][0])]
    ground_truth_dict[tgt_name] = configuration_dict["tgt_layout"][int(configuration_dict["tgt"][0])]

    boundaries = shape_meta["info"]["boundaries"]
    use_attn = shape_meta["info"]["use_attn"]
    N_per_inst = shape_meta["info"]["N_per_inst"]
    max_pts_num = shape_meta["shape"][1]
    reference_frame = shape_meta["info"]["reference_frame"]

    num_bots = 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    T, V, H, W, C = color_seq.shape
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_src_pts_color_ls = []

    dynamics_dict = {"static": ["bg", tgt_name], "movable": [init_name]}

    masked_rgb_list = []


    for t in range(T):
        rgb = color_seq[t]
        obs = {
            "color": rgb,
            "depth": depth_seq[t],
            "pose": extri_seq[t][:, :3, :],
            "K": intri_seq[t],
        }

        fusion.update(
            obs,
            update_dino = use_attn and (not fusion.xmem_first_mask_loaded),
        )

        generated_code = None
        attn_centroid_dict = None
        attn_dict = None

        if (not fusion.xmem_first_mask_loaded) and use_attn:
            url = local_image_to_data_url(color_seq[0, 3])


            vis = Vision(
                fusion=fusion,
                boundaries=None,
                query_threshold=None,
                N_per_inst=None,
                dynamics_dict=dynamics_dict,
            )

            aggr_pcd, _ = aggr_point_cloud_from_data(
                color_seq[0],
                depth_seq[0],
                intri_seq[0],
                extri_seq[0],
                downsample=True,
                downsample_r=0.002,
                boundaries=shape_meta["info"]["boundaries"],
                out_o3d=False,
            )

            aggr_feat_list, aggr_pcd_list, _, _ = fusion.select_features_from_pcd(
                aggr_pcd,
                18000,
                per_instance=True,
                use_seg=False,
                use_dino=use_attn,
            )
            aggr_pcd = aggr_pcd_list[0]
            aggr_feat = aggr_feat_list[0]

            src_dict = {
                "img": rgb,
                "intrinsic": intri_seq[0],
                "extrinsic": extri_seq[0],
            }
            src_dict["pts"] = aggr_pcd
            src_dict["feat"] = aggr_feat.cpu().numpy()


            def get_ref_lib_path(current_file_path):
                ref_lib_path = os.path.join(current_file_path, "../../../../ref_lib")
                return os.path.abspath(ref_lib_path)

            current_file_path = os.path.dirname(os.path.abspath(__file__))
            lib_root = get_ref_lib_path(current_file_path)
            lib_path = os.path.join(lib_root, task_name)

            ref_dict = load_ref(lib_path, [init_name, tgt_name])

            vis.update(src_dict, th = cluster_threshold)
            vis.semantic_cluster(ref_dict, False)
            vis.bounding_box(src_dict)


            
            if prompt_info is not None:

                task = prompt_info["task"]
                obj_list = prompt_info["obj_list"]
                prompt = prompt_info["prompt"]

                curr_dir = os.path.dirname(os.path.abspath(__file__))
                prompt_log_path = os.path.join(curr_dir, "prompt_log.py")
                attention = Attention(
                    vis=vis, feat_dict=None, obj_list=obj_list, out_path=prompt_log_path
                )
                attn_dict, generated_code = attention.compose(
                    task=task, obj_list=obj_list, instruction=prompt, url=url
                )

                attn_centroid_dict = {}
                for key in attn_dict.keys():
                    attn_centroid_dict[key] = np.mean(attn_dict[key],axis = 0)


            else:
                attn_dict = vis.get_attn_for_training(ground_truth_dict)
            

            bbox_list, attn_list = vis.decouple_into_bbox(attn_dict)

            for key in vis.attn_group.keys():
                if key in vis.dynamics_dict["static"] and key in attn_dict.keys():
                    tgt_pcd = attn_dict[key]
                    fix_obs = np.concatenate(
                        [tgt_pcd, np.ones([tgt_pcd.shape[0], 1])], axis=-1
                    )

            fusion.static_pts = fix_obs
            fusion.init_bbox = bbox_list
            fusion.attn_list = attn_list

        _, ee_pcd, robot_pcd = generate_robot_pcd(
            robot_base_pose_in_world_seq[t],
            teleop_robot,
            qpos_seq[t],
            num_bots,
            400,
        )

        fusion.exclude_camera = None

        fusion.bbox_for_inst_mask(
            fusion.init_bbox,
            boundaries,
            expected_labels=expected_labels,
            voxel_size=0.012,
            merge_iou=0.005,
            pause=False,
        )

        
        obj_idx_list, obj_idx_dict = fusion.get_mask_idx_dict()

        # select objects with attention
        attention_obj_idx = []
        attention_obj_name = []
        query_obj_idx = []
        for idx, item in enumerate(
            [f"inst_{i}" for i in range(len(fusion.attn_list))]
        ):
            if fusion.attn_list[idx] == 1:
                attention_obj_idx += obj_idx_dict[item]
                attention_obj_name += [item]
            query_obj_idx += obj_idx_dict[item]

        inst_mask = fusion.get_mask(attention_obj_name[0])
        env_mask = fusion.project_pcd_to_mask(fusion.static_pts[:,:3], extri_seq[t][:, :3, :], intri_seq[t])
        ee_mask = fusion.project_pcd_to_mask(ee_pcd, extri_seq[t][:, :3, :], intri_seq[t])

        mask = np.maximum(np.maximum(env_mask.astype(np.uint8), inst_mask.astype(np.uint8)), ee_mask.astype(np.uint8))
        masked_rgb = np.concatenate([rgb*mask],axis=-1).astype(np.uint8)
        masked_rgb_list.append(masked_rgb[None, ...])

    agg_masked_rgb = np.concatenate(masked_rgb_list, axis = 0)
    return agg_masked_rgb.astype(np.uint8), generated_code, attn_centroid_dict, attn_dict


# basically the same as d3fields_proc, but to keep the original code clean, we create a new function
def d3fields_proc_for_vis(
    fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    exclude_threshold=0.01,
    exclude_colors=[],
    return_raw_feats=True,
):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    boundaries = shape_meta["info"]["boundaries"]
    use_dino = (
        shape_meta["info"]["use_dino"] if "use_dino" in shape_meta["info"] else False
    )
    distill_dino = (
        shape_meta["info"]["distill_dino"]
        if "distill_dino" in shape_meta["info"]
        else False
    )
    distill_obj = (
        shape_meta["info"]["distill_obj"]
        if "distill_obj" in shape_meta["info"]
        else False
    )
    N_total = shape_meta["shape"][1]

    resize_ratio = shape_meta["info"]["resize_ratio"]
    reference_frame = (
        shape_meta["info"]["reference_frame"]
        if "reference_frame" in shape_meta["info"]
        else "world"
    )

    num_bots = (
        robot_base_pose_in_world_seq.shape[1]
        if len(robot_base_pose_in_world_seq.shape) == 4
        else 1
    )
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)

    new_color_seq = np.zeros(
        (
            color_seq.shape[0],
            color_seq.shape[1],
            resize_H,
            resize_W,
            color_seq.shape[-1],
        ),
        dtype=np.uint8,
    )
    new_depth_seq = np.zeros(
        (depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32
    )
    new_intri_seq = np.zeros(
        (intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32
    )
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t, v] = cv2.resize(
                color_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_depth_seq[t, v] = cv2.resize(
                depth_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_intri_seq[t, v] = intri_seq[t, v] * resize_ratio
            new_intri_seq[t, v, 2, 2] = 1.0
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_raw_feats_ls = []
    rob_mesh_ls = []
    # for t in tqdm(range(T), desc=f'Computing D3Fields'):
    for t in range(T):
        obs = {
            "color": color_seq[t],
            "depth": depth_seq[t],
            "pose": extri_seq[t][:, :3, :],
            "K": intri_seq[t],
        }

        fusion.update(obs, update_dino=(use_dino or distill_dino))

        if teleop_robot is not None:
            # compute robot pcd
            curr_qpos = qpos_seq[t]
            qpos_dim = curr_qpos.shape[0] // num_bots

            robot_pcd_ls = []
            robot_meshes_ls = []
            for rob_i in range(num_bots):
                # compute robot pcd
                robot_pcd = teleop_robot.compute_robot_pcd(
                    curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                    num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))],
                    pcd_name=f"robot_pcd_{rob_i}",
                )
                robot_meshes = teleop_robot.gen_robot_meshes(
                    curr_qpos[qpos_dim * rob_i : qpos_dim * (rob_i + 1)],
                    link_names=[
                        "vx300s/left_finger_link",
                        "vx300s/right_finger_link",
                        "vx300s/gripper_bar_link",
                        "vx300s/gripper_prop_link",
                        "vx300s/gripper_link",
                    ],
                )

                # transform robot pcd to world frame
                robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i]
                robot_pcd = (
                    robot_base_pose_in_world
                    @ np.concatenate(
                        [robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1
                    ).T
                ).T[:, :3]
                # transform mesh to the frame of first robot
                for mesh in robot_meshes:
                    first_robot_base_pose_in_world = robot_base_pose_in_world_seq[
                        t, 0
                    ]  # (4, 4)
                    mesh.transform(
                        np.linalg.inv(first_robot_base_pose_in_world)
                        @ robot_base_pose_in_world
                    )
                robot_meshes_ls = robot_meshes_ls + robot_meshes

                # save to list
                robot_pcd_ls.append(robot_pcd)
            # convert to numpy array
            robot_pcd = np.concatenate(robot_pcd_ls, axis=0)

            obj_pcd = fusion.extract_pcd_in_box(
                boundaries=boundaries,
                downsample=True,
                downsample_r=0.002,
                excluded_pts=robot_pcd,
                exclude_threshold=exclude_threshold,
                exclude_colors=exclude_colors,
            )
        else:
            obj_pcd = fusion.extract_pcd_in_box(
                boundaries=boundaries,
                downsample=True,
                downsample_r=0.002,
                exclude_colors=exclude_colors,
            )
        # res = fusion.eval(obj_pcd)
        # src_pts_list = [obj_pcd]
        # src_feat_list = [res['dino_feats']]
        src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(
            obj_pcd,
            N_total,
            per_instance=True,
            use_seg=False,
            use_dino=(use_dino or distill_dino),
        )

        aggr_src_pts = np.concatenate(src_pts_list, axis=0)  # (N, 3)
        aggr_feats = (
            torch.concat(src_feat_list, axis=0).detach().cpu().numpy()
            if (use_dino or distill_dino)
            else None
        )  # (N, 1024)

        if return_raw_feats:
            if aggr_feats is not None:
                aggr_raw_feats = aggr_feats.copy()
                aggr_raw_feats_ls.append(aggr_raw_feats)

        if distill_dino:
            aggr_feats = (
                fusion.eval_dist_to_sel_feats(
                    torch.concat(src_feat_list, axis=0),
                    obj_name=distill_obj,
                )
                .detach()
                .cpu()
                .numpy()
            )

        # transform to reference frame
        if reference_frame == "world":
            pass
        elif reference_frame == "robot":
            # aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            aggr_src_pts = (
                np.linalg.inv(robot_base_pose_in_world_seq[t, 0])
                @ np.concatenate(
                    [aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1
                ).T
            ).T[:, :3]

        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        if use_dino or distill_dino:
            aggr_feats_ls.append(aggr_feats.astype(np.float32))
        if teleop_robot is not None:
            rob_mesh_ls.append(robot_meshes_ls)

    if return_raw_feats:
        return aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feats_ls, rob_mesh_ls
    return aggr_src_pts_ls, aggr_feats_ls, rob_mesh_ls


# basically the same as d3fields_proc, but to keep the original code clean, we create a new function
def d3fields_proc_for_tsne(
    fusion,
    shape_meta,
    color_seq,
    depth_seq,
    extri_seq,
    intri_seq,
    robot_base_pose_in_world_seq=None,
    teleop_robot=None,
    qpos_seq=None,
    exclude_threshold=0.01,
    exclude_colors=[],
    return_raw_feats=True,
):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    boundaries = shape_meta["info"]["boundaries"]
    use_dino = (
        shape_meta["info"]["use_dino"] if "use_dino" in shape_meta["info"] else False
    )
    distill_dino = (
        shape_meta["info"]["distill_dino"]
        if "distill_dino" in shape_meta["info"]
        else False
    )
    distill_obj = (
        shape_meta["info"]["distill_obj"]
        if "distill_obj" in shape_meta["info"]
        else False
    )
    N_total = shape_meta["shape"][1]

    resize_ratio = shape_meta["info"]["resize_ratio"]
    reference_frame = (
        shape_meta["info"]["reference_frame"]
        if "reference_frame" in shape_meta["info"]
        else "world"
    )

    num_bots = (
        robot_base_pose_in_world_seq.shape[1]
        if len(robot_base_pose_in_world_seq.shape) == 4
        else 1
    )
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(
        robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4
    )

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)

    new_color_seq = np.zeros(
        (
            color_seq.shape[0],
            color_seq.shape[1],
            resize_H,
            resize_W,
            color_seq.shape[-1],
        ),
        dtype=np.uint8,
    )
    new_depth_seq = np.zeros(
        (depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32
    )
    new_intri_seq = np.zeros(
        (intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32
    )
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t, v] = cv2.resize(
                color_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_depth_seq[t, v] = cv2.resize(
                depth_seq[t, v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST
            )
            new_intri_seq[t, v] = intri_seq[t, v] * resize_ratio
            new_intri_seq[t, v, 2, 2] = 1.0
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_raw_feats_ls = []
    rob_mesh_ls = []
    # for t in tqdm(range(T), desc=f'Computing D3Fields'):
    for t in range(T):
        # print()
        obs = {
            "color": color_seq[t],
            "depth": depth_seq[t],
            "pose": extri_seq[t][:, :3, :],
            "K": intri_seq[t],
        }

        fusion.update(obs, update_dino=(use_dino or distill_dino))

        grid, _ = create_init_grid(boundaries, step_size=0.005)  # (N, 3)
        grid = grid.to(device=fusion.device, dtype=fusion.dtype)
        # res = fusion.eval(obj_pcd)
        # src_pts_list = [obj_pcd]
        # src_feat_list = [res['dino_feats']]
        # with torch.no_grad():
        #     grid_eval_res = fusion.batch_eval(grid, return_names=['dino_feats'])

        # grid_feats = grid_eval_res['dino_feats']

        src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(
            grid.detach().cpu().numpy(),
            -1,
            per_instance=False,
            use_seg=False,
            use_dino=(use_dino or distill_dino),
        )
        aggr_src_pts = np.concatenate(src_pts_list, axis=0)  # (N, 3)
        aggr_feats = (
            torch.concat(src_feat_list, axis=0).detach().cpu().numpy()
            if (use_dino or distill_dino)
            else None
        )  # (N, 1024)

        if return_raw_feats:
            if aggr_feats is not None:
                aggr_raw_feats = aggr_feats.copy()
                aggr_raw_feats_ls.append(aggr_raw_feats)

        if distill_dino:
            aggr_feats = (
                fusion.eval_dist_to_sel_feats(
                    torch.concat(src_feat_list, axis=0),
                    obj_name=distill_obj,
                )
                .detach()
                .cpu()
                .numpy()
            )

        # # transform to reference frame
        # if reference_frame == 'world':
        #     pass
        # elif reference_frame == 'robot':
        #     # aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
        #     aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]

        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        if use_dino or distill_dino:
            aggr_feats_ls.append(aggr_feats.astype(np.float32))

    if return_raw_feats:
        return aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feats_ls, rob_mesh_ls
    return aggr_src_pts_ls, aggr_feats_ls, rob_mesh_ls


def vis_actions(raw_actions):
    # (T, 7)
    import open3d as o3d
    import transforms3d

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    # add a random mesh
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
    action_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    action_mesh_pts = np.asarray(action_mesh.vertices).copy()
    visualizer.add_geometry(action_mesh)
    for t in range(raw_actions.shape[0]):
        pos = raw_actions[t, :3]
        rot_euler = raw_actions[t, 3:6]
        rot_mat = transforms3d.euler.euler2mat(rot_euler[0], rot_euler[1], rot_euler[2])
        if t == 0:
            print("rot_mat: ", rot_mat)
        action_mesh_pts_new = rot_mat @ action_mesh_pts.T + pos.reshape(-1, 1)
        action_mesh_pts_new = action_mesh_pts_new.T
        action_mesh.vertices = o3d.utility.Vector3dVector(action_mesh_pts_new)
        visualizer.update_geometry(action_mesh)
        visualizer.poll_events()
        visualizer.update_renderer()
        if t == 0:
            visualizer.run()
        time.sleep(0.03)
    visualizer.destroy_window()


def vis_post_actions(actions):
    # (T, 10)
    import open3d as o3d
    import pytorch3d
    import torch

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    # add a random mesh
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
    action_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    action_mesh_pts = np.asarray(action_mesh.vertices).copy()
    visualizer.add_geometry(action_mesh)
    for t in range(actions.shape[0]):
        pos = actions[t, :3]
        rot_6d = actions[t, 3:9]
        rot_mat = (
            pytorch3d.transforms.rotation_6d_to_matrix(
                torch.from_numpy(rot_6d).unsqueeze(0)
            )
            .squeeze(0)
            .numpy()
        )
        if t == 0:
            print("rot_6d: ", rot_6d)
            print("rot_mat: ", rot_mat)
        action_mesh_pts_new = rot_mat @ action_mesh_pts.T + pos.reshape(-1, 1)
        action_mesh_pts_new = action_mesh_pts_new.T
        action_mesh.vertices = o3d.utility.Vector3dVector(action_mesh_pts_new)
        visualizer.update_geometry(action_mesh)
        visualizer.poll_events()
        visualizer.update_renderer()
        if t == 0:
            visualizer.run()
        time.sleep(0.03)
    visualizer.destroy_window()


def _convert_actions(raw_actions, rotation_transformer, action_key):
    act_num, act_dim = raw_actions.shape
    is_bimanual = act_dim == 14
    # vis_actions(raw_actions[:,:7])
    # vis_actions(raw_actions[:,7:])
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)

    if action_key == "cartesian_action" or action_key == "observations/ee_pos":
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    elif action_key == "joint_action":
        pass
    elif action_key == "motion_prim":
        pass
    elif action_key == "next_waypoint":
        pass
    else:
        raise RuntimeError("unsupported action_key")
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    # vis_post_actions(actions[:,10:])
    return actions
