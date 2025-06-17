import os
import sys

import cv2
import h5py
import hydra
import numpy as np
import sapien.core as sapien
import transforms3d
from d3fields.utils.text_embedding import from_text_to_embedding
from diffusion_policy.common.kinematics_utils import KinHelper
from instruction_generate.generate_instruction import instruction_generater
from omegaconf import OmegaConf

# from diffusion_policy.common.data_utils import save_dict_to_hdf5
from sapien_env.gui.teleop_gui_trossen import (
    META_CAMERA,
    TABLE_TOP_CAMERAS,
    VIEWER_CAMERA,
    GUIBase,
)
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.utils.my_utils import bcolors

curr_path = os.path.abspath(__file__)
for _ in range(3):
    curr_path = os.path.dirname(curr_path)
sys.path.append(curr_path)


class instruction_recorder:
    def __init__(self, file_path):
        self.file_path = file_path

    def append_to_json(self, new_element):
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file)

        data.append(new_element)

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)


def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic


def transform_action_from_world_to_robot(action: np.ndarray, pose: sapien.Pose):
    # :param action: (7,) np.ndarray in world frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # :param pose: sapien.Pose of the robot base in world frame
    # :return: (7,) np.ndarray in robot frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # transform action from world to robot frame
    action_mat = np.zeros((4, 4))
    action_mat[:3, :3] = transforms3d.euler.euler2mat(action[3], action[4], action[5])
    action_mat[:3, 3] = action[:3]
    action_mat[3, 3] = 1
    action_mat_in_robot = np.matmul(
        np.linalg.inv(pose.to_transformation_matrix()), action_mat
    )
    action_robot = np.zeros(7)
    action_robot[:3] = action_mat_in_robot[:3, 3]
    action_robot[3:6] = transforms3d.euler.mat2euler(
        action_mat_in_robot[:3, :3], axes="sxyz"
    )
    action_robot[6] = action[6]
    return action_robot


def task_to_cfg(
    task,
    manip_obj=None,
    task_level_multimodality=False,
    assign_idx=None,
):
    if task == "hang_mug":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.hang_mug_env.HangMugRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": manip_obj,
                "assign_idx": assign_idx,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.hang_mug_scripted_policy.SingleArmPolicy",
            }
        )
    elif task == "mug_collect":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.mug_collect_env.MugCollectRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": "pepsi" if manip_obj is None else manip_obj,
                "randomness_level": "half",
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.mug_collect_scripted_policy.SingleArmPolicy",
            }
        )
    elif task == "pen_insertion":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.pen_insertion_env.PenInsertionRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": "pencil" if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.pen_insertion_scripted_policy.SingleArmPolicy",
            }
        )
    elif task == "stow":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.stow_env.StowRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": "book_1" if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.stow_scripted_policy.SingleArmPolicy",
            }
        )
    elif task == "circle":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.circle_env.CircleRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": "book_1" if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.circle_scripted_policy.SingleArmPolicy",
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.can_scripted_policy.SingleArmPolicy",
            }
        )
    else:
        raise ValueError(f"Unknown task {task}")
    return cfg, policy_cfg


def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    print(filename)
    with h5py.File(filename, "w") as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, "/", dic, config_dict)


def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
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


def main_env(
    episode_idx,
    dataset_dir,
    headless,
    mode,
    task_name,
    manip_obj,
    vis_info,
    slackness_type,
    task_level_multimodality=False,
    assign_idx=None,
    recorder=None,
):
    # initialize env
    os.system(f"mkdir -p {dataset_dir}")
    kin_helper = KinHelper(robot_name="panda")

    cfg, policy_cfg = task_to_cfg(
        task_name,
        manip_obj=manip_obj,
        task_level_multimodality=task_level_multimodality,
        assign_idx=assign_idx,
    )

    with open(os.path.join(dataset_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f.name)

    # collect data
    env: BaseRLEnv = hydra.utils.instantiate(cfg)
    env.seed(episode_idx)
    env.reset()
    arm_dof = env.arm_dof

    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=headless, resolution=(160, 120))  #
    for _, params in TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.create_camera(**META_CAMERA, meta=True)
    if not gui.headless:
        gui.viewer.set_camera_xyz(**VIEWER_CAMERA["position"])
        gui.viewer.set_camera_rpy(**VIEWER_CAMERA["rotation"])
        gui.viewer.set_fovy(1)
    scene = env.scene
    scene.step()

    dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")

    scripted_policy = hydra.utils.instantiate(policy_cfg)

    # set up data saving hyperparameters
    init_configuration = env.get_layout(assign_idx)

    generater = instruction_generater(
        seed=episode_idx,
        keys=["mug", "branch"],
        template_path="/home/neo/lyt_dp/diffusion/sapien_env/sapien_env/teleop/instruction_generate/template.json",
        slackness_type=slackness_type,
    )
    descriptive_element = generater.from_config_to_string(
        init_configuration, manip_obj, vis_info
    )
    instruction = generater.fill_in_template(descriptive_element)
    print(instruction)
    embedding = from_text_to_embedding(instruction)
    init_configuration["embedding"] = embedding

    # init_poses = env.get_init_poses()
    data_dict = {
        "observations": {
            "joint_pos": [],
            "joint_vel": [],
            "full_joint_pos": [],  # this is to compute FK
            "robot_base_pose_in_world": [],
            "ee_pos": [],
            "ee_vel": [],
            "images": {},
            "meta_images": {},
        },
        "joint_action": [],
        "cartesian_action": [],
        "progress": [],
        "attention_config": {},
    }

    data_dict["attention_config"] = init_configuration

    cams = gui.cams
    for cam in cams:
        data_dict["observations"]["images"][f"{cam.name}_color"] = []
        data_dict["observations"]["images"][f"{cam.name}_depth"] = []
        data_dict["observations"]["images"][f"{cam.name}_intrinsic"] = []
        data_dict["observations"]["images"][f"{cam.name}_extrinsic"] = []

    meta_cam = gui.meta_cam
    data_dict["observations"]["meta_images"]["color"] = []
    data_dict["observations"]["meta_images"]["intrinsic"] = []
    data_dict["observations"]["meta_images"]["extrinsic"] = []

    attr_dict = {
        "sim": True,
    }
    config_dict = {"observations": {"images": {}, "meta_images": {}}}
    for cam_idx, cam in enumerate(gui.cams):
        color_save_kwargs = {
            "chunks": (1, cam.height, cam.width, 3),  # (1, 480, 640, 3)
            "compression": "gzip",
            "compression_opts": 9,
            "dtype": "uint8",
        }
        depth_save_kwargs = {
            "chunks": (1, cam.height, cam.width),  # (1, 480, 640)
            "compression": "gzip",
            "compression_opts": 9,
            "dtype": "uint16",
        }
        config_dict["observations"]["images"][f"{cam.name}_color"] = color_save_kwargs
        config_dict["observations"]["images"][f"{cam.name}_depth"] = depth_save_kwargs
    meta_color_save_kwargs = {
        "chunks": (1, meta_cam.height, meta_cam.width, 3),  # (1, 480 * 2, 640 * 2, 3)
        "compression": "gzip",
        "compression_opts": 9,
        "dtype": "uint8",
    }
    config_dict["observations"]["meta_images"]["color"] = meta_color_save_kwargs

    # env.add_full_traj()
    # rgbs, depths = gui.render(depth=True)
    # img = rgbs[0]
    # print(img,img.shape)

    # img = np.concatenate([img[:,:,2][:,:,None],img[:,:,1][:,:,None],img[:,:,0][:,:,None]],axis = -1)
    # img = Image.fromarray(img)
    # img.save(f"/home/sim/general_dp-neo-attention_map/init_config/output_{episode_idx}.jpg")
    # raise
    while 0:
        action = np.array(
            [
                -2.21402311,
                0.17274794,
                2.23800898,
                -2.27481246,
                -0.16332519,
                2.16096449,
                0.90828639,
                0.09,
            ]
        )
        obs, reward, done, _ = env.step(action)
        rgbs, depths = gui.render(depth=True)
        # print("hang")

    index = 0

    while True:
        index += 1
        action = np.zeros(arm_dof + 1)
        cartisen_action, next_waypoint, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose(), mode=mode, assign_idx=assign_idx
        )
        if quit:
            break
        cartisen_action_in_rob = transform_action_from_world_to_robot(
            cartisen_action, env.robot.get_pose()
        )
        action[:arm_dof] = kin_helper.compute_ik_sapien(
            env.robot.get_qpos()[:], cartisen_action_in_rob
        )[:arm_dof]
        action[arm_dof:] = cartisen_action_in_rob[6]

        obs, reward, done, _ = env.step(action[: arm_dof + 1])

        rgbs, depths = gui.render(depth=True)

        data_dict["observations"]["joint_pos"].append(env.robot.get_qpos()[:-1])
        data_dict["observations"]["joint_vel"].append(env.robot.get_qvel()[:-1])
        data_dict["observations"]["full_joint_pos"].append(env.robot.get_qpos())
        data_dict["observations"]["robot_base_pose_in_world"].append(
            env.robot.get_pose().to_transformation_matrix()
        )

        ee_translation = env.palm_link.get_pose().p
        ee_rotation = transforms3d.euler.quat2euler(
            env.palm_link.get_pose().q, axes="sxyz"
        )
        ee_gripper = env.robot.get_qpos()[arm_dof]
        ee_pos = np.concatenate([ee_translation, ee_rotation, [ee_gripper]])
        ee_vel = np.concatenate(
            [
                env.palm_link.get_velocity(),
                env.palm_link.get_angular_velocity(),
                env.robot.get_qvel()[arm_dof : arm_dof + 1],
            ]
        )

        data_dict["observations"]["ee_pos"].append(ee_pos)
        data_dict["observations"]["ee_vel"].append(ee_vel)

        data_dict["joint_action"].append(action.copy())
        data_dict["cartesian_action"].append(cartisen_action.copy())

        data_dict["progress"].append(scripted_policy.progress)
        for cam_idx, cam in enumerate(gui.cams):
            data_dict["observations"]["images"][f"{cam.name}_color"].append(
                rgbs[cam_idx]
            )
            data_dict["observations"]["images"][f"{cam.name}_depth"].append(
                depths[cam_idx]
            )
            data_dict["observations"]["images"][f"{cam.name}_intrinsic"].append(
                cam.get_intrinsic_matrix()
            )
            data_dict["observations"]["images"][f"{cam.name}_extrinsic"].append(
                cam.get_extrinsic_matrix()
            )

        meta_image = gui.take_meta_view()
        data_dict["observations"]["meta_images"]["color"].append(meta_image)
        data_dict["observations"]["meta_images"]["intrinsic"].append(
            meta_cam.get_intrinsic_matrix()
        )
        data_dict["observations"]["meta_images"]["extrinsic"].append(
            meta_cam.get_extrinsic_matrix()
        )

    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
    emoji = "✅" if reward == 1 else "❌"
    text = "Success :)" if reward == 1 else "Failed :("
    print(f"{txt_color}{emoji} {text}{bcolors.ENDC}")
    if reward < 1:
        if not gui.headless:
            gui.viewer.close()
            cv2.destroyAllWindows()
        env.close()
        # raise Exception("Failed at episode {}".format(episode_idx))
    else:
        data_dict = stack_dict(data_dict)
        recorder.append_to_json({str(episode_idx): instruction})
        save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)
        if not gui.headless:
            gui.viewer.close()
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    import argparse
    import random
    from pathlib import Path

    obj_pool = ["nescafe_mug_1", "nescafe_mug_2", "nescafe_mug_3", "nescafe_mug_4"]
    obj_color_pool = ["red mug", "white mug", "blue mug", "green mug"]

    parser = argparse.ArgumentParser(description="sum the integers at the command line")
    parser.add_argument("--start_idx", default=0)
    parser.add_argument("--end_idx", default=360)
    parser.add_argument("--dataset_dir", default="/home/neo/lyt_dp/data/mug_test")
    parser.add_argument("--mode", default="straight")
    parser.add_argument("--task_name", default="hang_mug")
    parser.add_argument("--headless", default=False)
    parser.add_argument("--obj_name", default="nescafe_mug_1")
    parser.add_argument("--extra_obj_name", default=["nescafe_mug_4", "nescafe_mug_2"])
    parser.add_argument("--other", default=2, type=int)
    # parser.add_argument("--output_type", default="pose")
    parser.add_argument("--task_level_multimodality", default=False)
    args = parser.parse_args()

    save_repo_path = "sapien_env"
    save_repo_dir = os.path.join(args.dataset_dir, save_repo_path)
    os.system(f"mkdir -p {save_repo_dir}")

    file_path = os.path.join(args.dataset_dir, "instructions.json")

    initial_data = ["begin to record"]

    import json

    with open(file_path, "w") as json_file:
        json.dump(initial_data, json_file, indent=4)

    recorder = instruction_recorder(file_path=file_path)

    curr_repo_dir = Path(__file__).parent.parent
    ignore_list = [".git", "__pycache__", "data"]
    for sub_dir in os.listdir(curr_repo_dir):
        if sub_dir not in ignore_list:
            os.system(f"cp -r {os.path.join(curr_repo_dir, sub_dir)} {save_repo_dir}")

    for exec_idx in range(args.start_idx, args.end_idx):
        random.seed(exec_idx)
        mug_num = np.random.randint(1, 4)

        select_index = random.sample(range(len(obj_pool)), k=2)
        manip_obj = [obj_pool[select_index[0]], obj_pool[select_index[1]]]
        obj_color = [obj_color_pool[select_index[0]], obj_color_pool[select_index[1]]]

        vis_info = {
            "mug": {
                manip_obj[0]: [obj_color[0], obj_color[1]],
                manip_obj[1]: [obj_color[1], obj_color[0]],
            },
            "branch": {
                "0": ["right-topmost branch", "left-topmost branch"],
                "1": ["left-topmost branch", "right-topmost branch"],
                "2": ["middle-right branch", "middle-left branch"],
            },
        }

        mug_idx = 0
        branch_idx = exec_idx % 3

        assign_idx = [mug_idx, branch_idx]
        slackness_type = random.choice(
            ["no_slackness", "mug_slackness", "branch_slackness", "both_slackness"]
        )

        main_env(
            episode_idx=exec_idx,
            dataset_dir=args.dataset_dir,
            headless=args.headless,
            mode=args.mode,
            task_name=args.task_name,
            manip_obj=manip_obj,
            vis_info=vis_info,
            slackness_type=slackness_type,
            # output_type=args.output_type,
            task_level_multimodality=args.task_level_multimodality,
            assign_idx=assign_idx,
            recorder=recorder,
        )
