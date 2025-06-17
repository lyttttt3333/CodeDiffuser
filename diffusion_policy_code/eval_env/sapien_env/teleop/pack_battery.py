import os
import sys
from typing import Optional

import cv2
import hydra
import numpy as np
import transforms3d
from diffusion_policy.common.data_utils import save_dict_to_hdf5
from diffusion_policy.common.kinematics_utils import KinHelper
from omegaconf import OmegaConf
from pathlib import Path

from sapien_env.gui.teleop_gui_trossen import (
    META_CAMERA,
    TABLE_TOP_CAMERAS,
    VIEWER_CAMERA,
    GUIBase,
)
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.utils.my_utils import bcolors
from sapien_env.utils.pose_utils import transform_action_from_world_to_robot

curr_path = os.path.abspath(__file__)
for _ in range(3):
    curr_path = os.path.dirname(curr_path)
sys.path.append(curr_path)


def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic


def task_to_cfg(
    task: str,
    seed: Optional[int] = None,
    manip_obj=None,
    extra_manip_obj=None,
    task_level_multimodality: bool = False,
    stand_mode: bool = False,
    simple_mode: bool = False,
    fix_pick: bool = False,
    num_obj_done: Optional[int] = None,
    num_obj_wait: Optional[int] = None,
    num_slot_wait: Optional[int] = None,
    assign_num: Optional[bool] = None,
):
    if task == "hang_mug":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.hang_mug_env.HangMugRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": "nescafe_mug" if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.hang_mug_scripted_policy.SingleArmPolicy",
                "seed": seed,
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
                "seed": seed,
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
                "seed": seed,
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
                "seed": seed,
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
                "seed": seed,
            }
        )
    elif task == "can":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.can_env.CanRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "use_visual_obs": False,
                "manip_obj": "cola" if manip_obj is None else manip_obj,
                "extra_manip_obj": (
                    "pepsi" if extra_manip_obj is None else extra_manip_obj
                ),
                "randomness_level": "full",
                "task_level_multimodality": task_level_multimodality,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.can_scripted_policy.SingleArmPolicy",
                "seed": seed,
            }
        )
    elif task == "pack":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.pack_env.PackRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "task_level_multimodality": False,
                "nine_pos_mode": False,
                "stand_mode": stand_mode,
                "simple_mode": simple_mode,
                "fix_pick": fix_pick,
                "num_obj_done": num_obj_done,
                "num_obj_wait": num_obj_wait,
                "num_slot_wait": num_slot_wait,
                "seed": seed,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.pack_scripted_policy.SingleArmPolicy",
                "seed": seed,
            }
        )
    elif task == "insert_soda":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.insert_soda_env.InsertSodaRLEnv",
                "use_gui": True,
                "robot_name": "panda",
                "frame_skip": 10,
                "task_level_multimodality": False,
                "stand_mode": stand_mode,
                "simple_mode": simple_mode,
                "fix_pick": fix_pick,
                "num_obj_wait": 1,
                "num_obj_done": num_obj_done,
                "seed": seed,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.insert_soda_scripted_policy.SingleArmPolicy",
                "seed": seed,
            }
        )
    elif task == "pack_battery":
            cfg = OmegaConf.create(
                {
                    "_target_": "sapien_env.rl_env.pack_battery_env.PackBatteryRLEnv",
                    "use_gui": True,
                    "robot_name": "panda",
                    "frame_skip": 10,
                    "task_level_multimodality": True,
                    "stand_mode": stand_mode,
                    "simple_mode": simple_mode,
                    "fix_pick": fix_pick,
                    "num_obj_wait": num_obj_wait,
                    "num_slot_wait": num_slot_wait,
                    "num_obj_done": num_obj_done,
                    "assign_num": assign_num,
                    "seed": seed,
                }
            )
            policy_cfg = OmegaConf.create(
                {
                    "_target_": "sapien_env.teleop.pack_battery_scripted_policy.SingleArmPolicy",
                    "seed": seed,
                }
            )
    else:
        raise ValueError(f"Unknown task {task}")
    return cfg, policy_cfg


def main_env(
    episode_idx,
    seed,
    dataset_dir,
    headless,
    task_name,
    manip_obj,
    extra_manip_obj=None,
    task_level_multimodality: bool = False,
    stand_mode: bool = False,
    simple_mode: bool = False,
    fix_pick: bool = False,
    num_obj_done: Optional[int] = None,
    num_obj_wait: Optional[int] = None,
    num_slot_wait: Optional[int] = None,
    assign_num: Optional[bool] = None,
):
    # initialize env
    os.system(f"mkdir -p {dataset_dir}")
    kin_helper = KinHelper(robot_name="panda")

    cfg, policy_cfg = task_to_cfg(
        task_name,
        seed=seed,
        manip_obj=manip_obj,
        extra_manip_obj=extra_manip_obj,
        task_level_multimodality=task_level_multimodality,
        stand_mode=stand_mode,
        simple_mode=simple_mode,
        fix_pick=fix_pick,
        num_obj_done=num_obj_done,
        num_obj_wait=num_obj_wait,
        num_slot_wait=num_slot_wait,
        assign_num=assign_num,
    )

    with open(os.path.join(dataset_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f.name)

    # collect data
    env: BaseRLEnv = hydra.utils.instantiate(cfg)
    scripted_policy = hydra.utils.instantiate(policy_cfg)

    env.reset()
    arm_dof = env.arm_dof

    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=headless, resolution=(160, 120))
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

    # set up data saving hyperparameters
    init_poses = env.get_init_poses()
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
        "info": {"init_poses": init_poses},
        "attention_config":{},
        "env_config":{},
    }

    cams = gui.cams
    for cam in cams:
        data_dict["observations"]["images"][f"{cam.name}_color"] = []
        data_dict["observations"]["images"][f"{cam.name}_depth"] = []
        data_dict["observations"]["images"][f"{cam.name}_intrinsic"] = []
        data_dict["observations"]["images"][f"{cam.name}_extrinsic"] = []

    # meta_cam = gui.meta_cam
    # data_dict["observations"]["meta_images"]["color"] = []
    # data_dict["observations"]["meta_images"]["intrinsic"] = []
    # data_dict["observations"]["meta_images"]["extrinsic"] = []

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

    # meta_color_save_kwargs = {
    #     "chunks": (1, meta_cam.height, meta_cam.width, 3),  # (1, 480 * 2, 640 * 2, 3)
    #     "compression": "gzip",
    #     "compression_opts": 9,
    #     "dtype": "uint8",
    # }
    # config_dict["observations"]["meta_images"]["color"] = meta_color_save_kwargs

    rand_img_path = os.path.join(dataset_dir, "randomness")
    if not os.path.exists(rand_img_path):
        os.makedirs(rand_img_path)
    gui.render()
    picture = gui.take_meta_view()
    picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(rand_img_path, f"{episode_idx}.png"), picture)


    # import time
    # time.sleep(5)
    while True:
        action = np.zeros(arm_dof + 1)
        cartisen_action, next_waypoint, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose()
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

        # meta_image = gui.take_meta_view()
        # data_dict["observations"]["meta_images"]["color"].append(meta_image)
        # data_dict["observations"]["meta_images"]["intrinsic"].append(
        #     meta_cam.get_intrinsic_matrix()
        # )
        # data_dict["observations"]["meta_images"]["extrinsic"].append(
        #     meta_cam.get_extrinsic_matrix()
        # )

    data_dict["attention_config"]=scripted_policy.attention_config
    data_dict["env_config"]=env.get_layout()
    print(env.get_layout())


    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
    emoji = "✅" if reward == 1 else "❌"
    text = "Success :)" if reward == 1 else "Failed :("
    print(f"{txt_color}{emoji} {text}{bcolors.ENDC}")

    if not gui.headless:
        gui.viewer.close()
        cv2.destroyAllWindows()
    env.close()

    if reward < 1:
        return False
    else:
        data_dict = stack_dict(data_dict)
        save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)
        return True


if __name__ == "__main__":
    import argparse

    from tqdm import tqdm

    wait_num_obj = 4
    wait_num_slot = 4

    # data_img_{wait_num_obj}x{wait_num_slot}

    parser = argparse.ArgumentParser(description="Scripted policy rollout")
    parser.add_argument("--start_idx", type=int,default=0)
    parser.add_argument("--end_idx", type=int,default=180)
    parser.add_argument("--dataset_dir", default=f"/media/yixuan_2T/lyt/img_data/{wait_num_obj}x{wait_num_slot}")
    parser.add_argument("--task_name", default="pack_battery")
    parser.add_argument("--headless", default=True)
    parser.add_argument("--obj_name", default=None)
    parser.add_argument("--extra_obj_name", default=None)
    parser.add_argument("--task_level_multimodality", action="store_true")
    parser.add_argument("--stand_mode", action="store_true")
    parser.add_argument("--simple_mode", action="store_true")
    parser.add_argument("--fix_pick", action="store_true")
    parser.add_argument("--num_obj_done", default=7, type=int)
    parser.add_argument("--num_obj_wait", default=wait_num_obj, type=int)
    parser.add_argument("--num_slot_wait", default=wait_num_slot, type=int)
    parser.add_argument("--assign_num", default=False, type=bool)
    args = parser.parse_args()

    assert args.start_idx < args.end_idx

    save_repo_path = f"sapien_env"
    save_repo_dir = os.path.join(args.dataset_dir, save_repo_path)
    os.system(f"mkdir -p {save_repo_dir}")

    curr_repo_dir = Path(__file__).parent.parent
    ignore_list = [".git", "__pycache__", "data"]
    for sub_dir in os.listdir(curr_repo_dir):
        if sub_dir not in ignore_list:
            os.system(f"cp -r {os.path.join(curr_repo_dir, sub_dir)} {save_repo_dir}")

    for i in tqdm(
        range(args.start_idx, args.end_idx),
        colour="green",
        ascii=" 123456789>",
        desc="Dataset Generation"
    ):
        success = False
        retry = 0
        if True:
            success = main_env(
                episode_idx=i,
                seed=i,
                dataset_dir=args.dataset_dir,
                headless=args.headless,
                task_name=args.task_name,
                manip_obj=args.obj_name,
                extra_manip_obj=args.extra_obj_name,
                task_level_multimodality=args.task_level_multimodality,
                stand_mode=args.stand_mode,
                simple_mode=args.simple_mode,
                fix_pick=args.fix_pick,
                num_obj_done=args.num_obj_done,
                num_obj_wait=args.num_obj_wait,
                num_slot_wait=args.num_slot_wait,
                assign_num=args.assign_num,
            )
            # if not success:
            #     retry += 1
            #     print(f"{bcolors.FAIL}Retry {retry} for episode {i}{bcolors.ENDC}")
            #     if retry > 10:
            #         raise ValueError("Too many retries")