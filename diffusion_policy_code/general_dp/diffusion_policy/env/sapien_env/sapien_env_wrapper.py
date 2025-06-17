import os
from math import inf
from typing import Optional

import cv2
import h5py
import hydra
import numpy as np
import sapien.core as sapien
import transforms3d
from d3fields.fusion import Fusion
from diffusion_policy.common.data_utils import d3fields_proc_infer, d3fields_proc
from diffusion_policy.common.kinematics_utils import KinHelper
from diffusion_policy.common.rob_mesh_utils import load_mesh
from gym import spaces
from omegaconf import OmegaConf
from tqdm import tqdm

from sapien_env.gui.teleop_gui_trossen import TABLE_TOP_CAMERAS, GUIBase
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light


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


# necessary methods and attributes:
# methods:
# - get_observation
# - seed
# - render
# - step
# - reset
# - close
# attributes:
# - action_space
# - observation_space
class SapienEnvWrapper:
    def __init__(
        self,
        env: BaseRLEnv,
        shape_meta: dict,
        init_states: Optional[np.ndarray] = None,
        render_obs_keys=["right_bottom_view"],
        fusion: Optional[Fusion] = None,
        pca_model=None,
        expected_labels=None,
        output_dir=None,
    ):
        # :param init_states: (k,4,4) np.ndarray. The initial pose of the object in world frame
        self.env = env
        self.shape_meta = shape_meta
        self.init_states = init_states
        self.render_obs_keys = render_obs_keys
        self.fusion = fusion
        self.pca_model = pca_model
        self.expected_labels = expected_labels
        self.output_dir = output_dir
        self.is_joint = (
            "key" in self.shape_meta["action"]
            and self.shape_meta["action"]["key"] == "joint_action"
        )

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        observation_space = spaces.Dict()
        self.img_h = 480
        self.img_w = 640
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("color"):
                min_value, max_value = 0, 1
                self.img_h, self.img_w = shape[1:]
            elif key.endswith("depth"):
                min_value, max_value = 0, 1
                self.img_h, self.img_w = shape[1:]
            elif key.endswith("quat"):
                min_value, max_value = -1, 1
            elif key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos"):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith("vel"):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith("d3fields"):
                min_value, max_value = -1, 1
            elif key.endswith("embedding"):
                min_value, max_value = -inf, inf
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

        # setup sapien rendering
        self.gui = GUIBase(
            env.scene, env.renderer, headless=True, resolution=(480, 360)
        )
        for name, params in TABLE_TOP_CAMERAS.items():
            split_name = name.split("_")
            split_name.append("view")
            rejoin_name = "_".join(split_name)
            if rejoin_name in self.render_obs_keys:
                if "rotation" in params:
                    self.gui.create_camera_from_pos_rot(**params)
                else:
                    self.gui.create_camera(**params)

        # setup sapien control
        # self.teleop = TeleopRobot(env.robot_name)
        self.teleop = KinHelper(env.robot_name)

        # load meshes
        self.meshes, self.mesh_offsets = load_mesh(env.robot_name)
        self.pause_reset = False

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = {}
            arm_dof = self.env.arm_dof
            raw_obs["joint_pos"] = self.env.robot.get_qpos()[:-1]
            raw_obs["joint_vel"] = self.env.robot.get_qvel()[:-1]
            raw_obs["embedding"] = self.env.get_text_embedding()
            ee_translation = self.env.palm_link.get_pose().p
            ee_rotation = transforms3d.euler.quat2euler(
                self.env.palm_link.get_pose().q, axes="sxyz"
            )
            ee_gripper = self.env.robot.get_qpos()[arm_dof]
            ee_pos = np.concatenate([ee_translation, ee_rotation, [ee_gripper]])
            ee_vel = np.concatenate(
                [
                    self.env.palm_link.get_velocity(),
                    self.env.palm_link.get_angular_velocity(),
                    self.env.robot.get_qvel()[arm_dof : arm_dof + 1],
                ]
            )
            raw_obs["ee_pos"] = ee_pos
            raw_obs["ee_vel"] = ee_vel
            rgbs, depths = self.gui.render(depth=True)
            for cam_idx, cam in enumerate(self.gui.cams):
                curr_rgb = rgbs[cam_idx].copy()
                curr_depth = depths[cam_idx].copy()
                curr_rgb = cv2.resize(
                    curr_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA
                )
                curr_depth = cv2.resize(
                    curr_depth,
                    (self.img_w, self.img_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                raw_obs[f"{cam.name}_color"] = curr_rgb.transpose(2, 0, 1) / 255.0
                raw_obs[f"{cam.name}_depth"] = curr_depth[None] / 1000.0
                raw_obs[f"{cam.name}_intrinsic"] = cam.get_intrinsic_matrix()
                raw_obs[f"{cam.name}_extrinsic"] = cam.get_extrinsic_matrix()

            if "d3fields" in self.shape_meta["obs"]:
                use_dino = self.shape_meta["obs"]["d3fields"]["info"]["use_dino"]
                distill_dino = (
                    self.shape_meta["obs"]["d3fields"]["info"]["distill_dino"]
                    if "distill_dino" in self.shape_meta["obs"]["d3fields"]["info"]
                    else False
                )
                intrinsics = np.stack(
                    [cam.get_intrinsic_matrix() for cam in self.gui.cams], axis=0
                )  # (V, 3, 3)
                extrinsics = np.stack(
                    [cam.get_extrinsic_matrix() for cam in self.gui.cams], axis=0
                )  # (V, 4, 4)
                robot_base_pose_in_world = (
                    self.env.robot.get_pose().to_transformation_matrix()
                )  # (4, 4)
                qpos = self.env.robot.get_qpos()


                aggr_src_pts_ls, generated_code =   d3fields_proc(
                        fusion=self.fusion,
                        shape_meta=self.shape_meta["obs"]["d3fields"],
                        color_seq=np.stack(rgbs)[None],
                        depth_seq=np.stack(depths)[None] / 1000.0,
                        extri_seq=extrinsics[None],
                        intri_seq=intrinsics[None],
                        robot_base_pose_in_world_seq=robot_base_pose_in_world[None],
                        qpos_seq=qpos[None],
                        teleop_robot=self.teleop,
                        expected_labels=self.expected_labels,
                        attention_config=self.env.init_configuration,
                        prompt_info=self.prompt_info,
                    )
                
                if (self.prompt_info is not None) and (generated_code is not None):
                    self.log_code_reasoning(root_path = os.path.join(self.output_dir, "instructions_log"),
                                            save_idx = self.seed_idx, 
                                            instruction = self.instruction,
                                            code = generated_code,
                                            )


                aggr_src_pts = aggr_src_pts_ls[0]
                if use_dino or distill_dino:
                    # aggr_pts_feats = np.concatenate([aggr_src_pts, aggr_feats], axis=-1)
                    aggr_pts_feats = aggr_src_pts
                else:
                    aggr_pts_feats = aggr_src_pts

                raw_obs["d3fields"] = aggr_pts_feats.transpose(1, 0)

        self.render_cache = [
            (raw_obs[f"{render_obs_key}_color"].transpose(1, 2, 0) * 255).astype(
                np.uint8
            )
            for render_obs_key in self.render_obs_keys
        ]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs
    
    def log_code_reasoning(self, root_path, save_idx, instruction, code):
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        log_file_path = os.path.join(root_path, f"log_{save_idx}.txt")
        
        try:
            with open(log_file_path, "w") as log_file:
                log_file.write("Instruction:\n")
                log_file.write(instruction + "\n\n")
                log_file.write("Code:\n")
                log_file.write(code + "\n")
            print(f"Log saved successfully at {log_file_path}")
        except Exception as e:
            print(f"Failed to save log: {e}")

    def seed(self, seed=None):
        self.seed_idx = seed
        self.env.seed(seed)

    def render(self, mode=None):
        if self.render_cache is not None:
            image_horizon = np.concatenate(self.render_cache, axis=1).astype(
                np.uint8
            )  # concat horizontally
            return image_horizon, np.array(self.render_cache).astype(np.uint8)
        rgbs = self.gui.render()
        rgbs_sel = []
        for cam_idx, cam in enumerate(self.gui.cams):
            if cam.name in self.render_obs_keys:
                rgbs_sel.append(rgbs[cam_idx])
        # concat horizontally
        image_horizon = np.concatenate(rgbs_sel, axis=1).astype(np.uint8)
        return image_horizon, np.array(rgbs_sel).astype(np.uint8)

    def step(self, action_command):
        action = action_command["action"]
        action_proc = action_command["action_proc"]
        index = action_command["index"]
        first_step = action_command["first_step"]
        vis = action_command["vis"]

        if not self.is_joint:
            action_in_robot = transform_action_from_world_to_robot(
                action, self.env.robot.get_pose()
            )

            arm_dof = self.env.arm_dof
            joint_action = np.zeros(arm_dof + 1)
            joint_action[:arm_dof] = self.teleop.compute_ik_sapien(
                self.env.robot.get_qpos()[:], action_in_robot
            )[:arm_dof]
            joint_action[arm_dof:] = action_in_robot[-1]
        else:
            joint_action = action

        _, reward, done, info = self.env.step(joint_action)
        if vis and first_step:
            self.env.add_traj(action_proc, joint_action, index)
        obs = self.get_observation()
        return obs, reward, done, info

    def reset(self):
        if not self.pause_reset:
            self.env.reset()
            if self.init_states is not None:
                self.env.set_init(self.init_states)
        self.init_crate = None
        self.first_flag = True
        self.instruction, self.slackness = self.env.set_instruction()
        print("instruction : ", self.env.instruction, self.slackness)
        prompt_info = {}
        prompt_info["task"] = "# hang a mug on a branch \n"
        prompt_info["obj_list"] = "['mug','branch'] \n"
        prompt_info["prompt"] = self.env.instruction + "\n"
        self.prompt_info = prompt_info
        return self.get_observation()

    def close(self):
        self.env.close()
        self.gui.close()
        if self.fusion is not None:
            self.fusion.clear_xmem_memory()


def test():
    # dataset_dir = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/2023-10-17-02-09-09-909762'
    # dataset_dir = '/home/bing4090/yixuan_old_branch/general_dp/data/sapien_demo/demo_3'
    dataset_dir = (
        "/media/yixuan_2T/diffusion_policy/data/sapien_demo/mixed_mug_demo_200_v5"
    )
    env_cfg = OmegaConf.load(f"{dataset_dir}/config.yaml")
    env_cfg.manip_obj = "aluminum_mug"
    env: BaseRLEnv = hydra.utils.instantiate(env_cfg)
    add_default_scene_light(env.scene, env.renderer)
    shape_meta = {
        "action": {
            "shape": (8,),
            "key": "joint_action",
        },
        # 'action': {
        #     'shape': (10,)
        # },
        "obs": {
            "right_bottom_view_color": {
                "shape": (3, 480, 640),
                "type": "rgb",
            },
        },
    }
    dataset_path = os.path.join(dataset_dir, "episode_200.hdf5")
    with h5py.File(dataset_path) as f:
        init_states = f["info"]["init_poses"][()]
        wrapper = SapienEnvWrapper(env, shape_meta, init_states=init_states)
        wrapper.reset()
        action_seq = f["joint_action"][()]
        # action_seq = f['cartesian_action'][()]
        T = action_seq.shape[0]
        for t in tqdm(range(T)):
            action = action_seq[t]
            # action_rot = rotation_transformer.forward(action[3:6])
            # action = np.concatenate([action[:3], action_rot, action[6:]])
            _ = wrapper.step(action)
            img = wrapper.render()
            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    test()
