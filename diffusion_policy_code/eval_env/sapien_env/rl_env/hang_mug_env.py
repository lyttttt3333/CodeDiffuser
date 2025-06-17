import sys
from functools import cached_property
from typing import Optional

sys.path.append("/home/bing4090/yixuan_old_branch/general_dp/sapien_env")

import numpy as np
import sapien.core as sapien
import transforms3d
from d3fields.utils.instruction_generate.generate_instruction import (
    instruction_generater,
)
from d3fields.utils.text_embedding import from_text_to_embedding
from sapien.utils import Viewer

from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.sim_env.hang_mug_env import HangMugEnv
from sapien_env.utils.common_robot_utils import (
    generate_arm_robot_hand_info,
    generate_free_robot_hand_info,
    generate_panda_info,
)


class HangMugRLEnv(HangMugEnv, BaseRLEnv):
    def __init__(
        self,
        use_gui=False,
        frame_skip=5,
        robot_name="xarm6",
        constant_object_state=False,
        object_scale=1.0,
        randomness_scale=1,
        friction=1,
        object_pose_noise=0.01,
        manip_obj="nescafe_mug",
        assign_idx=None,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui,
            frame_skip,
            object_scale,
            randomness_scale,
            friction,
            use_ray_tracing=False,
            manip_obj=manip_obj,
            **renderer_kwargs,
        )
        self.setup(robot_name)

        self.constant_object_state = constant_object_state
        self.object_pose_noise = object_pose_noise
        self.mug_idx = assign_idx[0]
        self.branch_idx = assign_idx[1]

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError
        self.palm_link_name = info.palm_name
        self.palm_link = [
            link
            for link in self.robot.get_links()
            if link.get_name() == self.palm_link_name
        ][0]

        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["panda_leftfinger", "panda_rightfinger"]

        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_tip_names
        ]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()
        self.last_seq = np.zeros([16, 7])

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = (
            self.object_episode_init_pose
            if self.constant_object_state
            else self.manipulated_object[self.mug_idx].get_pose()
        )
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        z_axis = object_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.sum(np.array([0, 0, 1]) * z_axis)
        palm_pose = self.palm_link.get_pose()
        object_in_palm = object_pose.p - palm_pose.p
        v = self.manipulated_object[self.mug_idx].get_velocity()
        w = self.manipulated_object[self.mug_idx].get_angular_velocity()
        return np.concatenate(
            [
                robot_qpos_vec,
                object_pose_vec,
                v,
                w,
                object_in_palm,
                np.array([theta_cos]),
            ]
        )

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def set_instruction(self):
        import random

        random.seed(self.seed_idx)

        # slackness_type = random.choice(
        #     ["no_slackness", "mug_slackness", "branch_slackness", "both_slackness"]
        # )

        slackness_type = random.choice(
            ["no_slackness", "mug_slackness"]
        )

        slackness_type = random.choice(
            ["no_slackness"]
        )

        mug_idx = 0
        branch_idx = self.seed_idx % 3

        assign_idx = [mug_idx, branch_idx]

        obj_color_pool = ["red mug", "white mug", "blue mug", "green mug"]

        manip_obj = self.manip_obj_name
        obj_color = [
            obj_color_pool[self.select_index[0]],
            obj_color_pool[self.select_index[1]],
        ]
        vis_info = {
            "mug": {
                manip_obj[0]: [obj_color[0], obj_color[1]],
                manip_obj[1]: [obj_color[1], obj_color[0]],
            },
            "branch": {
                "0": ["right-topmost branch", "left-topmost branch"],
                "1": ["left-topmost branch", "right-topmost branch"],
                "2": ["right-middle branch", "left-middle branch"],
            },
        }
        init_configuration = self.get_layout(assign_idx)
        self.init_configuration = init_configuration
        self.generater = instruction_generater(
            seed=self.seed_idx,
            keys=["mug", "branch"],
            template_path="/home/yitong/diffusion/diffusion_policy_code/d3fields_dev/d3fields/utils/instruction_generate/template.json",
            slackness_type=slackness_type,
        )
        descriptive_element = self.generater.from_config_to_string(
            init_configuration, manip_obj, vis_info
        )
        self.instruction = self.generater.fill_in_template(descriptive_element)
        self.slackness_type = slackness_type
        self.assign_idx = assign_idx
        self.embedding = from_text_to_embedding(self.instruction)
        return self.instruction, slackness_type

    def get_text_embedding(self):
        return self.embedding

    def get_reward(self, action):
        # 1.0 if success, 0.0 otherwise
        branch_0 = np.array([[0, 0.05], [0.08, 0.12], [0.250, 0.350]])

        branch_1 = np.array([[-0.05, 0], [0.08, 0.12], [0.250, 0.350]])
        branch_2 = np.array([[-0.05, 0.1], [0.08, 0.12], [0.15, 0.250]])
        branch_position = [branch_0, branch_1, branch_2]

        if (
            self.slackness_type == "no_slackness"
            or self.slackness_type == "mug_slackness"
        ):
            final_position_constraint = [branch_position[self.assign_idx[-1]]]
        else:
            final_position_constraint = branch_position

        mug_positions = []
        for obj in self.manipulated_object:
            position = obj.get_pose().p
            mug_positions.append(position)

        if (
            self.slackness_type == "no_slackness"
            or self.slackness_type == "branch_slackness"
        ):
            final_position = [mug_positions[self.assign_idx[0]]]
        else:
            final_position = mug_positions

        # if self.slackness_type == "both_slackness":
        #     return True

        # if self.branch_idx == 0:
        #     obj_pose = [self.mug_idx].get_pose()
        #     tree_pose = self.mug_tree.get_pose()
        #     obj_pos = obj_pose.p
        #     z = obj_pos[-1]
        #     height_th = 0.27
        #     reward = z > height_th
        # else:
        #     obj_pose = self.manipulated_object[self.mug_idx].get_pose()
        #     tree_pose = self.mug_tree.get_pose()
        #     obj_pos = obj_pose.p
        #     z = obj_pos[-1]
        #     height_th = 0.15
        #     reward = z > height_th

        if self.check_fit_in(
            position=final_position, constraint=final_position_constraint
        ):
            reward = 1
        else:
            reward = 0
        return float(reward)

    def check_fit_in(self, position, constraint):
        for tgt_pos in position:
            for tgt_con in constraint:
                x, y, z = tgt_pos
                x_min = tgt_con[0, 0]
                x_max = tgt_con[0, 1]
                y_min = tgt_con[1, 0]
                y_max = tgt_con[1, 1]
                z_min = tgt_con[2, 0]
                z_max = tgt_con[2, 1]
                if (
                    (x > x_min)
                    and (x_max > x)
                    and (y > y_min)
                    and (y_max > y)
                    and (z > z_min)
                    and (z_max > z)
                ):
                    return True
        return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        if self.is_xarm:
            qpos = np.zeros(self.robot.dof)
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        elif self.is_trossen_arm:
            # print("trossen_arm")
            qpos = np.zeros(self.robot.dof)
            qpos[self.arm_dof :] = [0.021, -0.021]
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))

        elif self.is_panda:
            qpos = self.robot_info.arm_init_qpos.copy()
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array([0.0, -0.5, -0.1])
            init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
            init_pose = sapien.Pose(init_pos, init_ori)
        else:
            init_pose = sapien.Pose(
                np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0)
            )
        self.robot.set_pose(init_pose)
        self.reset_internal()

        ###
        bound_theta_low = -np.pi / 9
        bound_theta_high = np.pi / 9

        position_list = []

        region = np.random.uniform(-1, 1)
        theta = np.random.uniform(bound_theta_low, bound_theta_high)
        pose = transforms3d.euler.euler2quat(
            np.pi / 2, 0, (+np.pi / 2) + theta, axes="sxyz"
        )
        if region >= 0:
            x = self.np_random.uniform(low=0.15, high=0.225)
            y = self.np_random.uniform(low=-0.125, high=-0.05)
        else:
            x = self.np_random.uniform(low=-0.10, high=-0.00)
            y = self.np_random.uniform(low=-0.15, high=-0.1)
        position_list.append(np.array([x, y]))
        pose = sapien.Pose(p=np.array([x, y, 0.05]), q=pose)
        self.manipulated_object[self.mug_idx].set_pose(pose=pose)

        for idx, obj in enumerate(self.manipulated_object):
            if idx != self.mug_idx:
                theta = np.random.uniform(bound_theta_high, bound_theta_low)
                pose = transforms3d.euler.euler2quat(
                    np.pi / 2, 0, (+np.pi / 2) + theta, axes="sxyz"
                )
                while True:
                    if region <= 0:
                        x = self.np_random.uniform(low=0.15,high=0.225)
                        y = self.np_random.uniform(low=-0.125,high=-0.05)
                    else:
                        x = self.np_random.uniform(low=-0.10,high=-0.00)
                        y = self.np_random.uniform(low=-0.15,high=-0.1)
                    min_dist = None

                    for position in position_list:
                        dist = np.linalg.norm(position - np.array([x, y]), ord=2)
                        if min_dist == None:
                            min_dist = dist
                        elif dist < min_dist:
                            min_dist = dist
                        else:
                            pass
                    if min_dist > 0.12:
                        position_list.append(np.array([x, y]))
                        break
                    # if region >= 0:
                    #     x = np.random.uniform(low=-0.10,high=-0.00)
                    #     # y = np.random.uniform(low=-0.225,high=-0.1)
                    #     y = np.random.uniform(low=-0.15,high=-0.1)
                    # else:
                    #     x = np.random.uniform(low=0.15,high=0.225)
                    #     # y = np.random.uniform(low=-0.20,high=-0.05)
                    #     y = np.random.uniform(low=-0.125,high=-0.05)

                pose = sapien.Pose(p=np.array([x, y, 0.05]), q=pose)
                obj.set_pose(pose=pose)

        for i in range(100):
            self.robot.set_qf(
                self.robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False
                )
            )
            self.scene.step()
        # self.object_episode_init_pose = self.manipulated_object.get_pose()
        # random_quat = transforms3d.euler.euler2quat(
        #     *(self.np_random.randn(3) * self.object_pose_noise * 10)
        # )
        # random_pos = self.np_random.randn(3) * self.object_pose_noise
        # self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(
        #     random_pos, random_quat
        # )

        return self.get_observation()

    # def set_init(self, init_states):
    #     init_pose = sapien.Pose.from_transformation_matrix(init_states[0])
    #     self.manipulated_object.set_pose(init_pose)
    #     init_box_pose = sapien.Pose.from_transformation_matrix(init_states[1])
    #     self.mug_tree.set_pose(init_box_pose)

    def add_traj(self, action_seq):
        if np.sum(np.abs(action_seq - self.last_seq)) == 0:
            pass
        else:
            try:
                for i in range(len(self.wp)):
                    actor = self.wp[i]
                    self.scene.remove_actor(actor=actor)
            except:
                pass
            self.wp = []
            blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            for i in range(len(action_seq)):
                action = action_seq[i] + np.array([0, 0, -0.1, 0, 0, 0, 0])
                pose = sapien.Pose(p=action[:3])
                builder = self.scene.create_actor_builder()
                color = blue * i / 10 + red * (1 - i / 10)
                builder.add_box_visual(
                    pose=pose,
                    half_size=0.0005 * np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    color=color,
                )
                wp = builder.build_static(name=f"wp_{i}")
                self.wp.append(wp)
        self.last_seq = action_seq

    def add_full_traj(self):
        full_action_seq = np.load("/home/sim/variation_with_color.npy")
        num = full_action_seq.shape[0]
        blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for j in range(num):
            action_seq = full_action_seq[j]
            for i in range(action_seq.shape[0]):
                action = action_seq[i]
                pose = sapien.Pose(p=action[:3] + np.array([0, 0, -0.1]))
                builder = self.scene.create_actor_builder()
                color = blue * (1 - i / 60) + red * (i / 60)
                color = action[3:]
                builder.add_box_visual(
                    pose=pose,
                    half_size=0.002 * np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    color=color,
                )
                wp = builder.build_static(name=f"wp_{i}_{j}")
                # self.wp.append(wp)

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 3 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 10000


def main_env():
    env = HangMugRLEnv(
        use_gui=True, robot_name="panda", frame_skip=10, use_visual_obs=False
    )
    base_env = env
    robot_dof = env.arm_dof + 1
    env.seed(0)
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    viewer.set_fovy(2.0)
    base_env.viewer = viewer

    viewer.toggle_pause(False)
    from tqdm import tqdm

    for i in tqdm(range(200)):
        action = np.zeros(robot_dof)
        action[2] = 0.01
        obs, reward, done, _ = env.step(action)
        env.render()

    viewer.toggle_pause(True)
    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == "__main__":
    main_env()
