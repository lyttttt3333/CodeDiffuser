import IPython
import numpy as np
import transforms3d
from pyquaternion import Quaternion

e = IPython.embed
import os
import random
import sys

import sapien.core as sapien

from sapien_env.utils.random_utils import np_random

curr_path = os.path.abspath(__file__)
for _ in range(3):
    curr_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
import sys

sys.path.append("/home/bing4090/yixuan_old_branch/general_dp/sapien_env")
sys.path.append("/home/bing4090/yixuan_old_branch/general_dp/general_dp")

from sapien_env.rl_env.hang_mug_env import HangMugRLEnv

MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


class SingleArmPolicy:
    def __init__(self, seed=0, inject_noise=False):
        self.seed(random.randint(0, 999))
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None
        self.transitional_noise_scale = 0.025
        self.rotational_noise_scale = 0.3

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        # quat = curr_quat + (next_quat - curr_quat) * t_frac
        # interpolate quaternion using slerp
        curr_quat_obj = Quaternion(curr_quat)
        next_quat_obj = Quaternion(next_quat)
        quat = Quaternion.slerp(curr_quat_obj, next_quat_obj, t_frac).elements
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self, env, ee_link_pose, mode="straight", assign_idx=None):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(
                env, ee_link_pose, mode, assign_idx[0], assign_idx[1]
            )

        if self.trajectory[0]["t"] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)

        if len(self.trajectory) == 0:
            quit = True
            return None, None, quit
        else:
            quit = False

        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(
            self.curr_waypoint, next_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        cartisen_action_dim = 6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim + grip_dim)
        eluer = transforms3d.euler.quat2euler(quat, axes="sxyz")
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper

        self.progress = 0.1

        return cartisen_action, None, quit

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def transitional_noise(self):
        x_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        y_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        z_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        return np.array([x_noise, y_noise, z_noise])

    def grasp_noise(self):
        x_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        y_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        z_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        return np.array([x_noise, y_noise, z_noise]) * 5

    def rotational_noise(self):
        z_noise = self.np_random.uniform(
            -self.rotational_noise_scale, self.rotational_noise_scale
        )
        return np.array([0, 0, z_noise])

    def generate_trajectory(
        self, env: HangMugRLEnv, ee_link_pose, mode="straight", mug_idx=0, branch_idx=0
    ):
        init_position = env.manipulated_object[mug_idx].get_pose().p

        # if env.manip_obj_name == 'nescafe_mug':
        if True:
            pre_grasp_p = init_position + np.array([-0.2, -0.1, 0.3])
            grasp_p = init_position + np.array([-0.2, -0.1, 0.16])
            ee_init_euler = np.array([np.pi, 0, np.pi / 2])
            grasp_q = transforms3d.euler.euler2quat(*ee_init_euler)

            place_p_in_world = np.array([0.14, 0.14, 0.62])

            post_place_p_in_world = np.array([0.1, 0.15, 0.42])

            leave_p_in_world = np.array([0.05, 0.1, 0.45])

            place_q_in_world_mat = np.array(
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
            )
            # place_q_in_world_mat = np.array([[1.0, 0.0, 0.0],
            #                                  [0.0, -1.0, 0.0],
            #                                  [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(
                transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist()
            )

            place_q_in_world = grasp_q
            # place_q_in_world = np.array(transforms3d.euler.euler2quat(1.15*np.pi, -0.5*np.pi, np.pi/2.0, axes='sxyz').tolist())

        pre_grasp_pose = sapien.Pose(pre_grasp_p, grasp_q)
        grasp_pose = sapien.Pose(grasp_p, grasp_q)

        mug_pose_mat = (
            env.manipulated_object[mug_idx].get_pose().to_transformation_matrix()
        )
        grasp_pose_in_mug_mat = grasp_pose.to_transformation_matrix()
        grasp_pose_in_world_mat = mug_pose_mat @ grasp_pose_in_mug_mat
        grasp_pose_in_world = sapien.Pose.from_transformation_matrix(
            grasp_pose_in_world_mat
        )

        pre_grasp_pose_in_mug_mat = pre_grasp_pose.to_transformation_matrix()
        pre_grasp_pose_in_world_mat = mug_pose_mat @ pre_grasp_pose_in_mug_mat
        pre_grasp_pose_in_world = sapien.Pose.from_transformation_matrix(
            pre_grasp_pose_in_world_mat
        )

        init_position = env.manipulated_object[mug_idx].get_pose().p

        grasp_euler = np.array([np.pi, 0, np.pi / 2])
        grasp_quat = transforms3d.euler.euler2quat(*grasp_euler)
        pre_grasp_p = init_position + np.array([-0.0, -0.0, 0.3])
        grasp_p = init_position + np.array([-0.0, -0.0, 0.12])

        z_offset = np.pi / 2
        GRASP_Height = 0.12
        PLACE_Height_0 = 0.52
        PLACE_Height_1 = 0.52
        PLACE_Height_2 = 0.412
        CLOSE = 0.0
        x_offset = 0.0

        PRE_GRASP_HEIGHT = 0.3

        if branch_idx == 0:
            ee_init_euler = np.array([np.pi, 0, np.pi / 2])
            init_quat = transforms3d.euler.euler2quat(*ee_init_euler)
            init_position = env.manipulated_object[mug_idx].get_pose().p
            theta = (
                transforms3d.euler.quat2euler(
                    env.manipulated_object[mug_idx].get_pose().q
                )[-1]
                + z_offset
            )

            grasp_euler = np.array([np.pi, 0, -np.pi / 2 + theta])
            grasp_quat = transforms3d.euler.euler2quat(*grasp_euler)
            pre_grasp_p = init_position + np.array([-0.0, -0.0, PRE_GRASP_HEIGHT])
            grasp_p = init_position + np.array([-0.0, -0.0, GRASP_Height])

            place_p_in_world = np.array([0.14, 0.105, PLACE_Height_0])
            pre_place_p = np.array([0.14, 0.105, PLACE_Height_0 - 0.02])
            place_p = np.array([0.02, 0.105, PLACE_Height_0 - 0.02])
            post_place_p = np.array([0.05, 0.105, 0.6])
            place_q = transforms3d.euler.euler2quat(*ee_init_euler)
        elif branch_idx == 1:
            ee_init_euler = np.array([np.pi, 0, np.pi / 2])
            init_quat = transforms3d.euler.euler2quat(*ee_init_euler)
            init_position = env.manipulated_object[mug_idx].get_pose().p
            theta = (
                transforms3d.euler.quat2euler(
                    env.manipulated_object[mug_idx].get_pose().q
                )[-1]
                + z_offset
            )

            grasp_euler = np.array([np.pi, 0, -np.pi / 2 + theta])
            grasp_quat = transforms3d.euler.euler2quat(*grasp_euler)
            pre_grasp_p = init_position + np.array([-0.0, -0.0, PRE_GRASP_HEIGHT])
            grasp_p = init_position + np.array([-0.0, -0.0, GRASP_Height])

            place_p_in_world = np.array([-0.14, 0.105, PLACE_Height_1])
            pre_place_p = np.array([-0.14, 0.105, PLACE_Height_1 - 0.02])
            place_p = np.array([-0.02, 0.105, PLACE_Height_1 - 0.02])
            post_place_p = np.array([-0.05, 0.105, 0.6])
            place_q = transforms3d.euler.euler2quat(*ee_init_euler)
        elif branch_idx == 2:
            ee_init_euler = np.array([np.pi, 0, np.pi / 2])
            init_quat = transforms3d.euler.euler2quat(*ee_init_euler)
            init_position = env.manipulated_object[mug_idx].get_pose().p
            theta = (
                transforms3d.euler.quat2euler(
                    env.manipulated_object[mug_idx].get_pose().q
                )[-1]
                + z_offset
            )

            grasp_euler = np.array([np.pi, 0, -np.pi / 2 + theta])
            grasp_quat = transforms3d.euler.euler2quat(*grasp_euler)

            pre_grasp_p = init_position + np.array([-0.0, -0.0, PRE_GRASP_HEIGHT])
            grasp_p = init_position + np.array([-0.0, -0.0, GRASP_Height])

            place_p_in_world = np.array([0.02, 0.045, PLACE_Height_2])
            pre_place_p = np.array([0.0 + x_offset, 0.047, PLACE_Height_2 - 0.02])
            place_p = np.array([0.0 + x_offset, 0.076, PLACE_Height_2 - 0.02])
            post_place_p = np.array([0.0, 0.076, 0.6])
            ee_end_euler = np.array([np.pi, 0, np.pi / 2 - np.pi / 6])
            place_q = transforms3d.euler.euler2quat(*ee_end_euler)
        else:
            raise

        np.random.seed()
        pre_grasp_rotation_noise = transforms3d.euler.euler2quat(
            *self.rotational_noise()
        )
        R = transforms3d.euler.euler2mat(0, 0, theta)

        if mode == "straight":
            self.trajectory = [
                {
                    "t": 0,
                    "xyz": ee_link_pose.p,
                    "quat": ee_link_pose.q,
                    "gripper": 0.09,
                },
                {
                    "t": 20,
                    "xyz": pre_grasp_p + self.transitional_noise() * 0.05,
                    "quat": grasp_quat + pre_grasp_rotation_noise * 0.05,
                    "gripper": 0.09,
                },
                {"t": 50, "xyz": grasp_p, "quat": grasp_quat, "gripper": 0.09},
                {"t": 60, "xyz": grasp_p, "quat": grasp_quat, "gripper": CLOSE},
                {
                    "t": 70,
                    "xyz": pre_grasp_p + self.transitional_noise() * 0.05,
                    "quat": init_quat,
                    "gripper": CLOSE,
                },
                {
                    "t": 100,
                    "xyz": place_p_in_world + self.transitional_noise() * 0.05,
                    "quat": place_q,
                    "gripper": CLOSE,
                },
                {"t": 110, "xyz": pre_place_p, "quat": place_q, "gripper": CLOSE},
                {"t": 130, "xyz": place_p, "quat": place_q, "gripper": CLOSE},
                {"t": 140, "xyz": place_p, "quat": place_q, "gripper": 0.09},
                {"t": 150, "xyz": post_place_p, "quat": place_q, "gripper": 0.09},
            ]
            place_waypoint = np.array([0.0, 0.076, 0.392])
            self.grasp_keypoint = np.concatenate(
                [
                    grasp_p[None, :]
                    + np.array([0, 0, -0.1] + self.transitional_noise() * 0),
                    place_waypoint[None, :]
                    + np.array([0, 0, -0.1])
                    + self.transitional_noise() * 0,
                ],
                axis=0,
            )
        else:
            raise RuntimeError("mode not implemented")

    def key_point(self):
        return self.grasp_keypoint
