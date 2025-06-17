from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

import sys
import os
sys.path.append(os.environ['SAPIEN_ROOT'])
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.relocate_env import RelocateEnv
from sapien_env.rl_env.para import ARM_INIT

OBJECT_LIFT_LOWER_LIMIT = -0.03

class RelocateRLEnv(RelocateEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", constant_object_state=False,
                 rotation_reward_weight=0, object_category="YCB", object_name="tomato_soup_can", object_scale=1.0,
                 randomness_scale=1, friction=1, object_pose_noise=0.01,use_ray_tracing=False, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_category, object_name, object_scale, randomness_scale, friction,use_ray_tracing=use_ray_tracing,
                         **renderer_kwargs)
        self.setup(robot_name)
        self.constant_object_state = constant_object_state
        self.rotation_reward_weight = rotation_reward_weight
        self.object_pose_noise = object_pose_noise

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["vx300s/left_finger_link", "vx300s/right_finger_link"]
        contact_sensor = True
        if contact_sensor:
            finger_contact_link_name = [
                "vx300s/left_tactile_map_link1", "vx300s/left_tactile_map_link2","vx300s/left_tactile_map_link3","vx300s/left_tactile_map_link4",
                "vx300s/left_tactile_map_link5", "vx300s/left_tactile_map_link6","vx300s/left_tactile_map_link7","vx300s/left_tactile_map_link8",
                "vx300s/left_tactile_map_link9", "vx300s/left_tactile_map_link10","vx300s/left_tactile_map_link11","vx300s/left_tactile_map_link12",
                "vx300s/left_tactile_map_link13", "vx300s/left_tactile_map_link14","vx300s/left_tactile_map_link15","vx300s/left_tactile_map_link16",
                "vx300s/right_tactile_map_link1","vx300s/right_tactile_map_link2","vx300s/right_tactile_map_link3","vx300s/right_tactile_map_link4",
                "vx300s/right_tactile_map_link5","vx300s/right_tactile_map_link6","vx300s/right_tactile_map_link7","vx300s/right_tactile_map_link8",
                "vx300s/right_tactile_map_link9","vx300s/right_tactile_map_link10","vx300s/right_tactile_map_link11","vx300s/right_tactile_map_link12",
                "vx300s/right_tactile_map_link13","vx300s/right_tactile_map_link14","vx300s/right_tactile_map_link15","vx300s/right_tactile_map_link16",

            ]
        else:
            finger_contact_link_name = [
                "vx300s/left_finger_link", "vx300s/right_finger_link"
            ]
        
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_tip_names]
        self.finger_contact_links = [self.robot.get_links()[robot_link_names.index(name)] for name in
                                     finger_contact_link_name]
        self.finger_tip_pos = np.zeros([len(finger_tip_names), 3])

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)
        self.object_in_tip = np.zeros([len(finger_tip_names), 3])
        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0

        self.cube_link = [
            link for link in self.manipulated_object.get_links() if link.get_name() == "cube"][0]
        # Contact buffer
        self.robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # four tip, palm
        self.sensor_boolean = np.zeros(2)


    def update_cached_state(self):
        check_contact_links = self.finger_contact_links
        # print("actior1:",check_contact_links)
        # print("actor2:",self.cube_link)
        contacts = self.check_actor_contacts_continuous(check_contact_links, self.cube_link)
        self.sensors = contacts
        # print(self.sensor_boolean)
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.object_in_tip = self.object_pose.p[None, :] - self.finger_tip_pos
        self.object_lift = self.object_pose.p[2] - self.object_height
        self.target_in_object = self.target_pose.p - self.object_pose.p
        self.target_in_object_angle[0] = np.arccos(
            np.clip(np.power(np.sum(self.object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        target_in_object = self.target_pose.p - object_pose.p
        target_in_palm = self.target_pose.p - palm_pose.p
        object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        theta = np.arccos(np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        return np.concatenate(
            [robot_qpos_vec, object_pose_vec, palm_v, palm_w, object_in_palm, target_in_palm, target_in_object,
             self.target_pose.q, np.array([theta])])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q])

    def get_reward(self, action):
        object_pose = self.manipulated_object.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.check_contact(self.robot_collision_links, [self.manipulated_object])

        reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
        if is_contact:
            reward += 0.1
            lift = min(object_pose.p[2], self.target_pose.p[2]) - self.object_height
            lift = max(lift, 0)
            reward += 5 * lift
            if lift > 0.015:
                reward += 2
                obj_target_distance = min(np.linalg.norm(object_pose.p - self.target_pose.p), 0.5)
                reward += -1 * min(np.linalg.norm(palm_pose.p - self.target_pose.p), 0.5)
                reward += -3 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.1:
                    reward += (0.1 - obj_target_distance) * 20
                    theta = np.arccos(
                        np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
                    reward += max((np.pi / 2 - theta) * self.rotation_reward_weight, 0)
                    if theta < np.pi / 4 and self.rotation_reward_weight >= 1e-6:
                        reward += (np.pi / 4 - theta) * 6 * self.rotation_reward_weight

        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        if not self.is_robot_free:
            if self.is_xarm:
                qpos = np.zeros(self.robot.dof)
                arm_qpos = self.robot_info.arm_init_qpos
                qpos[:self.arm_dof] = arm_qpos
                self.robot.set_qpos(qpos)
                self.robot.set_drive_target(qpos)
                init_pos = ARM_INIT + self.robot_info.root_offset
                init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
            if self.is_trossen_arm:
                print("trossen_arm")
                qpos = np.zeros(self.robot.dof)
                qpos[self.arm_dof:] =[0.021,-0.021]
                arm_qpos = self.robot_info.arm_init_qpos
                qpos[:self.arm_dof] = arm_qpos
                self.robot.set_qpos(qpos)
                self.robot.set_drive_target(qpos)
                init_pos = ARM_INIT + self.robot_info.root_offset
                init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
            
        else:
            init_pose = sapien.Pose(np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        self.robot.set_pose(init_pose)
        self.reset_internal()
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        self.update_cached_state()
        return self.get_observation()

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return len(self.get_oracle_state())
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return self.manipulated_object.pose.p[2] - self.object_height < OBJECT_LIFT_LOWER_LIMIT

    @cached_property
    def horizon(self):
        return 250


def main_env():
    env = RelocateRLEnv(use_gui=True, robot_name="trossen_vx300s_tactile_thin",
                        object_name="mustard_bottle", frame_skip=10, use_visual_obs=False, use_ray_tracing=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    for link in env.robot.get_links():
        print(link.get_name(),link.get_pose())
    for joint in env.robot.get_active_joints():
        print(joint.get_name())

    action = np.zeros(7)
    viewer.toggle_pause(True)
    for i in range(5000):
        # action[5] = 0.06
        env.reset()
        # obs, reward, done, _ = env.step(action)
        env.render()

    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
