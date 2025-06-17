from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d

from sapien_env.rl_env.base_pack import BaseRLEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.sim_env.pack_env import PackEnv
from sapien_env.utils.common_robot_utils import (
    generate_arm_robot_hand_info,
    generate_free_robot_hand_info,
    generate_panda_info,
)


class PackRLEnv(PackEnv, BaseRLEnv):
    def __init__(
        self,
        use_gui: bool = False,
        frame_skip: int = 10,
        task_level_multimodality: bool = False,
        nine_pos_mode: bool = False,
        stand_mode: bool = False,
        simple_mode: bool = False,
        fix_pick: bool = False,
        num_obj_wait: int = 1,
        num_obj_done: Optional[int] = None,
        robot_name: str = "panda",
        object_pose_noise: float = 0.01,
        seed: Optional[int] = None,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui,
            frame_skip,
            task_level_multimodality=task_level_multimodality,
            nine_pos_mode=nine_pos_mode,
            stand_mode=stand_mode,
            simple_mode=simple_mode,
            fix_pick=fix_pick,
            num_obj_wait=num_obj_wait,
            num_obj_done=num_obj_done,
            seed=seed,
            **renderer_kwargs,
        )
        self.setup(robot_name)

        # NOTE: need to decide later
        self.object_pose_noise = object_pose_noise

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

        finger_tip_names = ["panda_leftfinger", "panda_rightfinger"]

        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_tip_names
        ]


    def get_robot_state(self) -> np.ndarray:
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def get_reward(self, action: Optional[np.ndarray] = None) -> float:
        # 1.0 if success, 0.0 otherwise
        reward = 0
        obj_container_dist_thresh = 0.03
        obj_align_thresh = 0.99

        for idx in range(len(self.obj_wait)):
            correct_pos, _ = self.get_obj_done_pose(idx + len(self.obj_done))
            obj_pose = self.obj_wait[idx]["actor"].get_pose()
            obj_pos = obj_pose.p
            obj_z_axis = obj_pose.to_transformation_matrix()[:3, 1]
            obj_box_dist = np.linalg.norm(
                obj_pos[:2] - correct_pos[:2]
            )  # ignore z axis
            obj_align = np.sum(np.array([0, 0, 1]) * obj_z_axis)
            if (
                obj_box_dist < obj_container_dist_thresh
                and obj_align > obj_align_thresh
            ):
                reward += 1 / len(self.obj_wait)

        reward = round(reward, 4)
        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        if self.is_xarm:
            qpos = np.zeros(self.robot.dof)
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        elif self.is_trossen_arm:
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
            init_pos = np.array([0.0, -0.5, 0.0])
            init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
            init_pose = sapien.Pose(init_pos, init_ori)
        else:
            init_pose = sapien.Pose(
                np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0)
            )
        self.robot.set_pose(init_pose)
        self.reset_internal()
        for _ in range(100):
            self.robot.set_qf(
                self.robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False
                )
            )
            self.scene.step()

        return self.get_observation()

    @cached_property
    def obs_dim(self) -> int:
        return self.robot.dof + 7 + 6 + 3 + 1

    def is_done(self) -> bool:
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self) -> int:
        return 10000


def main_env():
    from sapien.utils import Viewer

    task_level_multimodality = True
    nine_pos_mode = False
    num_obj_wait = 4
    num_obj_done = 0
    pause = True

    env = PackRLEnv(
        use_gui=True,
        robot_name="panda",
        frame_skip=10,
        task_level_multimodality=task_level_multimodality,
        nine_pos_mode=nine_pos_mode,
        num_obj_wait=num_obj_wait,
        num_obj_done=num_obj_done,
    )
    robot_dof = env.arm_dof + 1
    env.reset()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    viewer.set_fovy(1.2)
    env.viewer = viewer
    if pause:
        viewer.toggle_pause(True)

    for _ in range(5000):
        action = np.zeros(robot_dof)
        env.step(action)
        env.render()

    while not viewer.closed:
        env.render()


if __name__ == "__main__":
    main_env()