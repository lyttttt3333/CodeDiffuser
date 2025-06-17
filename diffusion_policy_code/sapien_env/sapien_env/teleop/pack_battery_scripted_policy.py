from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien_env.rl_env.pack_battery_env import PackBatteryRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.teleop.base import BasePolicy


class SingleArmPolicy(BasePolicy):
    def __init__(
        self,
        seed: Optional[int] = None,
        inject_noise: bool = False,
        velocity_scale: float = 1,
        use_cubic: bool = True,
    ):
        super().__init__(
            seed=seed,
            inject_noise=inject_noise,
            velocity_scale=velocity_scale,
            use_cubic=use_cubic,
        )

    def generate_trajectory(
        self, env: PackBatteryRLEnv, ee_link_pose: sapien.Pose
    ) -> None:
        ee_init_xyz = ee_link_pose.p
        ee_init_euler = np.array([np.pi, 0, np.pi / 2])

        self.trajectory = [
            {
                "t": 0,
                "xyz": ee_init_xyz,
                "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                "gripper": 0.09,
            }
        ]

        self.attention_config = dict()

        grasp_h = 0.11
        place_h = 0.19
        none_stand_grasp_h = 0.095
        none_stand_place_h = 0.06
        height4move = 0.25
        place_offset = -0.105

        
        for idx, obj in enumerate(env.obj_wait):
            init_position = obj["position"]
            init_angle = obj["euler"][-1]
            obj_stand = obj["stand"]
            goal_position, _ = env.get_obj_done_pose(env.target_idx)

            grasp_rotational_noise = self.rotational_noise()
            release_rotational_noise = self.rotational_noise()
            
            if obj_stand:
                init_position = init_position
                grasp_position = init_position + np.array([0, 0, grasp_h ])
            else:
                init_position = init_position + np.array([0, -0.027, 0 ]) @ transforms3d.euler.euler2mat(0,0,init_angle).transpose(1,0)
                grasp_position = init_position + np.array([0, 0, none_stand_grasp_h])

            self.attention_config["init_position"] = init_position
            self.attention_config["goal_position"] = goal_position

            self.trajectory += [
                {
                    "t": 15,
                    "xyz": init_position
                    + np.array([0, 0, height4move])
                    + self.transitional_noise()*3,
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                grasp_rotational_noise
                                if obj_stand
                                else [0, 0, init_angle]
                            )
                        )
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 10,
                    "xyz": init_position
                    + np.array(
                        [
                            0,
                            0,
                            (height4move + grasp_h) / 2
                            if obj_stand
                            else none_stand_grasp_h,
                        ]
                    ),
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                grasp_rotational_noise
                                if obj_stand
                                else [0, 0, init_angle]
                            )
                        )
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 15,
                    "xyz": grasp_position,
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                grasp_rotational_noise*0
                                if obj_stand
                                else [0, 0, init_angle]
                            )
                        )
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 10,
                    "xyz": grasp_position,
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                grasp_rotational_noise*0
                                if obj_stand
                                else [0, 0, init_angle]
                            )
                        )
                    ),
                    "gripper": 0.0,
                },
                {
                    "t": 5,
                    "xyz": grasp_position,
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                grasp_rotational_noise*0
                                if obj_stand
                                else [0, 0, init_angle]
                            )
                        )
                    ),
                    "gripper": 0.0,
                },
                {
                    "t": 15,
                    "xyz": init_position
                    + np.array([0, 0, height4move])
                    + self.transitional_noise(),
                    "quat": transforms3d.euler.euler2quat(
                        *(ee_init_euler + self.rotational_noise())
                    ),
                    "gripper": 0.0,
                },
                {
                    "t": 30,
                    "xyz": goal_position
                    + np.array(
                        [
                            0 if obj_stand else place_offset,
                            0,
                            height4move + (0 if obj_stand else -0.1),
                        ]
                    )
                    + self.transitional_noise(False) * 2,
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                release_rotational_noise
                                if obj_stand
                                else [-np.pi / 2, np.pi / 2, 0]
                            )
                            + self.rotational_noise()
                        )
                    ),
                    "gripper": 0.0,
                },
                {
                    "t": 15,
                    "xyz": goal_position
                    + np.array(
                        [
                            0 if obj_stand else place_offset,
                            0 ,
                            (height4move + 3 * place_h) / 4
                            if obj_stand
                            else none_stand_place_h,
                        ]
                    ),
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                release_rotational_noise
                                if obj_stand
                                else [-np.pi / 2, np.pi / 2, 0]
                            )
                        )
                    ),
                    "gripper": 0.0,
                },
                {
                    "t": 15,
                    "xyz": goal_position 
                    + np.array(
                        [
                            0 if obj_stand else place_offset,
                            0,
                            place_h if obj_stand else none_stand_place_h,
                        ]
                    ),
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                release_rotational_noise
                                if obj_stand
                                else [-np.pi / 2, np.pi / 2, 0]
                            )
                        )
                    ),
                    "gripper": 0.0,
                },
                {
                    "t": 10,
                    "xyz": goal_position
                    + np.array(
                        [
                            0 if obj_stand else place_offset,
                            0,
                            place_h if obj_stand else none_stand_place_h,
                        ]
                    ),
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                release_rotational_noise
                                if obj_stand
                                else [-np.pi / 2, np.pi / 2, 0]
                            )
                        )
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 15,
                    "xyz": goal_position
                    + np.array([0 if obj_stand else -0.112, 0, height4move])
                    + self.transitional_noise(),
                    "quat": transforms3d.euler.euler2quat(
                        *(
                            ee_init_euler
                            + (
                                release_rotational_noise
                                if obj_stand
                                else [-np.pi / 2, np.pi / 2, 0]
                            )
                            + self.rotational_noise()
                        )
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 15,
                    "xyz": ee_init_xyz,
                    "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                    "gripper": 0.09,
                },
            ]

        self.trajectory += [
            {
                "t": 15,
                "xyz": ee_init_xyz,
                "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                "gripper": 0.09,
            },
        ]