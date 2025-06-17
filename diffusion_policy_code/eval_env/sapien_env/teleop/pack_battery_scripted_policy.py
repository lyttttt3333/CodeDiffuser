from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d

from sapien_env.gui.teleop_gui_trossen import (
    META_CAMERA,
    TABLE_TOP_CAMERAS,
    VIEWER_CAMERA,
    GUIBase,
)
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



def main_env(
    count: int,
    headless: bool = False,
    pause: bool = False,
    task_level_multimodality: bool = False,
    stand_mode: bool = False,
    simple_mode: bool = False,
    fix_pick: bool = False,
    num_obj_wait: int = 1,
    num_obj_done: Optional[int] = None,
    img_path: Optional[str] = None,
    plot_traj: bool = False,
    video_path: Optional[str] = None,
):
    import cv2
    import matplotlib.pyplot as plt
    from diffusion_policy.common.kinematics_utils import KinHelper
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from sapien_env.utils.pose_utils import transform_action_from_world_to_robot

    kin_helper = KinHelper(robot_name="panda")

    env = PackBatteryRLEnv(
        use_gui=True,
        robot_name="panda",
        frame_skip=10,
        task_level_multimodality=task_level_multimodality,
        stand_mode=stand_mode,
        simple_mode=simple_mode,
        fix_pick=fix_pick,
        num_obj_wait=num_obj_wait,
        num_obj_done=num_obj_done,
        seed=count,
    )
    env.reset()
    arm_dof = env.arm_dof
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=headless)
    for _, params in TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.create_camera(**META_CAMERA, meta=True)
    if not gui.headless:
        gui.viewer.set_camera_xyz(**VIEWER_CAMERA["position"])
        gui.viewer.set_camera_rpy(**VIEWER_CAMERA["rotation"])
        gui.viewer.set_fovy(1.2)
    scene = env.scene
    scene.step()

    if pause:
        viewer = gui.viewer
        viewer.toggle_pause(True)

    if img_path is not None:
        gui.render()
        picture = gui.take_meta_view()

    if video_path is not None:
        print(f"Recording video to {video_path}")
        video = cv2.VideoWriter(
            video_path + f"/sapien_{count}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (640 * 2, 480 * 2),
        )
    scripted_policy = SingleArmPolicy(
        seed=count, inject_noise=False, velocity_scale=1, use_cubic=True
    )

    if not gui.headless:
        from tqdm import tqdm

        pbar = tqdm(total=100, colour="green", ascii=" 123456789>", desc="Test Rollout")

    trajectory_record = None
    while True:
        if img_path is not None and plot_traj is False:
            reward = 1
            break
        action = np.zeros(arm_dof + 1)
        cartisen_action, _, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose()
        )

        if not gui.headless:
            pbar.n = round(scripted_policy.progress * 100, 1)
            pbar.refresh()
        if quit:
            break
        if trajectory_record is None:
            trajectory_record = np.array([cartisen_action])
        else:
            trajectory_record = np.append(
                trajectory_record, np.array([cartisen_action]), axis=0
            )
        cartisen_action_in_rob = transform_action_from_world_to_robot(
            cartisen_action, env.robot.get_pose()
        )
        action[:arm_dof] = kin_helper.compute_ik_sapien(
            env.robot.get_qpos()[:], cartisen_action_in_rob
        )[:arm_dof]
        action[arm_dof:] = cartisen_action_in_rob[6]

        obs, reward, done, _ = env.step(action[: arm_dof + 1])
        gui.render()

        if video_path is not None:
            image = gui.take_meta_view()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)

    if img_path is not None:
        plt.imshow(picture)
        plt.axis("off")
        if plot_traj:
            points_3d = trajectory_record[:, :3]
            points = gui.points_in_meta_view(points_3d)
            plt.scatter(
                points[:, 0],
                points[:, 1],
                c=range(points.shape[0]),
                cmap="viridis",
                s=10,
            )
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cbar = plt.colorbar(cax=cax)
            cbar.ax.tick_params(labelsize=6)
        plt.tight_layout(pad=0)
        plt.savefig(f"{img_path}/sapien_{count}.png", dpi=600)
        plt.clf()
    if video_path is not None:
        video.release()
    if not gui.headless:
        pbar.close()
        gui.viewer.close()
    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
    emoji = "✅" if reward == 1 else "❌"
    print(f"{txt_color}{emoji} reward: {reward * 100:.2f}%{bcolors.ENDC}")
    env.close()
    return reward


if __name__ == "__main__":
    test_start = 0
    test_end = 20
    # img_path = "sapien_env/sapien_env/teleop/images"
    img_path = None
    plot_traj = False
    task_level_multimodality = True
    stand_mode = True
    simple_mode = False
    fix_pick = False

    num_obj_wait = 3
    num_obj_done = None
    headless = False
    pause = False
    # video_path = "sapien_env/sapien_env/teleop/videos"
    video_path = None

    count = test_start
    cumulative_reward = 0
    total_num_tests = test_end - test_start
    fail_idx = []
    import os

    from sapien_env.utils.my_utils import bcolors

    while count < test_end:
        term_size = os.get_terminal_size()
        print(f"{bcolors.OKBLUE}-{bcolors.ENDC}" * term_size.columns)
        print(f"{bcolors.OKBLUE}Test {count}{bcolors.ENDC}")
        reward = main_env(
            count,
            task_level_multimodality=task_level_multimodality,
            stand_mode=stand_mode,
            simple_mode=simple_mode,
            fix_pick=fix_pick,
            num_obj_wait=num_obj_wait,
            num_obj_done=num_obj_done,
            img_path=img_path,
            headless=headless,
            pause=pause,
            plot_traj=plot_traj,
            video_path=video_path,
        )
        cumulative_reward += reward
        if reward < 1:
            fail_idx.append(count)
        if len(fail_idx) > 0:
            print(f"{bcolors.FAIL}Failed tests: {fail_idx}{bcolors.ENDC}")
        count += 1
    if cumulative_reward == total_num_tests:
        print(f"{bcolors.OKGREEN}All tests passed!{bcolors.ENDC}")
    else:
        print(
            f"{bcolors.FAIL}{total_num_tests - int(cumulative_reward)} tests failed!{bcolors.ENDC}"
        )
