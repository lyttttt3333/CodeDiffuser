import numpy as np
import sapien.core as sapien
import transforms3d

import IPython

e = IPython.embed

from sapien_env.gui.teleop_gui_trossen import (
    GUIBase,
    VIEWER_CAMERA,
    META_CAMERA,
    TABLE_TOP_CAMERAS,
)
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.rl_env.stow_env import StowRLEnv
from sapien_env.utils.random_utils import np_random


class SingleArmPolicy:
    def __init__(self, seed=None, inject_noise=False, velocity_scale=1):
        self.seed(seed)

        self.inject_noise = inject_noise
        self.step_count = 0
        self.step_total = 0
        self.traj_length = 0
        self.trajectory = None
        self.progress = 0
        self.velocity_scale = velocity_scale
        self.transitional_noise_scale = 0.005
        self.rotational_noise_scale = 0.3

    def interpolate(self, curr_waypoint, next_waypoint, t):
        t_frac = t / next_waypoint["t"]
        # curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        # next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]

        # xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac

        q0 = curr_waypoint["xyz"]
        v0 = curr_waypoint["xyz_velocity"]

        q1 = next_waypoint["xyz"]
        v1 = next_waypoint["xyz_velocity"]

        xyz = np.zeros(3)
        for i in range(3):
            a0, a1, a2, a3 = self._cubic(q0[i], q1[i], v0[i], v1[i], next_waypoint["t"])
            xyz[i] = a0 + a1 * t + a2 * t**2 + a3 * t**3

        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def _cubic(self, q0, q1, v0, v1, T):
        """
        :param: q0: float
            the first data point
        :param: q1: float
            the second data point
        :param: v0: float
            the velocity of the first data point
        :param: v1: float
            the velocity of the second data point
        :param: t0: float
            the time of the first data point
        :param: t1: float
            the time of the second data point
        """
        try:
            abs(T) < 1e-6
        except ValueError:
            print("t0 and t1 must be different")

        h = q1 - q0

        a0 = q0
        a1 = v0
        a2 = (3 * h - (2 * v0 + v1) * T) / (T**2)
        a3 = (-2 * h + (v0 + v1) * T) / (T**3)
        return a0, a1, a2, a3

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

    def rotational_noise(self):
        z_noise = self.np_random.uniform(
            -self.rotational_noise_scale, self.rotational_noise_scale
        )
        return np.array([0, 0, z_noise])

    def single_trajectory(self, env, ee_link_pose, mode="straight", branch = None):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_total == 0:
            self.generate_trajectory(env, ee_link_pose, mode)
            if self.trajectory[0]["t"] != 0:
                raise ValueError("First waypoint must have t = 0")
            self.traj_length = np.sum([waypoint["t"] for waypoint in self.trajectory])

        if self.step_count == self.trajectory[0]["t"]:
            self.curr_waypoint = self.trajectory.pop(0)
            self.step_count = 0

        if len(self.trajectory) == 0:
            return None, self.curr_waypoint, True

        self.next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(
            self.curr_waypoint, self.next_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        self.step_total += 1
        cartisen_action_dim = 6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim + grip_dim)
        eluer = transforms3d.euler.quat2euler(quat, axes="sxyz")
        cartisen_action[:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper

        self.progress = self.step_total / self.traj_length
        # print(f"Progress: {self.progress * 100:.2f}%")

        return cartisen_action, self.next_waypoint, False

    def _inferVelocity(self):
        num_waypoints = len(self.trajectory)
        mid_velocity = np.zeros((num_waypoints, 3))
        for idx in range(1, num_waypoints):
            mid_velocity[idx] = (
                self.trajectory[idx]["xyz"] - self.trajectory[idx - 1]["xyz"]
            ) / self.trajectory[idx]["t"]
            mid_velocity[idx] *= self.velocity_scale + self.np_random.uniform(-0.2, 0.2)
        # sign = np.sign(mid_velocity)

        self.trajectory[0]["xyz_velocity"] = np.zeros(3)
        for idx in range(1, num_waypoints - 1):
            xyz_v = np.zeros(3)
            for i in range(3):
                # if sign[idx][i] != sign[idx + 1][i]:
                #     xyz_v[i] = 0.0
                # else:
                xyz_v[i] = (mid_velocity[idx][i] + mid_velocity[idx + 1][i]) / 2
            self.trajectory[idx]["xyz_velocity"] = xyz_v
        self.trajectory[-1]["xyz_velocity"] = np.zeros(3)

    # FIXME: abstract this to a base class
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def set_seed(self, seed=None):
        self.seed(seed)

    def generate_trajectory(self, env: StowRLEnv, ee_link_pose, mode="straight"):
        init_position = env.manip_obj.get_pose().p
        # init_rotation_quat = env.manip_obj.get_pose().q
        # init_rotation_eular = transforms3d.euler.quat2euler(init_rotation_quat)
        # print(init_rotation_eular)
        goal_position = env.box.get_pose().p
        #goal_position = np.array([0.09221713, 0.05  ,     0.25      ])
        init_y=init_position[1]

        if env.manip_obj_name == "book_1":
            grasp_h = 0.1
            place_h = 0.19
        elif env.manip_obj_name == "mtndew":
            grasp_h = 0.17
            place_h = 0.19
        elif env.manip_obj_name == "pepsi":
            grasp_h = 0.14
            place_h = 0.19
        else:
            raise NotImplementedError

        if env.task_level_multimodality:
            if env.extra_manip_obj_name == "cola":
                extra_grasp_h = 0.12
                extra_place_h = 0.19
            elif env.extra_manip_obj_name == "mtndew":
                extra_grasp_h = 0.17
                extra_place_h = 0.19
            elif env.extra_manip_obj_name == "pepsi":
                extra_grasp_h = 0.14
                extra_place_h = 0.19
            else:
                raise NotImplementedError

        height4move = 0.3
        THETA = - env.z_rotation
        DELTA_X = 0.2*np.sin(THETA)
        DELTA_Y = 0.2*np.cos(THETA)
        OFFSET_X = 0.07*np.sin(THETA)
        OFFSET_Y = 0.07*np.cos(THETA)
        OFFSET_Z = 0.022
        

        if mode == "straight":
            ee_init_xyz = ee_link_pose.p
            ee_init_euler = np.array([np.pi, 0, np.pi / 2])
            push_euler = np.array([np.pi, 0, 2*np.pi / 2])
            pick_euler = np.array([np.pi/2, 0, np.pi]) - np.array([0, 0, THETA])
            place_euler = np.array([np.pi, 0, np.pi/2])
            grasp_rotational_noise = self.rotational_noise()
            release_rotational_noise = self.rotational_noise()
            
            self.trajectory = [
                {
                    "t": 0,
                    "xyz": ee_init_xyz,
                    "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                    "gripper": 0.09,
                },
                # {
                #     "t": 15,
                #     "xyz": init_position
                #     + np.array([0.25, 0, height4move]),
                #     "quat": transforms3d.euler.euler2quat(
                #         *(push_euler)
                #     ),
                #     "gripper": 0.09,
                # },
                # {
                #     "t": 15,
                #     "xyz": init_position
                #     + np.array([0.05, 0, grasp_h]),
                #     "quat": transforms3d.euler.euler2quat(
                #         *(push_euler )
                #     ),
                #     "gripper": 0.09,
                # },
                # {
                #     "t": 15,
                #     "xyz": init_position+np.array([-0.15, PUSH_Y, grasp_h]),
                #     "quat": transforms3d.euler.euler2quat(
                #         *(push_euler )
                #     ),
                #     "gripper": 0.09,
                # },
                # {
                #     "t": 3,
                #     "xyz": init_position+np.array([-0.1, PUSH_Y, grasp_h]),
                #     "quat": transforms3d.euler.euler2quat(
                #         *(push_euler)
                #     ),
                #     "gripper": 0.09,
                # },
                # {
                #     "t": 15,
                #     "xyz": init_position+np.array([-0.15, PUSH_Y, height4move]),
                #     "quat": transforms3d.euler.euler2quat(
                #         *(push_euler )
                #     ),
                #     "gripper": 0.09,
                # },
                ########################################################################
                ### change to pick phase
                ########################################################################
                {
                    "t": 15,
                    "xyz": init_position+np.array([-OFFSET_X - DELTA_X, -OFFSET_Y - DELTA_Y, height4move]),
                    "quat": transforms3d.euler.euler2quat(
                        *(ee_init_euler)
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 15,
                    "xyz": init_position+np.array([-OFFSET_X - DELTA_X, -OFFSET_Y- DELTA_Y, OFFSET_Z]),
                    "quat": transforms3d.euler.euler2quat(
                        *(pick_euler)
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 15,
                    "xyz": init_position+np.array([-OFFSET_X, -OFFSET_Y, OFFSET_Z]),
                    "quat": transforms3d.euler.euler2quat(
                        *(pick_euler)
                    ),
                    "gripper": 0.09,
                },
                {
                    "t": 15,
                    "xyz": init_position+np.array([-OFFSET_X, -OFFSET_Y, OFFSET_Z]),
                    "quat": transforms3d.euler.euler2quat(
                        *(pick_euler)
                    ),
                    "gripper": 0.001,
                },
                {
                    "t": 15,
                    "xyz": init_position+np.array([-OFFSET_X, -OFFSET_Y, OFFSET_Z]),
                    "quat": transforms3d.euler.euler2quat(
                        *(pick_euler)
                    ),
                    "gripper": 0.001,
                },
                {
                    "t": 15,
                    "xyz": init_position+np.array([-OFFSET_X, -OFFSET_Y, height4move]),
                    "quat": transforms3d.euler.euler2quat(
                        *(pick_euler)
                    ),
                    "gripper": 0.001,
                },
                # ########################################################################
                # ### change to place phase
                # ########################################################################
                {
                    "t": 15,
                    "xyz": goal_position
                    + np.array([-0.05, -0.2, height4move]),
                    "quat": transforms3d.euler.euler2quat(
                        *(place_euler)
                    ),
                    "gripper": 0.005,
                },
                {
                    "t": 15,
                    "xyz": goal_position
                    + np.array([-0.05, 0.03, height4move]),
                    "quat": transforms3d.euler.euler2quat(
                        *(place_euler)
                    ),
                    "gripper": 0.005,
                },
                {
                    "t": 15,
                    "xyz": goal_position
                    + np.array([0.1, 0.03, height4move]),
                    "quat": transforms3d.euler.euler2quat(
                        *(place_euler)
                    ),
                    "gripper": 0.005,
                },
                {
                    "t": 15,
                    "xyz": goal_position
                    + np.array([0.1, 0.03, height4move]),
                    "quat": transforms3d.euler.euler2quat(
                        *(place_euler)
                    ),
                    "gripper": 0.09,
                },
            ]
            if not env.task_level_multimodality:
                if False:
                    self.trajectory += [
                        {
                            "t": 15,
                            "xyz": ee_init_xyz,
                            "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                            "gripper": 0.09,
                        },
                    ]
            else:
                print("#########################################")
                init_position = env.extra_manip_obj.get_pose().p
                goal_position = env.extra_box_pos
                grasp_rotational_noise = self.rotational_noise()
                release_rotational_noise = self.rotational_noise()
                self.trajectory += [
                    {
                        "t": 15,
                        "xyz": init_position
                        + np.array([0, 0, height4move] + self.transitional_noise()),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + self.rotational_noise())
                        ),
                        "gripper": 0.09,
                    },
                    {
                        "t": 15,
                        "xyz": init_position
                        + np.array([0, 0, extra_grasp_h])
                        + self.transitional_noise(),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + grasp_rotational_noise)
                        ),
                        "gripper": 0.09,
                    },
                    {
                        "t": 3,
                        "xyz": init_position + np.array([0, 0, extra_grasp_h]),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + grasp_rotational_noise)
                        ),
                        "gripper": 0.09,
                    },
                    {
                        "t": 15,
                        "xyz": init_position + np.array([0, 0, extra_grasp_h]),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + grasp_rotational_noise)
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 3,
                        "xyz": init_position + np.array([0, 0, extra_grasp_h]),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + grasp_rotational_noise)
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 15,
                        "xyz": init_position
                        + np.array([0, 0, height4move])
                        + self.transitional_noise(),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + self.rotational_noise())
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 30,
                        "xyz": goal_position
                        + np.array([0, 0, height4move])
                        + self.transitional_noise(),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + self.rotational_noise())
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 15,
                        "xyz": goal_position
                        + np.array([0, 0, extra_place_h])
                        + self.transitional_noise(),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + release_rotational_noise)
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 3,
                        "xyz": goal_position + np.array([0, 0, extra_place_h]),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + release_rotational_noise)
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 15,
                        "xyz": goal_position + np.array([0, 0, extra_place_h]),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + release_rotational_noise)
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 3,
                        "xyz": goal_position + np.array([0, 0, extra_place_h]),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + release_rotational_noise)
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 15,
                        "xyz": goal_position
                        + np.array([0, 0, height4move])
                        + self.transitional_noise(),
                        "quat": transforms3d.euler.euler2quat(
                            *(ee_init_euler + self.rotational_noise())
                        ),
                        "gripper": 0.01,
                    },
                    {
                        "t": 15,
                        "xyz": ee_init_xyz,
                        "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                        "gripper": 0.01,
                    },
                    {
                        "t": 15,
                        "xyz": ee_init_xyz,
                        "quat": transforms3d.euler.euler2quat(*ee_init_euler),
                        "gripper": 0.09,
                    },
                ]
        else:
            raise NotImplementedError

        self._inferVelocity()


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


def main_env(
    count,
    headless=False,
    pause=False,
    task_level_multimodality=False,
    img_path=None,
    plot_traj=False,
):
    from diffusion_policy.common.kinematics_utils import KinHelper
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    kin_helper = KinHelper(robot_name="panda")

    # obj_list = np.array(["cola", "mtndew", "pepsi"])
    obj_list = np.array(["cola", "pepsi"])
    obj_list = np.random.choice(obj_list, 2, replace=False)

    env = CanRLEnv(
        manip_obj=obj_list[0],
        extra_manip_obj=obj_list[1],
        use_gui=True,
        robot_name="panda",
        frame_skip=10,
        use_ray_tracing=False,
        task_level_multimodality=task_level_multimodality,
    )
    env.seed(count)
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
        gui.viewer.set_fovy(1)
    scene = env.scene
    scene.step()

    if pause:
        viewer = gui.viewer
        viewer.toggle_pause(True)

    if img_path is not None:
        gui.render()
        picture = gui.take_meta_view()

    scripted_policy = SingleArmPolicy(seed=count, inject_noise=False, velocity_scale=1)

    if not gui.headless:
        from tqdm import tqdm

        pbar = tqdm(total=100, colour="green", ascii=" 123456789>", desc="Test Rollout")

    trajectory_record = None
    while True:
        action = np.zeros(arm_dof + 1)
        cartisen_action, _, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose(), mode="straight"
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

    # plt.plot(trajectory_record[:, 0], label="x")
    # plt.plot(trajectory_record[:, 1], label="y")
    # plt.plot(trajectory_record[:, 2], label="z")
    # plt.legend()
    # plt.show()

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
    if not gui.headless:
        pbar.close()
        gui.viewer.close()
    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
    emoji = "✅" if reward == 1 else "❌"
    print(f"{txt_color}{emoji} reward: {reward}{bcolors.ENDC}")
    env.close()
    return reward


if __name__ == "__main__":
    test_start = 0
    test_end = 300
    visualize_randomness = True
    # visualize_randomness = False
    plot_traj = True
    # plot_traj = False
    task_level_multimodality = True
    # task_level_multimodality = False
    # headless = True
    headless = False
    # pause = True
    pause = False

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
        if visualize_randomness:
            img_path = "sapien_env/sapien_env/teleop/images"
        else:
            img_path = None
        reward = main_env(
            count,
            task_level_multimodality=task_level_multimodality,
            img_path=img_path,
            headless=headless,
            pause=pause,
            plot_traj=plot_traj,
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
