import numpy as np
import transforms3d
import sapien.core as sapien

import IPython

e = IPython.embed

# from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.gui.teleop_gui_trossen import (
    GUIBase,
    DEFAULT_TABLE_TOP_CAMERAS,
    YX_TABLE_TOP_CAMERAS,
)
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.rl_env.circle_env import CircleRLEnv


class SingleArmPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None
        self.progress = 0.0
        self.total_frame_len = -1

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        # curr_motion_prim = curr_waypoint["motion_prim"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        # next_motion_prim = next_waypoint["motion_prim"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        # motion_prim = curr_motion_prim
        return xyz, quat, gripper

    def single_trajectory(self, env, ee_link_pose, mode="straight"):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose, mode)

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

        assert self.total_frame_len >= 0
        self.progress = self.step_count / self.total_frame_len
        # print(f"progress: {self.progress:.1%}")

        return cartisen_action, next_waypoint, quit

    def generate_trajectory(self, env: CircleRLEnv, ee_link_pose, mode="straight"):
        # box_pos = env.box.get_pose().p
        # shelf_pos = env.shelf.get_pose().p
        # shelf_y = shelf_pos[1]

        # if env.manip_obj == "book_1":
        #     half_len = 0.1
        #     half_wid = 0.08
        # elif env.manip_obj == "book_2":
        #     raise NotImplementedError
        # elif env.manip_obj == "flakes_1":
        #     raise NotImplementedError
        # elif env.manip_obj == "flakes_2":
        #     raise NotImplementedError
        # else:
        #     raise NotImplementedError

        # push = np.empty(3)  # x, y, d
        # push[0] = box_pos[0] - 0.1
        # push[1] = box_pos[1]
        # push[2] = env.table_half_size[0] - push[0] - half_len - 0.01
        # transport = np.empty(2)  # x, h
        # transport[0] = shelf_pos[0]
        # transport[1] = half_len * 2 + 0.001
        # transport["d"] = 0
        # transport["theta"] = 0

        # print(f"push: {push}")
        # print(f"transport: {transport}")
        # self.motion_prim = np.concatenate((push, transport))
        # print(f"motion_prim: {self.motion_prim}")

        if mode == "straight":
            self.trajectory = [
                {
                    "t": 0,
                    "xyz": ee_link_pose.p,
                    "quat": ee_link_pose.q,
                    "gripper": 0.01,
                }
                # {
                #     "t": 25,
                #     "xyz": np.array([0, 0, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 75,
                #     "xyz": np.array([0, 0.1, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 100,
                #     "xyz": np.array([0.2, 0.1, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 150,
                #     "xyz": np.array([-0.2, 0.1, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 200,
                #     "xyz": np.array([0.2, 0.1, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 250,
                #     "xyz": np.array([-0.2, 0.1, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 275,
                #     "xyz": np.array([0, 0.1, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 300,
                #     "xyz": np.array([0, 0.2, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
                # {
                #     "t": 350,
                #     "xyz": np.array([0, 0.2, 0.3]),
                #     "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                #     "gripper": 0.01,
                # },
            ]
            circle_traj, circle_ending = circle_path(0, 0.05, 0.15, 20, 25, 140 * 3, 3)
            self.trajectory = self.trajectory + circle_traj
            self.trajectory.append(self.trajectory[-1].copy())
            self.trajectory[-1]["t"] = circle_ending + 25
            self.trajectory[-1]["xyz"] = self.trajectory[-1]["xyz"] + np.array([0, 0.20, 0])
            self.total_frame_len = self.trajectory[-1]["t"]
        else:
            raise NotImplementedError

def circle_path(center_x, center_y, radius, num_points_per_circle, start_time, total_time, circle_num):
    waypoints = []
    for i in range(num_points_per_circle * circle_num + 1):
        angle = 2 * np.pi * i / num_points_per_circle
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        waypoints.append(
            {
                "t": start_time + i * total_time / (num_points_per_circle * circle_num),
                "xyz": np.array([x, y, 0.3]),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2),
                "gripper": 0.01,
            }
        )
    return waypoints, start_time + total_time

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


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def main_env(count, img_path=None):
    from diffusion_policy.common.kinematics_utils import KinHelper

    kin_helper = KinHelper(robot_name="panda")
    # max_timesteps = 400
    # num_episodes = 50
    # onscreen = True
    # success = 0
    # success_rate = 0
    env = CircleRLEnv(
        manip_obj="book_1",
        use_gui=True,
        robot_name="panda",
        frame_skip=10,
        use_ray_tracing=False,
    )
    env.seed(count)
    env.reset()
    arm_dof = env.arm_dof
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=False)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if "rotation" in params:
            gui.create_camera_from_pos_rot(**params)
        else:
            gui.create_camera(**params)
    if not gui.headless:
        gui.viewer.set_camera_rpy(r=0, p=-1.2, y=np.pi / 2 - 0.65)
        gui.viewer.set_camera_xyz(x=-0.1, y=0.3, z=0.7)
    scene = env.scene
    scene.step()

    # viewer = gui.viewer
    # viewer.toggle_pause(True)

    if not gui.headless and img_path is not None:
        gui.render()
        picture = gui.viewer.window.get_float_texture("Color")
        from PIL import Image

        picture = (picture.clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(picture, mode="RGBA").save(
            img_path + f"/sapien_screenshot_{count}.png"
        )

    scripted_policy = SingleArmPolicy()
    # last_motion_prim = -1
    while True:
        action = np.zeros(arm_dof + 1)
        cartisen_action, _, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose(), mode="straight"
        )
        # transform cartisen_action from robot to world frame
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
        gui.render()

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
    if not gui.headless:
        gui.viewer.close()
    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
    emoji = "✅" if reward == 1 else "❌"
    print(f"{txt_color}{emoji} reward: {reward}{bcolors.ENDC}")
    env.close()
    return reward


if __name__ == "__main__":
    test_start = 0
    test_end = 60

    count = test_start
    cumulative_reward = 0
    total_num_tests = test_end - test_start
    while count < test_end:
        print(f"{bcolors.OKBLUE}-------------------{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Test {count}{bcolors.ENDC}")
        cumulative_reward += main_env(
            count, img_path="sapien_env/sapien_env/teleop/images"
        )
        count += 1
    if cumulative_reward == total_num_tests:
        print(f"{bcolors.OKGREEN}All tests passed!{bcolors.ENDC}")
    else:
        print(
            f"{bcolors.FAIL}{total_num_tests - int(cumulative_reward)} tests failed!{bcolors.ENDC}"
        )
