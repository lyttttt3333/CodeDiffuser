import numpy as np
import sapien.core as sapien
import transforms3d
from typing import Optional

from sapien_env.rl_env.pack_env import PackRLEnv
from sapien_env.utils.random_utils import np_random


class BasePolicy:
    def __init__(
        self,
        seed: Optional[int] = None,
        inject_noise: bool = False,
        velocity_scale: float = 1,
        use_cubic: bool = True,
    ):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.step_total = 0
        self.traj_length = 0
        self.trajectory = None
        self.progress = 0
        self.velocity_scale = velocity_scale
        self.velocity_noise_scale = 0.1
        self.transitional_noise_scale = 0.005
        self.rotational_noise_scale = 0.1
        self.use_cubic = use_cubic

        self.set_seed(seed)

    def interpolate(self, curr_waypoint: dict, next_waypoint: dict, t: int) -> tuple:
        t_frac = t / next_waypoint["t"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        q0 = curr_waypoint["xyz"]
        q1 = next_waypoint["xyz"]

        if self.use_cubic:
            v0 = curr_waypoint["xyz_velocity"]
            v1 = next_waypoint["xyz_velocity"]
            xyz = np.zeros(3)
            for i in range(3):
                a0, a1, a2, a3 = self._cubic(q0[i], q1[i], v0[i], v1[i], next_waypoint["t"])
                xyz[i] = a0 + a1 * t + a2 * t**2 + a3 * t**3
        else:
            xyz = q0 + (q1 - q0) * t_frac

        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def _cubic(self, q0: float, q1: float, v0: float, v1: float, T: int) -> tuple:
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

    def transitional_noise(self, none_pos= False) -> np.ndarray:
        x_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        if none_pos:
            y_noise = self.np_random.uniform(
                -self.transitional_noise_scale, 0
            )
        else:
            y_noise = self.np_random.uniform(
                -self.transitional_noise_scale, self.transitional_noise_scale
            )
        z_noise = self.np_random.uniform(
            -self.transitional_noise_scale, self.transitional_noise_scale
        )
        return np.array([x_noise, y_noise, z_noise])

    def rotational_noise(self) -> np.ndarray:
        z_noise = self.np_random.uniform(
            -self.rotational_noise_scale, self.rotational_noise_scale
        )
        return np.array([0, 0, z_noise])

    def single_trajectory(self, env: PackRLEnv, ee_link_pose: sapien.Pose) -> tuple:
        # generate trajectory at first timestep, then open-loop execution
        if self.step_total == 0:
            self.generate_trajectory(env, ee_link_pose)
            if self.use_cubic:
                self._inferVelocity()
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

    def _inferVelocity(self) -> None:
        num_waypoints = len(self.trajectory)
        mid_velocity = np.zeros((num_waypoints, 3))
        for idx in range(1, num_waypoints):
            mid_velocity[idx] = (
                self.trajectory[idx]["xyz"] - self.trajectory[idx - 1]["xyz"]
            ) / self.trajectory[idx]["t"]
            mid_velocity[idx] *= self.velocity_scale + self.np_random.uniform(
                -self.velocity_noise_scale, self.velocity_noise_scale
            )
        sign = np.sign(mid_velocity)

        self.trajectory[0]["xyz_velocity"] = np.zeros(3)
        for idx in range(1, num_waypoints - 1):
            xyz_v = np.zeros(3)
            for i in range(3):
                if sign[idx][i] != sign[idx + 1][i]:
                    xyz_v[i] = 0
                else:
                    xyz_v[i] = (mid_velocity[idx][i] + mid_velocity[idx + 1][i]) / 2
            self.trajectory[idx]["xyz_velocity"] = xyz_v
        self.trajectory[-1]["xyz_velocity"] = np.zeros(3)

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = np_random(seed)
        return [seed]

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed(seed)

    def generate_trajectory(self, env: PackRLEnv, ee_link_pose: sapien.Pose) -> None:
        raise NotImplementedError