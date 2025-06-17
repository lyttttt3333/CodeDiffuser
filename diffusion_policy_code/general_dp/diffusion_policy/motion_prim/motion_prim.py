import numpy as np
import transforms3d


def stow(env, params):
    assert len(params) == 6

    motion_prim_idx = params[0]

    push = np.empty(3)  # x, y, d
    push[0] = params[1]
    push[1] = params[2]
    push[2] = params[3]
    transport = np.empty(2)  # x, h
    transport[0] = params[4]
    transport[1] = params[5]

    shelf_y = 0.3
    half_len = 0.1
    half_wid = 0.08
    table_half_size = 0.35

    waypoints = None
    if motion_prim_idx < 0.5:
        waypoints = [
            {
                "t": 0,
                "xyz": env.palm_link.get_pose().p,
                "quat": env.palm_link.get_pose().q,
                "gripper": 0.09,
            },
            {
                "t": 20,
                "xyz": np.array([push[0], push[1], 0.3]),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, 0),
                "gripper": 0.05,
            },
            {
                "t": 40,
                "xyz": np.array([push[0], push[1], 0.12]),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, 0),
                "gripper": 0.05,
            },
            {
                "t": 80,
                "xyz": np.array(
                    [
                        push[0] + push[2],
                        push[1],
                        0.12,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, 0),
                "gripper": 0.05,
            },
            {
                "t": 100,
                "xyz": np.array(
                    [
                        push[0] + push[2],
                        push[1],
                        0.12,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, 0),
                "gripper": 0.05,
            },
            {
                "t": 120,
                "xyz": np.array([0, 0, 0.3]),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, 0),
                "gripper": 0.09,
            },
        ]
    elif motion_prim_idx < 1.5:
        waypoints = [
            {
                "t": 0,
                "xyz": np.array([0, 0, 0.3]),
                "quat": transforms3d.euler.euler2quat(np.pi, 0, 0),
                "gripper": 0.09,
            },
            {
                "t": 40,
                "xyz": np.array(
                    [
                        table_half_size + 0.05,
                        push[1] - 0.3,
                        0.3,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi, 0),
                "gripper": 0.09,
            },
            {
                "t": 80,
                "xyz": np.array(
                    [
                        table_half_size + 0.05,
                        push[1] - 0.3,
                        0.025,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi, 0),
                "gripper": 0.09,
            },
            {
                "t": 100,
                "xyz": np.array(
                    [
                        table_half_size + 0.05,
                        push[1] - half_wid - 0.05,
                        0.025,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi, 0),
                "gripper": 0.09,
            },
            {
                "t": 120,
                "xyz": np.array(
                    [
                        table_half_size + 0.05,
                        push[1] - half_wid - 0.05,
                        0.025,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi, 0),
                "gripper": 0.01,
            },
            {
                "t": 140,
                "xyz": np.array(
                    [
                        table_half_size + 0.05,
                        push[1] - half_wid - 0.05,
                        0.3,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi, 0),
                "gripper": 0.01,
            },
        ]
    else:
        waypoints = [
            {
                "t": 0,
                "xyz": np.array(
                    [
                        table_half_size + 0.05,
                        push[1] - half_wid - 0.05,
                        0.3,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi, 0),
                "gripper": 0.01,
            },
            {
                "t": 30,
                "xyz": np.array(
                    [
                        transport[0],
                        shelf_y - half_wid - 0.05,
                        transport[1] + 0.1,
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi / 2, 0),
                "gripper": 0.01,
            },
            {
                "t": 50,
                "xyz": np.array(
                    [
                        transport[0],
                        shelf_y - half_wid - 0.05,
                        transport[1],
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi / 2, 0),
                "gripper": 0.01,
            },
            {
                "t": 70,
                "xyz": np.array(
                    [
                        transport[0],
                        shelf_y - half_wid - 0.05,
                        transport[1],
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi / 2, 0),
                "gripper": 0.01,
            },
            {
                "t": 90,
                "xyz": np.array(
                    [
                        transport[0],
                        shelf_y - half_wid - 0.05,
                        transport[1],
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi / 2, 0),
                "gripper": 0.09,
            },
            {
                "t": 140,
                "xyz": np.array(
                    [
                        transport[0],
                        shelf_y - half_wid - 0.15,
                        transport[1],
                    ]
                ),
                "quat": transforms3d.euler.euler2quat(np.pi * 1.5, np.pi / 2, 0),
                "gripper": 0.09,
            },
        ]

    return waypoints


class MotionPrimitive:
    def __init__(self, task_name):
        if task_name == "stow":
            self.get_waypoints = stow
            self.total_motion_prims = 3
        self.step_count = 0
        self.waypoints = None

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
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self, env, params):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.waypoints = self.get_waypoints(env, params)

        if self.waypoints[0]["t"] == self.step_count:
            self.curr_waypoint = self.waypoints.pop(0)

        if len(self.waypoints) == 0:
            quit = True
            return None, quit
        else:
            quit = False

        next_waypoint = self.waypoints[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(
            self.curr_waypoint, next_waypoint, self.step_count
        )

        self.step_count += 1
        cartisen_action_dim = 6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim + grip_dim)
        eluer = transforms3d.euler.quat2euler(quat, axes="sxyz")
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper

        return cartisen_action, quit

    def get_trajectory(self, env, params):
        self.step_count = 0
        self.waypoints = None
        self.traj = []
        self.motion_prim_idx = params[0]
        # self.motion_prim_idx = 0
        while True:
            action, quit = self.single_trajectory(env, params)
            if quit:
                break
            else:
                self.traj.append(action)
        return self.traj, self.motion_prim_idx
