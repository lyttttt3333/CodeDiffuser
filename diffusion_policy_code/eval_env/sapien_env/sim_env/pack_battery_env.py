from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.object_utils import load_obj


class PackBatteryEnv(BaseSimulationEnv):
    def __init__(
        self,
        use_gui: bool = True,
        frame_skip: int = 5,
        use_ray_tracing: bool = False,
        task_level_multimodality: bool = False,
        stand_mode: bool = False,
        simple_mode: bool = False,
        fix_pick: bool = False,
        num_obj_wait: int = 1,
        num_slot_wait: int = 1,
        num_obj_done: Optional[int] = None,
        assign_num: Optional[bool] = None,
        seed: Optional[int] = None,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui=use_gui,
            frame_skip=frame_skip,
            use_ray_tracing=use_ray_tracing,
            seed=seed,
            **renderer_kwargs,
        )
        self.stand_mode = stand_mode
        self.simple_mode = simple_mode
        self.fix_pick = fix_pick

        # TODO: Implement simple_mode
        assert self.simple_mode is False

        if self.fix_pick:
            assert num_obj_wait == 1
            assert simple_mode is False

        if num_obj_done is None:
            num_obj_done = self.np_random.randint(0, 12 - num_obj_wait + 1)

        if not task_level_multimodality:
            assert num_obj_wait == 1
        assert 0 <= num_obj_wait
        assert num_obj_wait <= 8
        assert 0 <= num_obj_done
        assert num_obj_wait + num_obj_done <= 12
        assert num_slot_wait <= 12

        self.task_level_multimodality = task_level_multimodality

        self.num_slot_wait = num_slot_wait

        if assign_num is None:
            assign_num = True

        if assign_num:
            self.num_obj_wait = num_obj_wait
            self.num_obj_done = num_obj_done
        else:
            self.num_obj_done = seed % num_obj_done
            self.num_obj_wait = num_obj_wait

        # Set up workspace boundary
        self.wkspc_half_w = 0.08
        self.wkspc_half_l = 0.06
        self.wkspc_x = -0.01
        self.wkspc_y = -0.03

        self.container_fixed_x = -0.01
        self.container_fixed_y = 0.18
        self.obj_min_dist_thresh = 0.09

        obj_done_offset_dist = 0.042
        self.obj_done_offset_list = np.array(
            [
                [obj_done_offset_dist * 3 / 2, obj_done_offset_dist, 0],
                [obj_done_offset_dist / 2, obj_done_offset_dist, 0],
                [-obj_done_offset_dist / 2, obj_done_offset_dist, 0],
                [-obj_done_offset_dist * 3 / 2, obj_done_offset_dist, 0],
                [obj_done_offset_dist * 3 / 2, 0, 0],
                [obj_done_offset_dist / 2, 0, 0],
                [-obj_done_offset_dist / 2, 0, 0],
                [-obj_done_offset_dist * 3 / 2, 0, 0],
                [obj_done_offset_dist * 3 / 2, -obj_done_offset_dist, 0],
                [obj_done_offset_dist / 2, -obj_done_offset_dist, 0],
                [-obj_done_offset_dist / 2, -obj_done_offset_dist, 0],
                [-obj_done_offset_dist * 3 / 2, -obj_done_offset_dist, 0],
            ]
        )

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)

        # Load table
        self.table = self.create_table(
            table_height=0.6, table_half_size=[0.35, 0.7, 0.025], offset_y=0.3
        )

        # Load container
        container_actor = load_obj(
            self.scene, "battery_container", collision_shape="nonconvex", is_static=True
        )
        container_pos = np.array([self.container_fixed_x, self.container_fixed_y, 0])
        container_ori = np.array([np.pi / 2, 0, 0])
        container_pose = sapien.Pose(
            container_pos, transforms3d.euler.euler2quat(*container_ori)
        )
        container_actor.set_pose(container_pose)
        self.container = {"actor": container_actor, "pose": container_pose}

    def reset_env(self) -> None:
        self.obj_wait = []
        self.obj_done = []
        self.obj_layout = []
        # Load object
        obj_available = np.array(["battery_3", "battery_4", "battery_5"])
        obj_wait_list = self.np_random.choice(obj_available, self.num_obj_wait)
        # obj_wait_list[0] = "battery_3"
        obj_available = np.array(["battery_3", "battery_4", "battery_5"])

        success_flag = False
        while not success_flag:
            success_flag = True
            obj_pose_list = []
            for obj_name in obj_wait_list:
                position, euler, stand_flag, success = self.get_obj_wait_pose(
                    obj_pose_list, force_stand=True
                )
                if not success:
                    success_flag = False
                    break
                obj_pose_list.append(
                    {"position": position, "euler": euler, "stand": stand_flag}
                )
        for idx, obj_name in enumerate(obj_wait_list):
            actor = load_obj(self.scene, obj_name, density=5000)
            pose = sapien.Pose(
                obj_pose_list[idx]["position"],
                transforms3d.euler.euler2quat(*obj_pose_list[idx]["euler"]),
            )
            actor.set_pose(pose)
            if idx == 0:
                self.obj_wait.append(
                    {
                        "name": obj_name,
                        "actor": actor,
                        "position": obj_pose_list[idx]["position"],
                        "euler": obj_pose_list[idx]["euler"],
                        "stand": obj_pose_list[idx]["stand"],
                    }
                )
            self.obj_layout.append(
                {
                    "name": obj_name,
                    "actor": actor,
                    "position": obj_pose_list[idx]["position"],
                    "euler": obj_pose_list[idx]["euler"],
                    "stand": obj_pose_list[idx]["stand"],
                }
            )
        assert len(self.obj_wait) == 1

        candidate_index, free_list = self.get_done_pose_distribution(self.num_obj_done)
        self.free_list = free_list
        self.target_idx = self.np_random.choice(list(range(self.num_slot_wait)))

        print(candidate_index, self.target_idx)

        if self.target_idx in candidate_index:
            candidate_index.remove(self.target_idx)

        print(candidate_index, self.target_idx)

        obj_done_list = self.np_random.choice(obj_available, len(candidate_index))

        for idx, obj_name in enumerate(obj_done_list):
            actor = load_obj(self.scene, obj_name, density=1000)
            assert idx < len(self.obj_done_offset_list)
            position_idx = candidate_index[idx]
            assert idx < len(self.obj_done_offset_list)
            position, euler = self.get_obj_done_pose(position_idx)
            pose = sapien.Pose(
                position,
                transforms3d.euler.euler2quat(*euler),
            )
            actor.set_pose(pose)
            self.obj_done.append(
                {
                    "name": obj_name,
                    "actor": actor,
                    "position": position,
                    "euler": euler,
                }
            )

    def get_done_pose_distribution(self, num):
        index_pool = list(range(12))
        sel_index = self.np_random.choice(index_pool, num, replace=False).tolist()
        free_list = []
        for idx in index_pool:
            if idx in sel_index:
                pass
            else:
                free_list.append(idx)
        return sel_index, free_list

    def get_obj_wait_pose(
        self,
        obj_pose_list: list,
        force_stand=None,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        count = 0
        done = False
        position = np.zeros(3)
        euler = np.zeros(3)
        if force_stand is None:
            stand_flag = (
                self.np_random.choice([True, False]) if not self.stand_mode else True
            )
        else:
            stand_flag = force_stand
        while not done:
            done = True
            count += 1
            if count > 1000:
                return position, euler, stand_flag, False

            if not self.fix_pick:
                pos = np.array(
                    [
                        self.np_random.uniform(
                            -self.wkspc_half_w + self.wkspc_x,
                            self.wkspc_half_w + self.wkspc_x,
                        )
                        if not self.simple_mode
                        else self.np_random.choice(
                            [
                                -self.wkspc_half_w + self.wkspc_x,
                                self.wkspc_half_w + self.wkspc_x,
                            ]
                        ),
                        self.np_random.uniform(
                            -self.wkspc_half_l + self.wkspc_y,
                            self.wkspc_half_l + self.wkspc_y,
                        ),
                    ]
                )
            else:
                pos = np.array([self.wkspc_x, self.wkspc_y])

            for obj in obj_pose_list:
                dist = np.linalg.norm(pos - obj["position"][:2])
                if dist < self.obj_min_dist_thresh:
                    done = False
                    break

            position = np.array([pos[0], pos[1], 0.02])
            euler = np.array(
                [
                    np.pi / 2 if stand_flag else np.pi,
                    0,
                    self.np_random.uniform(-np.pi / 2, np.pi / 2),
                ]
            )
        return position, euler, stand_flag, True

    def get_obj_done_pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        container_position = self.container["pose"].p + np.array([0, 0, 0.02])
        position = container_position + self.obj_done_offset_list[index]
        euler = np.array([np.pi / 2, 0, self.np_random.uniform(0, np.pi)])
        return position, euler

    def get_init_poses(self) -> np.ndarray:
        init_poses = np.stack(
            [
                self.container["actor"].get_pose().to_transformation_matrix(),
                *[
                    obj["actor"].get_pose().to_transformation_matrix()
                    for obj in self.obj_wait
                ],
            ]
        )
        return init_poses

    def get_layout(self):
        pose_list = []
        for idx, item in enumerate(self.obj_layout):
            if idx == 0:
                position = item["position"]
                euler = item["euler"]
                pose = np.concatenate([position, euler], axis=-1)[None, :]
                pose_list.append(pose)
        if len(pose_list) != 0:
            target_layout = np.concatenate(pose_list, axis=0)
        else:
            target_layout = np.array([0])

        pose_list = []
        for idx, item in enumerate(self.obj_layout):
            if idx != 0:
                position = item["position"]
                euler = item["euler"]
                pose = np.concatenate([position, euler], axis=-1)[None, :]
                pose_list.append(pose)
        if len(pose_list) != 0:
            wait_layout = np.concatenate(pose_list, axis=0)
        else:
            wait_layout = np.array([0])

        pose_list = []
        for idx, item in enumerate(self.obj_done):
            position = item["position"]
            euler = item["euler"]
            pose = np.concatenate([position, euler], axis=-1)[None, :]
            pose_list.append(pose)
        if len(pose_list) != 0:
            done_layout = np.concatenate(pose_list, axis=0)
        else:
            done_layout = np.array([0])
        return {
            "target_layout": target_layout,
            "wait_layout": wait_layout,
            "done_layout": done_layout,
        }

    def get_goal(self):
        target_idx = self.np_random.choice(self.free_list)
        position, _ = self.get_obj_done_pose(target_idx)
        return position


def env_test():
    from sapien.utils import Viewer

    from sapien_env.sim_env.constructor import add_default_scene_light

    task_level_multimodality = True
    stand_mode = True
    simple_mode = False
    num_obj_wait = 1
    num_obj_done = 8
    pause = True

    env = PackBatteryEnv(
        task_level_multimodality=task_level_multimodality,
        stand_mode=stand_mode,
        simple_mode=simple_mode,
        num_obj_wait=num_obj_wait,
        num_obj_done=num_obj_done,
    )
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    viewer.set_fovy(1.2)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer
    if pause:
        viewer.toggle_pause(True)

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == "__main__":
    env_test()
