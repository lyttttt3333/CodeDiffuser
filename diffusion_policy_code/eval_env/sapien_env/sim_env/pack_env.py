from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base_pack import BaseSimulationEnv
from sapien_env.utils.object_utils import load_obj, load_open_box


class PackEnv(BaseSimulationEnv):
    def __init__(
        self,
        use_gui: bool = True,
        frame_skip: int = 5,
        use_ray_tracing: bool = False,
        task_level_multimodality: bool = False,
        nine_pos_mode: bool = False,
        stand_mode: bool = False,
        simple_mode: bool = False,
        fix_pick: bool = False,
        num_obj_wait: int = 1,
        num_obj_done: Optional[int] = None,
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
        self.nine_pos_mode = nine_pos_mode
        self.stand_mode = stand_mode
        self.simple_mode = simple_mode
        self.fix_pick = fix_pick

        if self.fix_pick:
            assert num_obj_wait == 1

        if num_obj_done is None:
            if self.nine_pos_mode:
                num_obj_done = self.np_random.randint(0, 9 - num_obj_wait + 1)
            else:
                num_obj_done = self.np_random.randint(0, 4 - num_obj_wait + 1)

        if not task_level_multimodality:
            assert num_obj_wait == 1
        assert 0 <= num_obj_wait
        assert 0 <= num_obj_done
        if self.nine_pos_mode:
            assert num_obj_wait + num_obj_done <= 9
        else:
            assert num_obj_wait + num_obj_done <= 4

        self.task_level_multimodality = task_level_multimodality
        self.num_obj_wait = num_obj_wait
        self.num_obj_done = num_obj_done

        # Set up workspace boundary
        self.wkspc_half_w = 0.1
        self.wkspc_half_l = 0.1
        self.wkspc_offset_l = 0.025

        if self.nine_pos_mode:
            self.box_half_w = 0.12
            self.box_half_l = 0.12
        else:
            self.box_half_w = 0.1
            self.box_half_l = 0.1

        self.container_fixed_x = -0.18
        self.container_fixed_y = 0
        if nine_pos_mode:
            self.obj_fixed_dist = 0.19
        else:
            self.obj_fixed_dist = 0.18
        self.obj_min_dist_thresh = 0.12

        self.obj_offset_x = 0.02

        # if self.simple_mode:
        #     self.obj_fixed_x = 0.12

        if self.nine_pos_mode:
            obj_done_offset_dist = 0.08
        else:
            obj_done_offset_dist = 0.055
        if self.nine_pos_mode:
            self.obj_done_offset_list = np.array(
                [
                    [obj_done_offset_dist, obj_done_offset_dist, 0],
                    [obj_done_offset_dist, 0, 0],
                    [obj_done_offset_dist, -obj_done_offset_dist, 0],
                    [0, obj_done_offset_dist, 0],
                    [0, 0, 0],
                    [0, -obj_done_offset_dist, 0],
                    [-obj_done_offset_dist, obj_done_offset_dist, 0],
                    [-obj_done_offset_dist, 0, 0],
                    [-obj_done_offset_dist, -obj_done_offset_dist, 0],
                ]
            )
        else:
            self.obj_done_offset_list = np.array(
                [
                    [obj_done_offset_dist, obj_done_offset_dist, 0],
                    [obj_done_offset_dist, -obj_done_offset_dist, 0],
                    [-obj_done_offset_dist, obj_done_offset_dist, 0],
                    [-obj_done_offset_dist, -obj_done_offset_dist, 0],
                ]
            )

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)

        # Load table
        self.table = self.create_table(
            table_height=0.6, table_half_size=[0.35, 0.7, 0.025]
        )

        # Load container
        container_actor = load_open_box(
            self.scene,
            self.renderer,
            half_w=self.box_half_w,
            half_l=self.box_half_l,
            h=0.02,
            floor_width=0.01,
            origin=np.zeros(3),
        )
        container_pos = np.array(
            [self.container_fixed_x, self.container_fixed_y + self.wkspc_offset_l, 0]
        )
        container_ori = np.zeros(3)
        container_pose = sapien.Pose(
            container_pos, transforms3d.euler.euler2quat(*container_ori)
        )
        container_actor.set_pose(container_pose)
        self.container = {"actor": container_actor, "pose": container_pose}

    def reset_env(self) -> None:
        self.obj_wait = []
        self.obj_done = []
        # Load object
        obj_available = np.array(["cola", "mtndew", "pepsi"])
        obj_wait_list = self.np_random.choice(obj_available, self.num_obj_wait)
        obj_done_list = self.np_random.choice(obj_available, self.num_obj_done)

        success_flag = False
        while not success_flag:
            success_flag = True
            obj_pose_list = []
            for obj_name in obj_wait_list:
                position, euler, stand_flag, success = self.get_obj_wait_pose(
                    obj_pose_list
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
            self.obj_wait.append(
                {
                    "name": obj_name,
                    "actor": actor,
                    "position": obj_pose_list[idx]["position"],
                    "euler": obj_pose_list[idx]["euler"],
                    "stand": obj_pose_list[idx]["stand"],
                }
            )
        assert len(self.obj_wait) == self.num_obj_wait

        for idx, obj_name in enumerate(obj_done_list):
            actor = load_obj(self.scene, obj_name, density=1000)
            assert idx < len(self.obj_done_offset_list)
            position, euler = self.get_obj_done_pose(idx)
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
        assert len(self.obj_done) == self.num_obj_done

    def get_obj_wait_pose(
        self, obj_pose_list: list
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        count = 0
        done = False
        position = np.zeros(3)
        euler = np.zeros(3)
        stand_flag = (
            self.np_random.choice([True, False]) if not self.stand_mode else True
        )
        while not done:
            done = True
            count += 1
            if count > 1000:
                return position, euler, stand_flag, False

            if not self.fix_pick:
                pos = np.array(
                    [
                        self.np_random.uniform(0, 2 * self.wkspc_half_w)
                        + self.obj_offset_x
                        if not self.simple_mode
                        else self.np_random.choice(
                            [
                                0.3 * self.wkspc_half_w + self.obj_offset_x,
                                1.7 * self.wkspc_half_w + self.obj_offset_x,
                            ]
                        ),
                        self.np_random.uniform(-self.wkspc_half_l, self.wkspc_half_l)
                        + self.wkspc_offset_l
                        if not self.fix_pick
                        else self.wkspc_offset_l,
                    ]
                )
            else:
                pos = np.array(
                    [self.wkspc_half_w + self.obj_offset_x, self.wkspc_offset_l]
                )

            for obj in obj_pose_list:
                dist = np.linalg.norm(pos - obj["position"][:2])
                if dist < self.obj_min_dist_thresh:
                    done = False
                    break

            position = np.array([pos[0], pos[1], 0.01])
            euler = np.array(
                [
                    np.pi / 2 if stand_flag else np.pi,
                    0,
                    self.np_random.uniform(-np.pi / 2, np.pi / 2),
                ]
            )
        return position, euler, stand_flag, True

    def get_obj_done_pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        container_position = self.container["pose"].p + np.array([0, 0, 0.01])
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


def env_test():
    from constructor import add_default_scene_light
    from sapien.utils import Viewer

    task_level_multimodality = False
    nine_pos_mode = False
    stand_mode = True
    simple_mode = True
    num_obj_wait = 1
    num_obj_done = 3
    pause = True

    env = PackEnv(
        task_level_multimodality=task_level_multimodality,
        nine_pos_mode=nine_pos_mode,
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