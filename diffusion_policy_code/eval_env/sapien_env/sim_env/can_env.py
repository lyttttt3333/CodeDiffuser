import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.object_utils import load_open_box, load_obj


class CanEnv(BaseSimulationEnv):
    def __init__(
        self,
        use_gui=True,
        frame_skip=5,
        object_scale=1,
        randomness_scale=1,
        friction=0.3,
        use_ray_tracing=False,
        manip_obj="cola",
        extra_manip_obj="pepsi",
        randomness_level="full",
        task_level_multimodality=False,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui=use_gui,
            frame_skip=frame_skip,
            use_ray_tracing=use_ray_tracing,
            **renderer_kwargs,
        )

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.friction = friction
        self.object_scale = object_scale
        # self.randomness_scale = randomness_scale
        self.randomness_level = randomness_level

        # Load table
        self.table = self.create_table(
            table_height=0.6, table_half_size=[0.35, 0.7, 0.025]
        )

        # Load object
        self.obj_list=["cola","pepsi"]
        self.manip_obj_name = manip_obj
        self.manip_obj = load_obj(self.scene, manip_obj, density=1000)
        self.ori_obj_pos = np.zeros(3)

        self.task_level_multimodality = task_level_multimodality
        if self.task_level_multimodality:
            self.extra_manip_obj_name = extra_manip_obj
            self.extra_manip_obj = load_obj(self.scene, extra_manip_obj, density=1000)
            self.extra_ori_obj_pos = np.zeros(3)

        # set up workspace boundary
        self.wkspc_half_w = 0.13
        self.wkspc_half_l = 0.13
        self.wkspc_offset_l = 0.05

        self.box_half_w = 0.045
        self.box_half_l = 0.045

        self.box_dist_thresh = 0.165
        self.can_dist_thresh = 0.12

        self.fixed_y = False
        self.fixed_box_y = 0.10
        self.fixed_can_y = -0.10

        self.box_ls = None
        if self.manip_obj_name == "cola":
            self.box_color = [0.5, 0.0, 0.0]
        elif self.manip_obj_name == "mtndew":
            self.box_color = [0.0, 0.5, 0.0]
        elif self.manip_obj_name == "pepsi":
            self.box_color = [0.0, 0.0, 0.5]
        else:
            raise NotImplementedError

        if self.task_level_multimodality:
            self.extra_box_ls = None
            if self.extra_manip_obj_name == "cola":
                self.extra_box_color = [0.5, 0.0, 0.0]
            elif self.extra_manip_obj_name == "mtndew":
                self.extra_box_color = [0.0, 0.5, 0.0]
            elif self.extra_manip_obj_name == "pepsi":
                self.extra_box_color = [0.0, 0.0, 0.5]
            else:
                raise NotImplementedError

    def reset_env(self):
        if self.randomness_level == "full":
            if not self.task_level_multimodality:
                if self.box_ls is None:
                    # Load box
                    self.box_pos = np.array([0.0, 0.0, 0.0])
                    self.box_ls = load_open_box(
                        self.scene,
                        self.renderer,
                        half_w=self.box_half_w,
                        half_l=self.box_half_l,
                        h=0.02,
                        floor_width=0.01,
                        origin=self.box_pos,
                        color=self.box_color,
                    )
                    self.box_pos = np.array(
                        [
                            self.np_random.uniform(
                                -self.wkspc_half_w,
                                self.wkspc_half_w,
                            ),
                            (
                                self.np_random.uniform(
                                    0 + self.box_half_l, self.wkspc_half_l
                                )
                                + self.wkspc_offset_l
                                if not self.fixed_y
                                else self.fixed_box_y
                            ),
                            0.0,
                        ]
                    )
                    self.box_ori = np.array([1, 0, 0, 0])
                    self.box_pose = sapien.Pose(self.box_pos, self.box_ori)
                    self.box_ls.set_pose(self.box_pose)
            else:
                if self.box_ls is None and self.extra_box_ls is None:
                    self.box_pos = np.array([0.0, 0.0, 0.0])
                    self.box_ls = load_open_box(
                        self.scene,
                        self.renderer,
                        half_w=self.box_half_w,
                        half_l=self.box_half_l,
                        h=0.02,
                        floor_width=0.01,
                        origin=self.box_pos,
                        color=self.box_color,
                    )
                    self.extra_box_pos = np.array([0.0, 0.0, 0.0])
                    self.extra_box_ls = load_open_box(
                        self.scene,
                        self.renderer,
                        half_w=self.box_half_w,
                        half_l=self.box_half_l,
                        h=0.02,
                        floor_width=0.01,
                        origin=self.extra_box_pos,
                        color=self.extra_box_color,
                    )
                    count = 0
                    while True:
                        count += 1
                        if count > 1000:
                            raise ValueError(
                                "Cannot find a valid position for extra box!"
                            )
                        pos = np.array(
                            [
                                self.np_random.uniform(
                                    -self.wkspc_half_w,
                                    self.wkspc_half_w,
                                ),
                                (
                                    self.np_random.uniform(
                                        0 + self.box_half_l, self.wkspc_half_l
                                    )
                                    + self.wkspc_offset_l
                                    if not self.fixed_y
                                    else self.fixed_box_y
                                ),
                                0.0,
                            ]
                        )
                        extra_pos = np.array(
                            [
                                self.np_random.uniform(
                                    -self.wkspc_half_w,
                                    self.wkspc_half_w,
                                ),
                                (
                                    self.np_random.uniform(
                                        0 + self.box_half_l, self.wkspc_half_l
                                    )
                                    + self.wkspc_offset_l
                                    if not self.fixed_y
                                    else self.fixed_box_y
                                ),
                                0.0,
                            ]
                        )
                        dist = np.linalg.norm(pos - extra_pos)
                        if dist > self.box_dist_thresh:
                            break
                    self.box_ori = np.array([1, 0, 0, 0])
                    self.box_pos = pos
                    self.box_pose = sapien.Pose(self.box_pos, self.box_ori)
                    self.box_ls.set_pose(self.box_pose)

                    self.extra_box_ori = np.array([1, 0, 0, 0])
                    self.extra_box_pos = extra_pos
                    self.extra_box_pose = sapien.Pose(
                        self.extra_box_pos, self.extra_box_ori
                    )
                    self.extra_box_ls.set_pose(self.extra_box_pose)
        else:
            raise NotImplementedError

        pose = self.generate_random_init_pose()
        self.manip_obj.set_pose(pose)
        self.ori_obj_pos = pose.p

        if self.task_level_multimodality:
            pose = self.generate_extra_random_init_pose()
            self.extra_manip_obj.set_pose(pose)
            self.extra_ori_obj_pos = pose.p

    def generate_random_init_pose(self):
        # select pos that is within workspace and not too close to the box
        if self.randomness_level == "full":
            count = 0
            while True:
                count += 1
                if count > 1000:
                    raise ValueError("Cannot find a valid position for object!")
                pos = np.array(
                    [
                        self.np_random.uniform(-self.wkspc_half_w, self.wkspc_half_w),
                        (
                            self.np_random.uniform(-self.wkspc_half_l, 0) + self.wkspc_offset_l
                            if not self.fixed_y
                            else self.fixed_can_y
                        ),
                    ]
                )
                if self.task_level_multimodality:
                    dist = np.linalg.norm(pos - self.box_pos[:2])
                    extra_dist = np.linalg.norm(pos - self.extra_box_pos[:2])
                    if (
                        dist > self.box_dist_thresh
                        and extra_dist > self.box_dist_thresh
                    ):
                        break
                else:
                    dist = np.linalg.norm(pos - self.box_pos[:2])
                    if dist > self.can_dist_thresh:
                        break
        else:
            raise NotImplementedError

        random_z_rotate = self.np_random.uniform(0, np.pi)
        orientation = transforms3d.euler.euler2quat(np.pi / 2, 0, random_z_rotate)
        position = np.array([pos[0], pos[1], 0.01])
        pose = sapien.Pose(position, orientation)
        return pose

    def generate_extra_random_init_pose(self):
        # select pos that is within workspace and not too close to the box
        if self.randomness_level == "full":
            count = 0
            while True:
                count += 1
                if count > 1000:
                    raise ValueError("Cannot find a valid position for extra object!")
                pos = np.array(
                    [
                        self.np_random.uniform(-self.wkspc_half_w, self.wkspc_half_w),
                        (
                            self.np_random.uniform(-self.wkspc_half_l, 0) + self.wkspc_offset_l
                            if not self.fixed_y
                            else self.fixed_can_y
                        ),
                    ]
                )
                dist = np.linalg.norm(pos - self.box_pos[:2])
                extra_dist = np.linalg.norm(pos - self.extra_box_pos[:2])
                dist_another_can = np.linalg.norm(pos - self.ori_obj_pos[:2])
                if (
                    dist > self.box_dist_thresh
                    and extra_dist > self.can_dist_thresh
                    and dist_another_can > self.can_dist_thresh
                ):
                    break
        else:
            raise NotImplementedError

        random_z_rotate = self.np_random.uniform(0, np.pi)
        orientation = transforms3d.euler.euler2quat(np.pi / 2, 0, random_z_rotate)
        position = np.array([pos[0], pos[1], 0.01])
        pose = sapien.Pose(position, orientation)
        return pose

    def get_init_poses(self):
        if self.task_level_multimodality:
            init_poses = np.stack(
                [
                    self.manip_obj.get_pose().to_transformation_matrix(),
                    self.box_ls.get_pose().to_transformation_matrix(),
                    self.extra_manip_obj.get_pose().to_transformation_matrix(),
                    self.extra_box_ls.get_pose().to_transformation_matrix(),
                ]
            )
        else:
            init_poses = np.stack(
                [
                    self.manip_obj.get_pose().to_transformation_matrix(),
                    self.box_ls.get_pose().to_transformation_matrix(),
                ]
            )
        return init_poses


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light

    task_level_multimodality = True

    obj_list = np.array(["cola", "mtndew", "pepsi"])
    obj_list = np.random.choice(obj_list, 2, replace=False)

    env = CanEnv(
        use_ray_tracing=False,
        manip_obj=obj_list[0],
        extra_manip_obj=obj_list[1],
        task_level_multimodality=task_level_multimodality,
    )
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == "__main__":
    env_test()
