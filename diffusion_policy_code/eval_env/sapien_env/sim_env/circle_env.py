import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.stow_object_utils import load_shelf, load_stow_obj


class CircleEnv(BaseSimulationEnv):
    def __init__(
        self,
        use_gui=True,
        frame_skip=5,
        # object_scale=1,
        # randomness_scale=1,
        # seed=None,
        use_ray_tracing=True,
        # randomness_level="full",
        box="book_1",
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
        # self.friction = friction
        # self.object_scale = object_scale
        # self.randomness_scale = randomness_scale
        # self.randomness_level = randomness_level

        # Load table
        self.table_half_size = [0.35, 0.7, 0.025]
        self.table = self.create_table(
            table_height=0.6, table_half_size=self.table_half_size
        )

        # Load shelf
        self.shelf = None
        self.box = load_stow_obj(self.scene, box)

        # set up workspace boundary
        # self.wkspc_half_w = 0.18
        # self.wkspc_half_l = 0.18

    def reset_env(self):
        # pass
        ### shelf ###
        if self.shelf is None:
            self.shelf = load_shelf(self.scene)
            random_x = self.np_random.uniform(-0.15, 0.15) + 0
            # random_y = self.np_random.uniform(-0.05, 0.05) + 0.3
            y = 0.3
            shelf_pos = np.array([random_x, y, 0])
            # shelf_pos = np.array([0, 0, 0])
            shelf_ori = transforms3d.euler.euler2quat(np.pi / 2, 0, -np.pi / 2)
            shelf_pose = sapien.Pose(shelf_pos, shelf_ori)
            self.shelf.set_pose(shelf_pose)

        ### book_1 ###
        random_x = self.np_random.uniform(-0.15, 0.15) + 0
        random_y = self.np_random.uniform(-0.05, 0.05) + 0
        random_rot = self.np_random.uniform(-0.1, 0.1)
        box_pos = np.array([random_x, random_y, 0.1])
        box_ori = transforms3d.euler.euler2quat(0, -np.pi / 2, random_rot + -np.pi / 2)
        box_pose = sapien.Pose(box_pos, box_ori)
        self.box.set_pose(box_pose)

        ### book_2 ###
        # box_pos = np.array([0.15, 0, 0.05])
        # box_ori = transforms3d.euler.euler2quat(np.pi / 2, 0, np.pi)
        # box_pose = sapien.Pose(box_pos, box_ori)
        # self.box.set_pose(box_pose)

        ### flakes_1 ###
        # box_pos = np.array([0.15, 0.15, 0.05])
        # box_ori = transforms3d.euler.euler2quat(np.pi / 2, 0, np.pi)
        # box_pose = sapien.Pose(box_pos, box_ori)
        # self.boxes[2].set_pose(box_pose)

        ### flakes_2 ###
        # box_pos = np.array([0.25, 0.15, 0.05])
        # box_ori = transforms3d.euler.euler2quat(np.pi / 2, 0, np.pi)
        # box_pose = sapien.Pose(box_pos, box_ori)
        # self.boxes[3].set_pose(box_pose)

    def get_init_poses(self):
        init_poses = np.stack(
            [
                self.box.get_pose().to_transformation_matrix(),
                self.shelf.get_pose().to_transformation_matrix(),
            ]
        )
        return init_poses
        pass


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light

    env = CircleEnv(use_ray_tracing=True)
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    viewer.set_fovy(2.0)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == "__main__":
    env_test()
