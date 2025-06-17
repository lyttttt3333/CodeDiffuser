import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.stow_object_utils import load_shelf, load_stow_obj, load_platform


class StowEnv(BaseSimulationEnv):
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
        self.offset_x=0.0
        self.offset_y=0.58
        self.offset_z=0.25
        self.table = self.create_table(
            table_height=0.6, table_half_size=self.table_half_size, 
            offset_z=self.offset_z, offset_y=self.offset_y,offset_x=self.offset_x
        )

        # Load shelf
        self.box = None
        self.manip_obj = load_stow_obj(self.scene, object_name=box, name=box, scale=np.array([0.3,0.95,0.8]))
        self.env_book_0 = load_stow_obj(self.scene, object_name=box, name="env_book_0",
                                        scale=np.array([0.4,0.7,0.7]),is_static=0)
        self.env_book_1 = load_stow_obj(self.scene, object_name=box, name="env_book_1",
                                        scale=np.array([0.3,0.65,0.6]),is_static=0)
        self.env_book_2 = load_stow_obj(self.scene, object_name="book_2", name="env_book_2",
                                        scale=np.array([0.7,0.65,0.6]),is_static=0)
        self.env_book_3 = load_stow_obj(self.scene, object_name="book_2", name="env_book_2",
                                        scale=np.array([0.7,0.65,0.6]),is_static=0)
        #self.platform = load_platform(self.scene)

        # set up workspace boundary
        # self.wkspc_half_w = 0.18
        # self.wkspc_half_l = 0.18

    def reset_env(self):
        ### shelf ###
        if self.box is None:
            self.box = load_shelf(self.scene)
            random_x = 0 * self.np_random.uniform(-0.15, 0.15) + 0.2
            # random_y = self.np_random.uniform(-0.05, 0.05) + 0.3
            y = 0.1
            shelf_pos = np.array([random_x, y, self.offset_z])
            # shelf_pos = np.array([0, 0, 0])
            shelf_ori = transforms3d.euler.euler2quat(np.pi / 2, 0, -np.pi / 2)
            shelf_pose = sapien.Pose(shelf_pos, shelf_ori)
            self.box.set_pose(shelf_pose)

        np.random.seed()

        ### book_1 ###
        random_x = np.random.uniform(-0.25,0.15)
        random_y = -0.18
        if random_x >= 0:
            random_rot =  np.random.uniform(-np.pi/4, -5*np.pi/3 * random_x)
        else:
            coin_flag = np.random.uniform(-1,1)
            if coin_flag > 0:
                random_rot =  np.random.uniform(-np.pi/4, 0)
            else:
                random_rot =  np.random.uniform(0, -2*np.pi/3 * random_x)
        box_pos = np.array([random_x, random_y, self.offset_z])
        self.z_rotation = random_rot
        box_ori = transforms3d.euler.euler2quat(0, -np.pi / 2, random_rot)
        box_pose = sapien.Pose(box_pos, box_ori)
        self.manip_obj.set_pose(box_pose)


        ### book on the shelf ###
        if True:
            random_rot = 0

            env_pos=shelf_pos+np.array([0.1,0.05,0.05])
            env_ori = transforms3d.euler.euler2quat(np.pi / 2, 1/3 * np.pi / 2, random_rot + 0*np.pi / 2)
            env_pose = sapien.Pose(env_pos, env_ori)
            self.env_book_0.set_pose(env_pose)

            env_pos=shelf_pos+np.array([0.01,0.05,0.05])
            env_ori = transforms3d.euler.euler2quat(np.pi / 2, 1/3 * np.pi / 2, random_rot + 0*np.pi / 2)
            env_pose = sapien.Pose(env_pos, env_ori)
            self.env_book_1.set_pose(env_pose)

            env_pos=shelf_pos+np.array([0.05,0.05,0.05])
            env_ori = transforms3d.euler.euler2quat(np.pi / 2, 1/3 * np.pi / 2, random_rot + 0*np.pi / 2)
            env_pose = sapien.Pose(env_pos, env_ori)
            self.env_book_2.set_pose(env_pose)

            env_pos=shelf_pos+np.array([0.07,0.05,0.05])
            env_ori = transforms3d.euler.euler2quat(np.pi / 2, 1/3 * np.pi / 2, random_rot + 0*np.pi / 2)
            env_pose = sapien.Pose(env_pos, env_ori)
            self.env_book_3.set_pose(env_pose)

            # env_pos=shelf_pos+np.array([0.25,0.05,0.25])
            # env_ori = transforms3d.euler.euler2quat(np.pi / 2, 1/3 * np.pi / 2, random_rot + 0*np.pi / 2)
            # env_pose = sapien.Pose(env_pos, env_ori)
            # self.platform.set_pose(env_pose)

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
                self.manip_obj.get_pose().to_transformation_matrix(),
                self.box.get_pose().to_transformation_matrix(),
            ]
        )
        return init_poses


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light

    env = StowEnv(use_ray_tracing=True, box="book_1")
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
