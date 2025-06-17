import gym
import numpy as np
import cv2
import copy
import open3d as o3d

from d3fields.utils.draw_utils import np2o3d
from diffusion_policy.real_world.video_recorder import VideoRecorder


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recoder: VideoRecorder,
        mode="rgb_array",
        file_path=None,
        real_time_vis=False,
        view_ctrl_info=None,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        self.mode = mode
        self.render_kwargs = kwargs
        self.file_path = file_path
        self.video_recoder = video_recoder
        self.real_time_vis = real_time_vis
        self.vis_flag = True
        self.visualizer = o3d.visualization.Visualizer()
        self.view_ctrl_info = view_ctrl_info
        self.recorder = None

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        cv2.destroyAllWindows()
        # self.video_recoder.stop()
        return obs

    def close(self):
        super().close()
        self.frames = list()
        cv2.destroyAllWindows()
        self.video_recoder.stop()

    def step(self, action):
        vis_pcd = True
        if not vis_pcd:
            first_step = action["first_step"]
            vis = action["vis"]
            if not first_step or not vis:
                exec_list = [-1]
            else:
                action_proc = action["action_proc"]
                exec_list = list(range(action_proc.shape[1]-10, action_proc.shape[1]))
            for i in exec_list:
                action["index"] = i
                result = super().step(action)
                all_view, view_list = self.env.render(mode=self.mode, **self.render_kwargs)

                single_view_1 = view_list[4]
                single_view_1 = cv2.pyrUp(single_view_1)
                single_view_2 = view_list[2]
                single_view_2 = cv2.pyrUp(single_view_2)
                record_image = np.concatenate([single_view_1[:940,:,:], single_view_2[:940,:,:]], axis=1)

                if self.real_time_vis:
                    cv2.imshow("Sapien", all_view)
                    cv2.waitKey(1)

                if self.file_path is not None:
                    if not self.video_recoder.is_ready():
                        self.video_recoder.start(self.file_path)
                    self.video_recoder.write_frame(record_image)

            return result
        

        action["index"] = 0
    
        result = super().step(action)
        all_view, view_list = self.env.render(mode=self.mode, **self.render_kwargs)

        pcd = result[0]["d3fields"]
        pcd = np.swapaxes(pcd, 0, 1)
        if pcd.shape[-1] == 4:
            weight = pcd[:, 3:].reshape(-1, 1)
            color = weight * np.array([[1, 0, 0]]) + (
                1 - weight
            ) * np.array([[0, 0, 1]])
        else:
            color = pcd[:, 3:].reshape(-1, 1)
        pcd_o3d = np2o3d(pcd[:, :3], color=color)

        if self.vis_flag:
            self.visualizer.create_window(
                width=640 * 2, height=480 * 2, visible=self.real_time_vis
            )
            self.curr_pcd = copy.deepcopy(pcd_o3d)
            self.visualizer.add_geometry(self.curr_pcd)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.visualizer.add_geometry(origin)
            # self.visualizer.run()
            self.vis_flag = False
        if self.view_ctrl_info is not None:
            view_control = self.visualizer.get_view_control()
            view_control.set_front(self.view_ctrl_info["front"])
            view_control.set_lookat(self.view_ctrl_info["lookat"])
            view_control.set_up(self.view_ctrl_info["up"])
            view_control.set_zoom(self.view_ctrl_info["zoom"])

        self.curr_pcd.points = pcd_o3d.points
        self.curr_pcd.colors = pcd_o3d.colors

        self.visualizer.update_geometry(self.curr_pcd)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        image = self.visualizer.capture_screen_float_buffer()
        image = (255.0 * np.asarray(image)).astype(np.uint8)
        o3d_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        single_view = view_list[0]
        single_view = cv2.pyrUp(single_view)
        record_image = np.concatenate([single_view, o3d_image], axis=1)

        if self.real_time_vis:
            cv2.imshow("Sapien", all_view)
            cv2.waitKey(1)

        if self.file_path is not None:
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)
            self.video_recoder.write_frame(record_image)

        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
