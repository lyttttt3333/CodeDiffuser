import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import IPython
import transforms3d
e = IPython.embed
import os
import h5py
import cv2
import sys
import torch
sys.path.append(os.environ['SAPIEN_ROOT'])
from sapien_env.rl_env.relocate_env import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300

class SingleArmPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self,env, ee_link_pose):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose)


        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)


        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        eluer = transforms3d.euler.quat2euler(quat,axes='sxyz')
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper
        return cartisen_action
    
    def generate_trajectory(self, env, ee_link_pose):
        cube_position= env.manipulated_object.get_pose().p +np.array([0.4,0,0])

        self.trajectory = [
            {"t": 0, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.021}, 
            {"t": 100, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.07}, 
            {"t": 300, "xyz": cube_position +np.array([-0.009,0,0.15]), "quat": ee_link_pose.q, "gripper": 0.07}, 
            {"t": 400, "xyz": cube_position+np.array([-0.009,0,0.08]), "quat": ee_link_pose.q, "gripper": 0.07}, 
            {"t": 500, "xyz": cube_position+np.array([-0.009,0,0.08]), "quat": ee_link_pose.q, "gripper": 0.025}, 
            {"t": 600, "xyz": cube_position +np.array([-0.009,0,0.15]), "quat": ee_link_pose.q, "gripper": 0.025}, 
            {"t": 700, "xyz": cube_position +np.array([0.1,0,0.15]), "quat": ee_link_pose.q, "gripper": 0.025},
            {"t": 750, "xyz": cube_position +np.array([0.1,0,0.1]), "quat": ee_link_pose.q, "gripper": 0.025},
            {"t": 800, "xyz": cube_position +np.array([0.1,0,0.1]), "quat": ee_link_pose.q, "gripper": 0.07},
        ]

def plot_contact_color_map(contact_data, sensor_number_dim):
    contact_data_left= contact_data[:sensor_number_dim]
    contact_data_right= contact_data[sensor_number_dim:]
    contact_data_left= contact_data_left.reshape((4,4))
    contact_data_right= contact_data_right.reshape((4,4))
    
    contact_data_left_norm = (contact_data_left - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
    contact_data_right_norm = (contact_data_right -  MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

    contact_data_left_norm_scaled = (contact_data_left_norm * 255).astype(np.uint8)
    contact_data_right_norm_scaled = (contact_data_right_norm * 255).astype(np.uint8)

    colormap_left = cv2.applyColorMap(contact_data_left_norm_scaled, cv2.COLORMAP_VIRIDIS)
    colormap_right = cv2.applyColorMap(contact_data_right_norm_scaled, cv2.COLORMAP_VIRIDIS)
    
    separator = np.ones((colormap_left.shape[0],1,3), dtype=np.uint8) * 255

    combined_colormap = np.concatenate((colormap_left, separator,colormap_right), axis=1)
    cv2.imshow("Left finger and Right finger Contact Data", combined_colormap )
    cv2.waitKey(1)

def main_env():
    teleop = TeleopRobot()
    max_timesteps = 800
    num_episodes = 50
    onscreen = True
    for episode_idx in range(num_episodes):
        env = RelocateRLEnv(use_gui=True, robot_name="trossen_vx300s_tactile_thin",
                            object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)
        env.seed(episode_idx)
        env.reset()
        # viewer = Viewer(base_env.renderer)
        # viewer.set_scene(base_env.scene)
        # base_env.viewer = viewer

        
        # Setup viewer and camera
        add_default_scene_light(env.scene, env.renderer)
        gui = GUIBase(env.scene, env.renderer)
        for cam_name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
            gui.create_camera(**params)
        gui.viewer.set_camera_rpy(0, -0.7, 0.01)
        gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
        scene = env.scene
        steps = 0
        scene.step()
        
        for link in env.robot.get_links():
            print(link.get_name(),link.get_pose())
        for joint in env.robot.get_active_joints():
            print(joint.get_name())

        plot_contact = True

        if plot_contact:
            # Create a named window with the WINDOW_NORMAL flag
            cv2.namedWindow("Left finger and Right finger Contact Data", cv2.WINDOW_NORMAL)

            # Set the window size
            cv2.resizeWindow("Left finger and Right finger Contact Data",WINDOW_WIDTH, WINDOW_HEIGHT)

            sensor_number_dim = int(env.sensors.shape[0]/2)

        env.seed(episode_idx)
        # env.reset_env()
        scene.step()
        scripted_policy = SingleArmPolicy()
        cartisen_action=teleop.init_cartisen_action(env.robot.get_qpos()[:])
        action = np.zeros(7)
        arm_dof = env.arm_dof
        # Set data dir

        for i in range(max_timesteps):
            cartisen_action = scripted_policy.single_trajectory(env,teleop.ee_link_pose)
            action[:arm_dof] = teleop.ik_vx300s(env.robot.get_qpos()[:],cartisen_action)
            action[arm_dof:] = cartisen_action[6]
            obs, reward, done, _ = env.step(action[:7])

            if plot_contact:
                contact_data = env.sensors
                plot_contact_color_map(contact_data, sensor_number_dim)
            if onscreen:
                gui.render()
        gui.viewer.close()

if __name__ == '__main__':
    main_env()