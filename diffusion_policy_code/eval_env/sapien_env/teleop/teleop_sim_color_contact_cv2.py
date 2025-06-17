import numpy as np
import cv2
import os
import sys
sys.path.append(os.environ['SAPIEN_ROOT'])

from sapien_env.rl_env.relocate_env import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.rl_env.para import ARM_INIT
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


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
    env = RelocateRLEnv(use_gui=True, robot_name="trossen_vx300s_tactile_thin",frame_skip=10, use_visual_obs=False, use_ray_tracing=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
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

    plot_contact =True

    if plot_contact:
        # Create a named window with the WINDOW_NORMAL flag
        cv2.namedWindow("Left finger and Right finger Contact Data", cv2.WINDOW_NORMAL)
        # Set the window size
        cv2.resizeWindow("Left finger and Right finger Contact Data",WINDOW_WIDTH, WINDOW_HEIGHT)
        sensor_number_dim = int(env.sensors.shape[0]/2)


    teleop = TeleopRobot()
    cartisen_action =teleop.init_cartisen_action(env.robot.get_qpos()[:])
    action = np.zeros(7)
    arm_dof = env.arm_dof
    # gui.viewer.toggle_pause(True)
    print(arm_dof)
    # env.reset()
    for i in range(10000):
        # print(cartisen_action)
        cartisen_action = teleop.keyboard_control(gui.viewer, cartisen_action)
        action[:arm_dof] = teleop.ik_vx300s(env.robot.get_qpos()[:],cartisen_action)
        action[arm_dof:] = cartisen_action[6]
        # print(action - env.robot.get_qpos()[:7])
        obs, reward, done, _ = env.step(action[:arm_dof+1])
        if plot_contact:
            contact_data = env.sensors
            plot_contact_color_map(contact_data, sensor_number_dim)
        gui.render()


if __name__ == '__main__':
    main_env()