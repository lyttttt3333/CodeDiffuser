from omegaconf import OmegaConf
import hydra
from tqdm import tqdm
import sapien.core as sapien

import sys
sys.path.append('/home/yixuan/sapien_env')

from sapien_env.rl_env.mug_collect_env import MugCollectRLEnv
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.gui.teleop_gui_trossen import GUIBase, YX_TABLE_TOP_CAMERAS
from sapien_env.sim_env.constructor import add_default_scene_light

num_envs = 100
headless = True

cfg = OmegaConf.create(
        {
            '_target_': 'sapien_env.rl_env.mug_collect_env.MugCollectRLEnv',
            'use_gui': True,
            'robot_name': 'panda',
            'frame_skip': 10,
            'use_visual_obs': False,
            'manip_obj': 'pepsi',
        }
    )

# collect data
envs = [hydra.utils.instantiate(cfg) for _ in range(num_envs)]
for i in range(num_envs):
    envs[i].seed(i)
    envs[i].reset()
    add_default_scene_light(envs[i].scene, envs[i].renderer)
    
guis = []
# Setup viewer and camera
for i in range(num_envs):
    env = envs[i]
    gui = GUIBase(env.scene, env.renderer,headless=headless)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    if not gui.headless:
        gui.viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
        gui.viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    guis.append(gui)

for i in tqdm(range(num_envs)):
    guis[i].render(depth=True)
    guis[i].close()
