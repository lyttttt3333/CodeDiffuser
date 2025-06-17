"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import time
import base64
from collections import OrderedDict

from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
import transforms3d
import open3d as o3d
from omegaconf import OmegaConf
import scipy.spatial.transform as st
import diffusers
from openai import OpenAI

from d3fields.utils.draw_utils import np2o3d
from d3fields.fusion import Fusion
from diffusion_policy.real_world.real_aloha_env import RealAlohaEnv
from diffusion_policy.real_world.aloha_master import AlohaMaster
from diffusion_policy.real_world.aloha_bimanual_master import AlohaBimanualMaster
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.kinematics_utils import KinHelper
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.aloha_utils import (
    torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions,
    START_ARM_POSE, START_EE_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, DT, MASTER2PUPPET_JOINT_FN
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def cv_to_base64(obs : np.ndarray, obs_fmt='bgr'):
    """opencv image to base64 image

    Args:
        obs (np.array): (H, W, C) np.array. BGR image
    """
    obs_copy = np.copy(obs)
    if obs_fmt == 'rgb':
        obs_copy = obs_copy[...,::-1]
    elif obs_fmt == 'bgr':
        pass
    else:
        raise RuntimeError("Unsupported obs_fmt: ", obs_fmt)
    _, img_encoded = cv2.imencode('.jpg', obs_copy)
    img_bytes = img_encoded.tobytes()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    return base64_image

def policy_action_to_env_action(policy_action, action_mode, num_bots):
    # policy_action: (T, Da), Da=10 * num_bots (3 dof translation, 6 dof rotation, 1 gripper)
    if action_mode == 'eef':
        T = policy_action.shape[0]
        action_reshape = policy_action.reshape((T * num_bots, 10))
        env_actions = np.zeros((T * num_bots, 7), dtype=np.float64)
        env_actions[:,:3] = action_reshape[:,:3]
        from pytorch3d.transforms import rotation_conversions as pt
        action_rot_mat = pt.rotation_6d_to_matrix(torch.from_numpy(action_reshape[:,3:9])).numpy()
        env_actions[:, 3:6] = st.Rotation.from_matrix(action_rot_mat).as_euler('xyz')
        env_actions[:, 6:] = action_reshape[:,9:]
        env_actions = env_actions.reshape((T, num_bots * 7))
    elif action_mode == 'joint':
        env_actions = policy_action
    return env_actions

def load_policy(ckpt_path, n_action_steps=-1, device = torch.device('cuda')):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.eval().to(device)

        # set inference params
        if n_action_steps > 0:
            policy.n_action_steps = n_action_steps
        policy.num_inference_steps = 16 # DDIM inference iterations
        noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        policy.noise_scheduler = noise_scheduler
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)
    # cfg.task.shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.025
    return policy, cfg

OmegaConf.register_new_resolver("eval", eval, replace=True)


class GPTPolicySwitch:
    def __init__(self) -> None:
        ### GPT init
        self.client = OpenAI()
        self.policy_idx = -1
        self.change_policy = False
        self.execution_plan = None

    def query(self, obs, policy_dict : OrderedDict, task_name, obs_fmt='bgr', verbose=False):
        """get the corresponding policy for the current observation

        Args:
            obs (np.array): (V, H, W, C) np.array. Multi-view RGB images
            policy_dict (OrderedDict): OrderedDict of policies, where keys are policy names and values are policy ckpts_path
        """
        base64_images = [cv_to_base64(obs[cam_i], obs_fmt=obs_fmt) for cam_i in range(obs.shape[0])]
        
        policy_names = list(policy_dict.keys())

        sys_texts = "You are a robot equipped with following policy: \n"
        for i in range(len(policy_names)):
            sys_texts += f"{i}. {policy_names[i]}\n"
        sys_texts += f"The target task is {task_name}."
        sys_texts += f" Select only one policy you need to execute now according to incoming multi-view or single view observations. Explain why you select the policy." # In the end of your response, output the final answer in the format of 'Final Answer: [policy index]'."
        if self.execution_plan is not None:
            sys_texts += f" There is no execution plan currently. Please also generate a execution plan for finishing the task."
        else:
            sys_texts += f" As planned before, the execution plan is: {self.execution_plan} (indexing the policy). If the plan is feasible, please indicate the current stage of the execution plan. Only if necessary, please generate a new execution plan for finishing the task, if the original plan is not feasible anymore. If a stage is finished, do not need to change the execution plan. You can just change the current stage to the next stage."
        sys_texts += f" The format of the response should contain following sections: \n\n"
        sys_texts += f"1. Observation analysis and justification for planning and justification: \n\n"
        sys_texts += f"2. Policy selection: (in the format of a single integer indexing the policy) \n\n"
        sys_texts += f"3. Execution plan: (in the format of a list of integers indexing the policy) \n\n"
        sys_texts += f"4. Current stage: (in the format of a single integer indexing the execution plan) \n\n"
        sys_texts += f"Example response: \n\n"
        sys_texts += f"Observation analysis and justification: there is one closed drawer and one object on the table. I need to open the drawer and grasp the object. \n\n"
        sys_texts += f"Policy selection: 0 \n\n"
        sys_texts += f"Execution plan: [2, 0, 3, 1] \n\n"
        sys_texts += f"Current stage: 1 \n\n"
        sys_texts += f"The format of the response should be strictly followed. No wrong newline or extra space is allowed."

        if verbose:
            start_time = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": sys_texts},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        # {"type": "text", "text": "What's in this image?"},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        } for base64_image in base64_images
                    ],
                }
            ],
            max_tokens=300,
        )
        if verbose:
            print(f'gpt-4-vision-preview: {time.time()-start_time}')
            print(response.choices[0])
        
        # final answer to policy index
        def gpt_output_to_policy_output(gpt_output):
            gpt_output_splits = gpt_output.split('\n\n')
            policy_selection = -1
            execution_plan = None
            current_stage = -1
            for gpt_output_split in gpt_output_splits:
                if 'Policy selection' in gpt_output_split:
                    policy_selection = int(gpt_output_split.split('Policy selection: ')[-1])
                elif 'Execution plan' in gpt_output_split:
                    execution_plan = eval(gpt_output_split.split('Execution plan: ')[-1])
                elif 'Current stage' in gpt_output_split:
                    current_stage = int(gpt_output_split.split('Current stage: ')[-1])
            try:
                assert policy_selection != -1
                assert execution_plan is not None
                assert current_stage != -1
            except:
                raise RuntimeError("Invalid GPT output: ", gpt_output)
            return policy_selection, execution_plan, current_stage

        policy_idx, execution_plan, current_stage = gpt_output_to_policy_output(response.choices[0].message.content)
        self.execution_plan = execution_plan
        self.current_stage = current_stage
        if policy_idx != self.policy_idx:
            self.policy_idx = policy_idx
            self.change_policy = True
        else:
            self.change_policy = False
        return policy_idx, self.change_policy, self.execution_plan, self.current_stage

@click.command()
@click.option('--input_dirs', '-i', required=True, multiple=True, help='Path to checkpoint')
@click.option('--policy_names', '-p', required=True, multiple=True, help='Name of the policy')
@click.option('--task_name', '-t', required=True, help='Name of the task')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--vis_d3fields', default=False, type=bool, help="Visualize d3fields.")
@click.option('--robot_sides', '-r', default=['right'], multiple=True, help="Which robot to control.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--n_action_steps', '-n', default=-1, type=int, help="Number of action steps to execute. -1 means invalid.")
@click.option('--robot_init_joints', '-j', default=None, multiple=True, help="Initial robot joint configuration.")
@click.option('--gpt_every_steps', '-g', default=300, type=int, help="Ask GPT after every n steps.")
def main(input_dirs, policy_names, task_name, output, match_dataset, match_episode,
    vis_camera_idx, vis_d3fields, robot_sides,
    steps_per_inference, max_duration,
    frequency, command_latency, robot_init_joints, n_action_steps, gpt_every_steps):
    assert len(input_dirs) == len(policy_names)
    policy_dict = OrderedDict()
    for i in range(len(input_dirs)):
        policy_dict[policy_names[i]] = input_dirs[i]
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    robot_init_joints = np.array(robot_init_joints, dtype=np.float32)
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_paths = input_dirs
    policy_idx = 0
    action_offset = 0
    device = torch.device('cuda')
    policy, cfg = load_policy(ckpt_paths[policy_idx], n_action_steps=n_action_steps, device=device)

    # setup experiment
    dt = 1/frequency
    os.system(f'mkdir -p {output}')
    kin_helper = KinHelper(robot_name=cfg.task.dataset.robot_name)
    fusion = None
    expected_labels = None
    for obs_key in cfg.task.shape_meta['obs'].keys():
        if 'd3fields' in obs_key:
            num_cam = len(cfg.task.shape_meta['obs'][obs_key]['info']['view_keys'])
            fusion = Fusion(num_cam=num_cam, dtype=torch.float16)
            expected_labels = cfg.task.expected_labels if 'expected_labels' in cfg.task else None
            break
    num_bots = len(robot_sides)
    if 'key' not in cfg.task.shape_meta['action'] or cfg.task.shape_meta['action']['key'] == 'eef_action':
        action_mode = 'eef'
    elif cfg.task.shape_meta['action']['key'] == 'joint_action':
        action_mode = 'joint'
    else:
        raise RuntimeError("Unsupported action mode: ", cfg.task.shape_meta['action']['key'])

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    # create GPT polic switch
    gpt_policy_switch = GPTPolicySwitch()

    with SharedMemoryManager() as shm_manager:
        with RealAlohaEnv(
                output_dir=output, 
                n_obs_steps=n_obs_steps,
                robot_sides=robot_sides,
                init_qpos=robot_init_joints,
                # recording resolution
                obs_image_resolution=(640,480),
                frequency=frequency,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                video_capture_fps=30,
                video_capture_resolution=(640,480),
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager) as env, \
                AlohaMaster(shm_manager=shm_manager, robot_side=robot_sides[0]) if len(robot_sides) == 1 else \
                    AlohaBimanualMaster(shm_manager=shm_manager, robot_sides=robot_sides) as master_bot:
            cv2.setNumThreads(1)

            # # Should be the same as demo
            # # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    fusion=fusion, expected_labels=expected_labels, teleop=kin_helper)

                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.episode_id
                    vis_img = obs[f'camera_{vis_camera_idx}_color'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        break

                    precise_wait(t_sample)

                    state = master_bot.get_motion_state()
                    target_state = state['joint_pos'].copy()
                    human_eef_actions = np.zeros(7 * len(robot_sides))
                    for rob_i in range(num_bots):
                        # save target joint
                        target_state[7*rob_i + 6] = MASTER2PUPPET_JOINT_FN(state['joint_pos'][7*rob_i + 6])

                        # save target eef
                        fk_joint_pos = np.zeros(8)
                        fk_joint_pos[0:6] = state['joint_pos'][rob_i*7:rob_i*7+6]
                        curr_puppet_gripper_joint = MASTER2PUPPET_JOINT_FN(state['joint_pos'][rob_i*7+6])
                        target_puppet_ee_pose = kin_helper.compute_fk_sapien_links(fk_joint_pos,
                                                                                   [kin_helper.sapien_eef_idx])[0] # (4, 4)
                        target_puppet_ee_pose = np.linalg.inv(env.puppet_bot.base_pose_in_world[0]) @ env.puppet_bot.base_pose_in_world[rob_i] @ target_puppet_ee_pose

                        human_eef_actions[rob_i*7:(rob_i+1)*7] = np.concatenate([target_puppet_ee_pose[:3,3],
                                                                            transforms3d.euler.mat2euler(target_puppet_ee_pose[:3,:3], axes='sxyz'),
                                                                            np.array([curr_puppet_gripper_joint])])
                    # robot_base_in_world = env.puppet_bot.base_pose_in_world
                    # eef_pose_mat = eef_pose.to_transformation_matrix()
                    # eef_actions_in_world_mat = robot_base_in_world @ eef_pose_mat
                    # human_eef_actions_in_world = np.concatenate([eef_actions_in_world_mat[:3,3],
                    #                                     transforms3d.euler.mat2euler(eef_actions_in_world_mat[:3,:3], axes='sxyz'),
                    #                                     eef_actions[-1:]])


                    # vis policy
                    if vis_d3fields:
                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                fusion=fusion, expected_labels=expected_labels, teleop=kin_helper)

                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy() # (T, Da), Da=10 (3 dof translation, 6 dof rotation, 1 gripper)
                            print('Inference latency:', time.time() - s)
                        
                        # visualize d3fields
                        if iter_idx == 0:
                            visualizer = o3d.visualization.Visualizer()
                            visualizer.create_window()

                            if 'd3fields' in obs_dict_np:
                                d3fields = obs_dict_np['d3fields'][0, :3].transpose()
                            else:
                                d3fields = np.zeros((1,3))
                            d3fields_o3d = np2o3d(d3fields)

                            curr_d3fields = d3fields_o3d
                            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                            visualizer.add_geometry(origin)
                            visualizer.add_geometry(curr_d3fields)

                        # visualize end effector actions
                        if action_mode == 'eef':
                            # convert policy action to env actions
                            eef_actions_in_world = policy_action_to_env_action(action, action_mode, num_bots)
                            
                            import matplotlib.cm as cm
                            cmap = cm.get_cmap('plasma')
                            act_color_list = cmap(1 - np.linspace(0, 1, len(eef_actions_in_world)))[...,:3]
                            coord_size_list = np.linspace(0.05, 0.02, len(eef_actions_in_world))

                            # initialize visualizer
                            if iter_idx == 0:
                                action_meshes = []
                                box_pts_ls = []

                                for act_idx in range(len(eef_actions_in_world)):
                                    ### vis using box
                                    # box = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
                                    # box.paint_uniform_color(act_color_list[act_idx])
                                    # box_pts = np.asarray(box.vertices) - np.array([0.005, 0.005, 0.005])
                                    # box.vertices = o3d.utility.Vector3dVector(box_pts)
                                    
                                    ### vis using coord
                                    for rob_i in range(num_bots):
                                        box = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size_list[act_idx])
                                        box_pts_ls.append(np.copy(np.asarray(box.vertices)))

                                        action_meshes.append(box)
                                        visualizer.add_geometry(action_meshes[-1])

                                ### vis using coord
                                for rob_i in range(num_bots):
                                    box = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
                                    box_pts_ls.append(np.copy(np.asarray(box.vertices)))

                                    action_meshes.append(box)
                                    visualizer.add_geometry(action_meshes[-1])
                                visualizer.poll_events()
                                visualizer.update_renderer()
                                visualizer.run()
                        
                        # visualize in open3d
                        if 'd3fields' in obs_dict_np:
                            d3fields = obs_dict_np['d3fields'][0, :3].transpose()
                        else:
                            d3fields = np.zeros((1,3))
                        d3fields_o3d = np2o3d(d3fields)

                        curr_d3fields.points = d3fields_o3d.points
                        curr_d3fields.colors = d3fields_o3d.colors
                        visualizer.update_geometry(curr_d3fields)

                        if action_mode == 'eef':
                            for act_idx in range(len(eef_actions_in_world)):
                                for rob_i in range(num_bots):
                                    eef_action = eef_actions_in_world[act_idx, rob_i*7:(rob_i+1)*7]
                                    eef_action_mat = np.eye(4)
                                    eef_action_mat[:3,3] = eef_action[:3]
                                    eef_action_mat[:3,:3] = st.Rotation.from_euler('xyz', eef_action[3:6]).as_matrix()

                                    # update open3d mesh
                                    ls_idx = act_idx * num_bots + rob_i
                                    box_pts_copy = np.copy(box_pts_ls[ls_idx])
                                    transformed_box_pts = eef_action_mat @ np.concatenate([box_pts_copy, np.ones((len(box_pts_copy),1))], axis=-1).T
                                    action_meshes[ls_idx].vertices = o3d.utility.Vector3dVector(transformed_box_pts[:3].T)
                                    visualizer.update_geometry(action_meshes[ls_idx])
                        
                            for rob_i in range(num_bots):
                                ls_idx = -(num_bots - rob_i)
                                curr_eef_pose_mat = kin_helper.compute_fk_sapien_links(obs['full_joint_pos'][-1][rob_i*8:(rob_i+1)*8], [kin_helper.sapien_eef_idx])[0]
                                curr_eef_pose_mat = np.linalg.inv(env.puppet_bot.base_pose_in_world[0]) @ env.puppet_bot.base_pose_in_world[rob_i] @ curr_eef_pose_mat
                                box_pts_copy = np.copy(box_pts_ls[ls_idx])
                                transformed_box_pts = curr_eef_pose_mat @ np.concatenate([box_pts_copy, np.ones((len(box_pts_copy),1))], axis=-1).T
                                action_meshes[ls_idx].vertices = o3d.utility.Vector3dVector(transformed_box_pts[:3].T)
                                visualizer.update_geometry(action_meshes[ls_idx])

                        visualizer.update_geometry(curr_d3fields)
                        visualizer.poll_events()
                        visualizer.update_renderer()

                    # execute teleop command
                    env.exec_actions(
                        joint_actions=[target_state],
                        eef_actions=[human_eef_actions],
                        timestamps=[time.time() + 0.1])
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy_idx = 0
                    policy = load_policy(ckpt_paths[policy_idx], n_action_steps=n_action_steps, device=device)[0]
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        if iter_idx % gpt_every_steps == 0:
                            # bgr_obs = np.stack([obs[f'camera_{cam_i}_color'][-1] for cam_i in range(3)])
                            bgr_obs = np.stack([obs[f'camera_{cam_i}_color'][-1] for cam_i in range(1,2)])
                            policy_idx, change_policy, execution_plan, current_stage = gpt_policy_switch.query(bgr_obs, policy_dict, task_name=task_name, verbose=True, obs_fmt='rgb')
                            if policy_idx >= len(ckpt_paths):
                                print("No more policies to evaluate!")
                                continue
                            if change_policy:
                                print('switching policy')
                                # Change to next policy
                                del policy
                                env.reset(policy_name=policy_names[policy_idx])
                                policy = load_policy(ckpt_paths[policy_idx], n_action_steps=n_action_steps, device=device)[0]
                                policy.reset()
                            else:
                                print("Policy not changed!")
                        if not os.path.exists(os.path.join(output, 'policy_info', f'episode_{env.episode_id}')):
                            os.system(f'mkdir -p {os.path.join(output, "policy_info", f"episode_{env.episode_id}")}')
                        with open(os.path.join(output, 'policy_info', f'episode_{env.episode_id}', f'policy_info_{iter_idx}.txt'), 'w') as f:
                            f.write(f"Policy {policy_idx} in control!\n")
                            f.write(f"Policy info:\n")
                            f.write(f"Policy name: {policy_names[policy_idx]}\n")
                            f.write(f"Policy ckpt: {ckpt_paths[policy_idx]}\n")
                            f.write(f"Execution plan: {execution_plan}\n")
                            f.write(f"Current stage: {current_stage}\n")
                            f.write('\n')
                        print(f"Policy {policy_idx} in control!")
                        print(f"Policy info:")
                        print(f"Policy name: {policy_names[policy_idx]}")
                        print(f"Policy ckpt: {ckpt_paths[policy_idx]}")
                        print(f"Execution plan: {execution_plan}")
                        print(f"Current stage: {current_stage}")
                        print()

                        loop_start_t = time.perf_counter()
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        # print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                fusion=fusion, expected_labels=expected_labels, teleop=kin_helper)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy() # (T, Da), Da=10 (3 dof translation, 6 dof rotation, 1 gripper)
                            print('Inference latency:', time.time() - s)
                        
                        ### visualize policy
                        if vis_d3fields:
                            # visualize in d3fields
                            if 'd3fields' in obs_dict_np:
                                d3fields = obs_dict_np['d3fields'][0, :3].transpose()
                            else:
                                d3fields = np.zeros((1,3))
                            d3fields_o3d = np2o3d(d3fields)

                            curr_d3fields.points = d3fields_o3d.points
                            curr_d3fields.colors = d3fields_o3d.colors
                            visualizer.update_geometry(curr_d3fields)

                            if action_mode == 'eef':
                                # convert policy action to env actions
                                eef_actions_in_world = policy_action_to_env_action(action, action_mode, num_bots)
                                
                                import matplotlib.cm as cm
                                cmap = cm.get_cmap('plasma')
                                act_color_list = cmap(1 - np.linspace(0, 1, len(eef_actions_in_world)))[...,:3]
                                coord_size_list = np.linspace(0.05, 0.02, len(eef_actions_in_world))
                                for act_idx in range(len(eef_actions_in_world)):
                                    for rob_i in range(num_bots):
                                        eef_action = eef_actions_in_world[act_idx, rob_i*7:(rob_i+1)*7]
                                        eef_action_mat = np.eye(4)
                                        eef_action_mat[:3,3] = eef_action[:3]
                                        eef_action_mat[:3,:3] = st.Rotation.from_euler('xyz', eef_action[3:6]).as_matrix()

                                        # update open3d mesh
                                        ls_idx = act_idx * num_bots + rob_i
                                        box_pts_copy = np.copy(box_pts_ls[ls_idx])
                                        transformed_box_pts = eef_action_mat @ np.concatenate([box_pts_copy, np.ones((len(box_pts_copy),1))], axis=-1).T
                                        action_meshes[ls_idx].vertices = o3d.utility.Vector3dVector(transformed_box_pts[:3].T)
                                        visualizer.update_geometry(action_meshes[ls_idx])
                                
                                # curr_eef_pose = obs['ee_pos'][-1]
                                # curr_eef_pose_mat = np.eye(4)
                                # curr_eef_pose_mat[:3,3] = curr_eef_pose[:3]
                                # curr_eef_pose_mat[:3,:3] = st.Rotation.from_euler('xyz', curr_eef_pose[3:6]).as_matrix()
                                for rob_i in range(num_bots):
                                    ls_idx = -(num_bots - rob_i)
                                    curr_eef_pose_mat = kin_helper.compute_fk_sapien_links(obs['full_joint_pos'][-1][rob_i*8:(rob_i+1)*8], [kin_helper.sapien_eef_idx])[0]
                                    curr_eef_pose_mat = np.linalg.inv(env.puppet_bot.base_pose_in_world[0]) @ env.puppet_bot.base_pose_in_world[rob_i] @ curr_eef_pose_mat
                                    box_pts_copy = np.copy(box_pts_ls[ls_idx])
                                    transformed_box_pts = curr_eef_pose_mat @ np.concatenate([box_pts_copy, np.ones((len(box_pts_copy),1))], axis=-1).T
                                    action_meshes[ls_idx].vertices = o3d.utility.Vector3dVector(transformed_box_pts[:3].T)
                                    visualizer.update_geometry(action_meshes[ls_idx])
                            visualizer.poll_events()
                            visualizer.update_renderer()
                        ### finish visualization
                        
                        env_actions = policy_action_to_env_action(action, action_mode, num_bots)

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.1
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            env_actions = env_actions[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            env_actions = env_actions[is_new]
                            action_timestamps = action_timestamps[is_new]
                        
                        # clip within local range
                        if action_mode == 'eef':
                            for rob_i in range(num_bots):
                                curr_eef_pose_mat = kin_helper.compute_fk_sapien_links(obs['full_joint_pos'][-1][rob_i*8:(rob_i+1)*8], [kin_helper.sapien_eef_idx])[0]
                                curr_eef_pose_mat = np.linalg.inv(env.puppet_bot.base_pose_in_world[0]) @ env.puppet_bot.base_pose_in_world[rob_i] @ curr_eef_pose_mat
                                curr_eef_pose = np.concatenate([curr_eef_pose_mat[:3,3],
                                                                transforms3d.euler.mat2euler(curr_eef_pose_mat[:3,:3], axes='sxyz')])
                                env_actions[:,rob_i*7:rob_i*7+3] = np.clip(
                                    env_actions[:,rob_i*7:rob_i*7+3], 
                                    curr_eef_pose[:3] - 0.1, curr_eef_pose[:3] + 0.1)
                                env_actions[:,rob_i*7+3:rob_i*7+6] = np.clip(
                                    env_actions[:,rob_i*7+3:rob_i*7+6], 
                                    curr_eef_pose[3:6] - np.pi/4.0, curr_eef_pose[3:6] + np.pi/4.0)
                        elif action_mode == 'joint':
                            curr_joint_pos = obs['joint_pos'][-1]
                            env_actions = np.clip(
                                env_actions, 
                                curr_joint_pos - np.pi/4.0, curr_joint_pos + np.pi/4.0)

                        # execute actions
                        with np.printoptions(precision=15, suppress=True):
                            # print('curr_time', time.time())
                            # print('env_actions', env_actions)
                            # print('action_timestamps', action_timestamps.astype(str))
                            action_send_delay = 0.1
                            if action_mode == 'eef':
                                env.exec_actions(
                                    joint_actions=[target_state for _ in range(len(env_actions))],
                                    eef_actions=env_actions,
                                    # timestamps=action_timestamps + 0.5,
                                    timestamps=action_timestamps + time.time() - obs_timestamps[-1] + action_send_delay, # debug only
                                    mode=action_mode,
                                    ik_init=[obs['full_joint_pos'][-1] for _ in range(len(env_actions))])
                            elif action_mode == 'joint':
                                env.exec_actions(
                                    joint_actions=env_actions,
                                    eef_actions=[human_eef_actions for _ in range(len(env_actions))],
                                    timestamps=action_timestamps + time.time() - obs_timestamps[-1] + action_send_delay, # debug only
                                    mode=action_mode)
                            print(f"Submitted {len(env_actions)} steps of actions at {time.time()}")

                        # visualize
                        episode_id = env.episode_id
                        vis_img = obs[f'camera_{vis_camera_idx}_color'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                        # time.sleep(max(0, 2.0 - (time.perf_counter() - loop_start_t)))

                        # measure frequency
                        loop_end_t = time.perf_counter()
                        print(f'Frequency: {1/(loop_end_t - loop_start_t)} Hz')

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
