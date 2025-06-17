import collections
import os
import pathlib
import pickle

import dill
import h5py
import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf

# import wandb.sdk.data_types.video as wv

wandb.require("core")

from d3fields.fusion import Fusion
from diffusion_policy.common.data_utils import _convert_actions
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env.sapien_env.sapien_env_wrapper import SapienEnvWrapper
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from utils.my_utils import bcolors

from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light


class SapienSeriesImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        dataset_dir,
        shape_meta: dict,
        n_train=10,
        # FIXME: remove unused parameters, n_train_vis and n_test_vis
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_keys=["right_bottom_view"],
        policy_keys=None,
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        train_obj_ls=None,
        test_obj_ls=None,
        repetitive=False,
        pca_name=None,
        expected_labels=None,
        output_type="pose",
        attention_mode=False,
        real_time_vis=False,
    ):
        super().__init__(output_dir)

        # max_steps = 400

        if n_envs is None:
            n_envs = n_train + n_test

        render_obs_keys += ["direct_up_view"]

        env_cfg_path = os.path.join(dataset_dir, "config.yaml")
        env_cfg = OmegaConf.load(env_cfg_path)

        _6d_2_euler = RotationTransformer(
            "rotation_6d", "euler_angles", to_convention="xyz"
        )
        _6d_2_quat = RotationTransformer("rotation_6d", "quaternion")
        _quat_2_euler = RotationTransformer(
            "quaternion", "euler_angles", to_convention="xyz"
        )

        fusion = None
        if "d3fields" in shape_meta.obs:
            num_cam = len(shape_meta.obs["d3fields"]["info"]["view_keys"])
            fusion = Fusion(num_cam=num_cam, device="cuda", dtype=torch.float16)

        if pca_name is not None:
            pca_model_path = f"pca_model/{pca_name}.pkl"
            pca_model = pickle.load(open(pca_model_path, "rb"))
        else:
            pca_model = None

        def env_fn(seed):
            env_cfg["seed"] = seed
            # if inst_name is not None:
            #     env_cfg.task_level_multimodality = False
            #     env_cfg.manip_obj = inst_name
            #     if inst_name == "cola":
            #         env_cfg.extra_manip_obj = "pepsi"
            #     elif inst_name == "pepsi":
            #         env_cfg.extra_manip_obj = "cola"
            #     else:
            #         raise ValueError("Unknown instance name")
            env: BaseRLEnv = hydra.utils.instantiate(env_cfg)
            add_default_scene_light(env.scene, env.renderer)

            view_ctrl_info = {
                "front": [0.0, 0.2 - 0.001, 3],
                "lookat": [0, 0.2, 0],
                "up": [0.0, 1, 1],
                "zoom": 1.4,
            }

            # if inst_name == "cola":
            #     query_obj = ["Coca-cola red can", "red pad"]
            # elif inst_name == "pepsi":
            #     query_obj = ["Pepsi blue can", "blue pad"]
            # else:
            #     raise ValueError("Unknown instance name")

            return MultiStepWrapper(
                VideoRecordingWrapper(
                    SapienEnvWrapper(
                        env=env,
                        shape_meta=shape_meta,
                        init_states=None,
                        render_obs_keys=render_obs_keys,
                        fusion=fusion,
                        pca_model=pca_model,
                        expected_labels=expected_labels,
                        output_dir=output_dir,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="bgr24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    real_time_vis=self.real_time_vis,
                    view_ctrl_info=None,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = []
        if train_obj_ls is not None:
            for obj in train_obj_ls:
                for _ in range(n_train):
                    env_fns.append(lambda obj=obj: env_fn(obj))
        if test_obj_ls is not None:
            for obj in test_obj_ls:
                for i in range(n_test):
                    seed = test_start_seed + i
                    env_fns.append(lambda seed=seed: env_fn(seed))
        if (train_obj_ls is None) and (test_obj_ls is None):
            env_fns = [env_fn()] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        env_objs = list()

        # train
        if train_obj_ls is not None:
            n_train_obj = len(train_obj_ls)
        else:
            n_train_obj = 1
        for j in range(n_train_obj):
            for i in range(n_train):
                if not repetitive:
                    seed = train_start_idx + i
                    dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
                else:
                    seed = train_start_idx
                    dataset_path = os.path.join(dataset_dir, "episode_0.hdf5")
                # enable_render = i < n_train_vis
                with h5py.File(dataset_path, "r") as f:
                    init_states = f["info"]["init_poses"][()]

                    def init_fn(env, init_states=init_states):
                        # setup rendering
                        # video_wrapper
                        assert isinstance(env.env, VideoRecordingWrapper)
                        # env.env.video_recoder.stop()
                        # env.env.file_path = None
                        # if enable_render:
                        # file_id = wv.util.generate_id()
                        file_id = f"train_{i + j * n_train}"
                        file_name = pathlib.Path(output_dir).joinpath(
                            "media", file_id + ".mp4"
                        )
                        file_name.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(file_name)

                        # switch to init_state reset
                        assert isinstance(env.env.env, SapienEnvWrapper)
                        env.env.env.init_states = init_states
                        env.seed(seed)
                        # env.env.env.reset()

                    env_seeds.append(seed)
                    env_prefixs.append("train/")
                    env_objs.append(train_obj_ls[j])
                    env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        if test_obj_ls is not None:
            n_test_obj = len(test_obj_ls)
        else:
            n_test_obj = 1
        for j in range(n_test_obj):
            for i in range(n_test):
                if not repetitive:
                    seed = test_start_seed + i
                else:
                    seed = test_start_seed

                def init_fn(env, seed=seed):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    file_id = f"test_{i + j * n_train}"
                    file_name = pathlib.Path(output_dir).joinpath(
                        "media", file_id + ".mp4"
                    )
                    file_name.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(file_name)

                    # switch to seed reset
                    assert isinstance(env.env.env, SapienEnvWrapper)
                    env.env.env.init_state = None
                    env.seed(seed)
                    # env.env.env.reset()

                env_seeds.append(seed)
                env_prefixs.append("test/")
                env_objs.append(test_obj_ls[j])
                env_init_fn_dills.append(dill.dumps(init_fn))

        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_objs = env_objs
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self._6d_2_euler = _6d_2_euler
        self._6d_2_quat = _6d_2_quat
        self._quat_2_euler = _quat_2_euler
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.policy_keys = policy_keys # + ["embedding"]
        self.fusion = fusion
        self.output_type = output_type
        self.attention_mode = attention_mode
        self.real_time_vis = real_time_vis
        self.eval_phase = 0

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env_fns = self.env_fns

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        assert n_inits == n_envs

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        fail_idx = []
        reward_record = []

        with tqdm.tqdm(
            total=n_envs,
            desc="Rollout envs",
            position=0,
            colour="green",
            ascii=" 123456789>",
        ) as env_pbar:
            action_ls = list()
            for env_idx in range(n_envs):
                # try:
                if True:
                    term_size = os.get_terminal_size()
                    print(f"{bcolors.OKBLUE}-{bcolors.ENDC}" * term_size.columns)
                    print(f"{bcolors.OKBLUE}Test {env_idx}{bcolors.ENDC}")

                    env = env_fns[env_idx]()
                    this_init_fn = self.env_init_fn_dills[env_idx]

                    # init envs
                    env.run_dill_function(this_init_fn)

                    # start rollout
                    obs = env.reset()
                    for k, v in obs.items():
                        obs[k] = v[None]
                    policy.reset()

                    rotation_transformer = RotationTransformer(
                        from_rep="euler_angles",
                        to_rep="rotation_6d",
                        from_convention="xyz",
                    )

                    # transform_ee_pos = _convert_actions(
                    #         raw_actions=obs["ee_pos"][0],
                    #         rotation_transformer=rotation_transformer,
                    #         action_key="observations/ee_pos",
                    #     )
                    # obs["ee_pos"] = transform_ee_pos[None,...]

                    pbar = tqdm.tqdm(
                        total=self.max_steps,
                        desc="Eval env",
                        leave=False,
                        colour="green",
                        ascii=" 123456789>",
                    )

                    env_action_ls = list()
                    done = False

                    while not done:
                        # create obs dict
                        np_obs_dict = dict(obs)


                        if "ee_pos" in np_obs_dict.keys():
                            transform_ee_pos = _convert_actions(
                                raw_actions=np_obs_dict["ee_pos"][0],
                                rotation_transformer=rotation_transformer,
                                action_key="observations/ee_pos",
                            )
                            np_obs_dict["ee_pos"] = transform_ee_pos[None, ...]

                        # device transfer
                        obs_dict = dict_apply(
                            np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                        )

                        # run policy
                        with torch.no_grad():
                            # filter out policy keys
                            if self.policy_keys is not None:
                                obs_dict = dict(
                                    (k, obs_dict[k]) for k in self.policy_keys
                                )
                            action_dict = policy.predict_action(obs_dict)

                        # device_transfer
                        np_action_dict = dict_apply(
                            action_dict, lambda x: x.detach().to("cpu").numpy()
                        )

                        action = np_action_dict["action"]
                        if not np.all(np.isfinite(action)):
                            print(action)
                            action = np.random.randn(*action.shape)
                            raise RuntimeError("Nan or Inf action")

                        # step env
                        if self.output_type == "pose":
                            if self.abs_action:
                                n_envs, n_steps, action_dim = action.shape
                                env_action = self.undo_transform_action(
                                    action.reshape(n_envs * n_steps, action_dim)
                                )  # (n_envs, n_steps, action_dim)
                                env_action = env_action.reshape(
                                    n_envs, n_steps, env_action.shape[-1]
                                )
                            else:
                                env_action = action
                        else:
                            raise ValueError(f"Unknown output type {self.output_type}")

                        env_action_ls.append(env_action[0])
                        if not self.attention_mode:
                            obs, reward, done, _ = env.step(env_action[0], None)
                        else:
                            env_rl = env.env.env.env
                            obj_list = self.judge_eval_phase(env_rl)
                            # env.env.env.eval_phase = self.eval_phase
                            env.env.env.object_list = obj_list
                            obs, reward, done, _ = env.step(env_action[0])
                        for k, v in obs.items():
                            obs[k] = v[None]
                        done = np.all(done)

                        # update pbar
                        pbar.update(action.shape[1])
                    env.close()
                    pbar.close()
                    env_pbar.update(1)
                    action_ls.append(np.concatenate(env_action_ls, axis=0))

                    # collect data for this round
                    all_video_paths[env_idx] = env.render()
                    all_rewards[env_idx] = env.reward

                    reward = np.max(env.reward)
                    reward_record.append(reward)
                    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
                    emoji = "✅" if reward == 1 else "❌"
                    print(f"{txt_color}{emoji} reward: {reward}{bcolors.ENDC}")
                    if reward < 1:
                        fail_idx.append(env_idx)
                    if len(fail_idx) > 0:
                        print(f"{bcolors.FAIL}Failed tests: {fail_idx}{bcolors.ENDC}")
                # except:
                else:
                    self.fusion.clear_xmem_memory()         
                    reward = 0     
                    all_rewards[env_idx] = reward     
                    reward_record.append(reward)
                    txt_color = bcolors.OKGREEN if reward == 1 else bcolors.FAIL
                    emoji = "✅" if reward == 1 else "❌"
                    print(f"{txt_color}{emoji} reward: {reward}{bcolors.ENDC}")
                    if reward < 1:
                        fail_idx.append(env_idx)
                    if len(fail_idx) > 0:
                        print(f"{bcolors.FAIL}Failed tests: {fail_idx}{bcolors.ENDC}")
            env_pbar.close()

        term_size = os.get_terminal_size()
        print(f"{bcolors.OKBLUE}-{bcolors.ENDC}" * term_size.columns)
        if any(reward < 1 for reward in reward_record):
            fail_count = sum(reward < 1 for reward in reward_record)
            print(f"{bcolors.FAIL}{fail_count} tests failed!{bcolors.ENDC}")
        else:
            print(f"{bcolors.OKGREEN}All tests passed!{bcolors.ENDC}")

        # log
        max_rewards = collections.defaultdict(list)
        max_rewards_obj = {}
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            obj_name = self.env_objs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            if prefix + obj_name not in max_rewards_obj:
                max_rewards_obj[prefix + obj_name] = []
            max_rewards_obj[prefix + obj_name].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}_" + obj_name] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}_" + obj_name] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in max_rewards_obj.items():
            name = prefix + "_mean_score"
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self._6d_2_euler.forward(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def judge_eval_phase(self, env):
        ee_position = env.palm_link.get_pose().p
        reward = env.get_reward(None)
        if self.eval_phase == 0 and reward > 0 and ee_position[2] > 0.35:
            self.eval_phase = 1
            self.done = True

        if self.eval_phase == 0:
            return ["Coca-cola red can", "red pad"]
        if self.eval_phase == 1:
            return ["Pepsi blue can", "blue pad"]

    # def attention(self, object_list: list):
    #     self.env.env.env.object_list = object_list


def test():
    cfg_path = "/home/bing4090/yixuan_old_branch/general_dp/general_dp/config/hang_mug_sim/original_dp.yaml"
    cfg = OmegaConf.load(cfg_path)
    os.system("mkdir -p temp")

    env_runner: BaseImageRunner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir="temp"
    )

    policy: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

    # configure dataset
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    policy.set_normalizer(normalizer)

    env_runner.run(policy)
    print("done")


if __name__ == "__main__":
    test()
