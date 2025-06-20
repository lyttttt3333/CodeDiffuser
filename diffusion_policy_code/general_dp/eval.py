import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import os
import pathlib

import click
import dill
import hydra
import torch
import wandb
from omegaconf import open_dict

wandb.require("core")

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from utils.my_utils import NpEncoder, bcolors


@click.command()
@click.option(
    "-c",
    "--checkpoint",
    default="/mnt/data/diffusion_policy/2024.08.10/norm_false_attention_false_params_keep_batch_128/checkpoints/norm_false_attention_false_params_keep_batch_128.ckpt",
)
@click.option("-o", "--output_dir", default="/mnt/data/diffusion_out")
@click.option("-t", "--test_env_path", default=None)
@click.option("-d", "--device", default="cuda:0")
@click.option("--max_steps", default=240, type=int)
@click.option("--n_test", default=75, type=int)
@click.option("--n_train", default=-1, type=int)
@click.option("--n_test_vis", default=-1, type=int)
@click.option("--n_train_vis", default=-1, type=int)
@click.option("--train_start_idx", default=-1, type=int)
@click.option("--test_start_idx", default=50, type=int)
@click.option("--dataset_dir", default=None, type=str)
@click.option("--repetitive", default=False, type=bool)
# @click.option("--vis_3d", default=False, type=bool)
@click.option("--train_obj_ls", default=None, type=str, multiple=True)
@click.option("--test_obj_ls", default=None, type=str, multiple=True)
# @click.option("--data_root", default="", type=str)
def main(
    checkpoint,
    output_dir,
    test_env_path,
    device,
    max_steps=-1,
    n_test=-1,
    n_train=-1,
    n_test_vis=-1,
    n_train_vis=-1,
    train_start_idx=-1,
    test_start_idx=-1,
    dataset_dir=None,
    vis_3d=False,
    repetitive=False,
    test_obj_ls=[],
    train_obj_ls=[],
    # data_root="",
):
    # test_start_idx = 10000
    # if os.path.exists(output_dir):
    #     click.confirm(
    #         f"{bcolors.WARNING}Output path {output_dir} already exists! Overwrite?{bcolors.ENDC}",
    #         abort=True,
    #     )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    # cfg["task"]["env_runner"][""]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    cfg.task.env_runner._target_ = (
        "diffusion_policy.env_runner.sapien_series_image_runner.SapienSeriesImageRunner"
    )
    cfg.task.env_runner.max_steps = (
        max_steps if max_steps > 0 else cfg.task.env_runner.max_steps
    )
    cfg.task.env_runner.n_test = n_test if n_test >= 0 else cfg.task.env_runner.n_test
    cfg.task.env_runner.n_train = (
        n_train if n_train >= 0 else cfg.task.env_runner.n_train
    )
    cfg.task.env_runner.n_test_vis = (
        n_test_vis if n_test_vis >= 0 else cfg.task.env_runner.n_test_vis
    )
    cfg.task.env_runner.n_train_vis = (
        n_train_vis if n_train_vis >= 0 else cfg.task.env_runner.n_train_vis
    )
    cfg.task.env_runner.train_start_idx = (
        train_start_idx if train_start_idx >= 0 else cfg.task.env_runner.train_start_idx
    )
    cfg.task.env_runner.dataset_dir = (
        dataset_dir if dataset_dir is not None else cfg.task.env_runner.dataset_dir
    )
    with open_dict(cfg.task.env_runner):
        cfg.task.env_runner.test_start_seed = (
            test_start_idx
            if test_start_idx >= 0
            else cfg.task.env_runner.test_start_seed
        )
        cfg.task.env_runner.repetitive = repetitive
        # cfg.task.env_runner.train_obj_ls = train_obj_ls
        # cfg.task.env_runner.test_obj_ls = test_obj_ls
        # cfg.task.env_runner.train_obj_ls = ["cola", "pepsi"]
        cfg.task.env_runner.train_obj_ls = ["cola"]
        # cfg.task.env_runner.test_obj_ls = ["cola", "pepsi"]
        cfg.task.env_runner.attention_mode = False
        cfg.task.env_runner.real_time_vis = True
        cfg.task.env_runner.pca_name = None
        # cfg.task.env_runner.policy_keys = [
        #     "direct_up_view_color",
        #     "left_top_view_color",
        #     "left_bottom_view_color",
        #     "right_top_view_color",
        #     "right_bottom_view_color",
        #     "ee_pos",
        #     "embedding",
        # ]
        cfg.task.env_runner.policy_keys = [
            "d3fields",
            # "embedding",
        ]
        # cfg.task.env_runner.vis_3d = vis_3d
    # add sapien_env from dataset_dir
    import sys

    # device_id = cfg.training.device_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)
    test_env_path = "/home/yitong/diffusion/diffusion_policy_code/eval_env" # os.path.join(current_folder_path, "eval_env")

    if test_env_path is None:
        sys.path.append(cfg.task.env_runner.dataset_dir)
    else:
        sys.path.append(test_env_path)
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir=output_dir, dataset_dir=test_env_path
    )
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True, cls=NpEncoder)


if __name__ == "__main__":
    main()


"""
python /home/yitong/diffusion/diffusion_policy_code/general_dp/eval.py --checkpoint /home/yitong/diffusion/ckpt/attn_1x12/epoch=360.ckpt -o /home/yitong/diffusion/data_eval/eval_result/large_batch_test
"""
