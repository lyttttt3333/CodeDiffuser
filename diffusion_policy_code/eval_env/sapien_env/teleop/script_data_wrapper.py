from pathlib import Path
from tqdm import tqdm
import sys
import os
import click
import numpy as np

from sapien_env.utils.misc_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms


@click.command()
@click.option("--multi", default=True, type=bool)
def main(multi=False):
    curr_path = os.path.abspath(__file__)
    for _ in range(3):
        curr_path = os.path.dirname(curr_path)
    sys.path.append(curr_path)

    mode = "straight"
    task_name = "hang_mug"

    headless = False
    s_idx = 0
    e_idx = 1
    seed_range = range(s_idx, e_idx)

    total_num_demo = e_idx - s_idx
    dataset_name = f"{task_name}_demo_{total_num_demo}"
    if multi:
        dataset_name += "_multi"

    data_root = "/media/yixuan_2T/lyt/mug_data/init_view"
    if dataset_name is None:
        dataset_dir = (
            f"{data_root}/{get_current_YYYY_MM_DD_hh_mm_ss_ms()}"
        )
    else:
        dataset_dir = f"{data_root}/{dataset_name}"
    os.system(f"mkdir -p {dataset_dir}")

    # copy current repo
    save_repo_path = f"sapien_env"
    save_repo_dir = os.path.join(dataset_dir, save_repo_path)
    os.system(f"mkdir -p {save_repo_dir}")

    curr_repo_dir = Path(__file__).parent.parent
    ignore_list = [".git", "__pycache__", "data"]
    for sub_dir in os.listdir(curr_repo_dir):
        if sub_dir not in ignore_list:
            os.system(f"cp -r {os.path.join(curr_repo_dir, sub_dir)} {save_repo_dir}")

    for i in tqdm(
        seed_range, colour="green", ascii=" 123456789>", desc="Dataset Generation"
    ): 
        
        # obj_list = np.array(["cola", "mtndew", "pepsi"])
        branch_name = i % 3
        obj_dist = i % 4
        if obj_dist == 0:
            obj_list = np.array(["nescafe_mug", "nescafe_mug_2"])
        if obj_dist == 1:
            obj_list = np.array(["nescafe_mug", "nescafe_mug"])        
        if obj_dist == 2:
            obj_list = np.array(["nescafe_mug_2", "nescafe_mug_2"])
        if obj_dist == 3:
            obj_list = np.array(["nescafe_mug_2", "nescafe_mug"])

        obj_list = np.array(["nescafe_mug_4", None, None])

        
        

        #branch_name = 1
        PY_CMD = f"python /home/yitong/diffusion/diffusion_policy_code/sapien_env/sapien_env/teleop/script_policy_rollout.py --episode_idx {i} --dataset_dir {dataset_dir} --mode {mode} --task_name {task_name}"
        if headless:
            PY_CMD += " --headless True"
        PY_CMD += f" --obj_name {obj_list[0]}"
        PY_CMD += f" --extra_obj_name {obj_list[1]}"
        if multi:
            PY_CMD += f" --extra_obj_name {obj_list[1]}"
            PY_CMD += f" --extra_obj_name_2 {obj_list[2]}"
            PY_CMD += " --task_level_multimodality true"
            PY_CMD += f" --other {branch_name}"
        print(PY_CMD)
        os.system(PY_CMD)


if __name__ == "__main__":
    main()
