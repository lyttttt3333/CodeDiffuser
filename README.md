# CodeDiffuser: Attention-Enhanced Diffusion Policy via VLM-Generated Code for Instruction Ambiguity

[Website](https://robopil.github.io/code-diffuser/) | [Paper](https://robopil.github.io/code-diffuser/media/pdf/main.pdf) |

<a target="_blank" href="https://www.linkedin.com/in/guang-yin-4a0176190/">Guang Yin</a><sup>2</sup>,
<a target="_blank" href="https://robopil.github.io/GenDP/">Yitong Li</a><sup>4</sup>,
<a target="_blank" href="https://wangyixuan12.github.io/">Yixuan Wang</a><sup>1</sup>,
<a target="_blank" href="https://sites.google.com/umich.edu/dmcconachie">Dale McConachie</a><sup>2</sup>,
<a target="_blank" href="https://www.paarthshah.me/about">Paarth Shah</a><sup>2</sup>,
<a target="_blank" href="https://www.linkedin.com/in/kunimatsu-hashimoto-450258a1/">Kunimatsu Hashimoto</a><sup>2</sup>,
<a target="_blank" href="https://huan-zhang.com/">Huan Zhang</a><sup>3</sup>
<a target="_blank" href="https://www.thekatherineliu.com/">Katherine Liu</a><sup>2</sup>
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
            
<sup>1</sup>Columbia University,
<sup>2</sup>Toyota Research Institute,
<sup>3</sup>University of Illinois Urbana-Champaign,
<sup>4</sup>Tsinghua University<br>


https://robopil.github.io/code-diffuser/


## :bookmark_tabs: Table of Contents
- [Overview](#video_game-overview)
- [Install](#hammer-install)
- [Generate Dataset](#floppy_disk-generate-dataset)
- [Train](#gear-train)
- [Infer in Simulation](#video_game-infer-in-simulation)

## :video_game: Overview
This codebase consists of the following components:
```console
./dataset                         # save generated data and corresponding checkpoints
./diffusion_policy_code           # save core code
./ref_lib                         # save pre-selected dino reference feature
```

In "diffusion_policy_code", the core components are following sub-directories:
```console
./d3fields_dev                    # perception modules
./general_dp                      # diffusion policy
./eval_env                        # policy evaluation environment
./sapien_env                      # data generation environment
```

For data generation, the main function is in 
```console
./diffusion_policy_code/sapien_env/sapien_env/teleop/hang_mug.py
```
or
```console
./diffusion_policy_code/sapien_env/sapien_env/teleop/pack_battery.py
```

For policy training , the main function is in 
```console
./diffusion_policy_code/general_dp/train.py
```

For policy evaluation, the main function is in 
```console
./diffusion_policy_code/general_dp/eval.py
```


## :hammer: Install
We use anaconda distribution for installation: 
```console
conda env create -f environment.yml
conda activate code_diffuser
cd diffusion_policy_code
pip install -e d3fields_dev/
pip install -e general_dp/
pip install -e robomimic/
pip install -e sapien_env/
```


## :floppy_disk: Generate Dataset

### Generate from Existing Environments
We use the [SAPIEN](https://sapien.ucsd.edu/docs/latest/index.html) to build the simulation environments. To create the data of heuristic policy, use the following command:
```console
python diffusion_policy_code/sapien_env/sapien_env/teleop/pack_battery.py \
        --start_idx [start_idx]\
        --end_idx [end_idx]\
        --dataset_dir [dataset_dir]\
        --resolution [resolution]\
        --view [view]
```
and 
```console
python diffusion_policy_code/sapien_env/sapien_env/teleop/hang_mug.py \
        --start_idx [start_idx]\
        --end_idx [end_idx]\
        --dataset_dir [dataset_dir]\
        --resolution [resolution]\
        --view [view]
```
Note that
```console
[resolution] in ["low", "middle", "high"]
[view] in ["default", "improved"]
```

By above commands respectively for hang-mug and pack-battery, you can generate data in ''dataset_dir'' following your own settings.

For arguements, end_idx - start_idx means the total number of episodes. ''resolution'' and ''view'' mean the camera parameters. 

Note that considering training load, low resolution is more suitable for rgb-based diffusion policy. And "improved" view is designed for pack-battery task specifically for better performance of baselines.

### Generate Large-Scale Data
For our own typical setting, we use bash commands to generate large-scale data.
```console
bash data_collect_[OBS]_[ATTN]_[TASK].sh
```
and
```console
OBS in ["pcd", "rgb"]
ATTN in ["attn","none"]
TASK in ["battery", "mug"]
```
You can find all typical combinations in the main directory.

## :gear: Train

### Train in Simulation
To run training, a training config needs to be specified.

A typical training script is like:
```console
python diffusion_policy_code/general_dp/train.py \
        --config-dir=diffusion_policy_code/general_dp/config \
        --config-name=hang_mug_act_rgb_attn.yaml \
        training.seed=42 \
        training.device=cuda:0 \
        hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

You can find all existing configs in 
```console
diffusion_policy_code/general_dp/config
```

The filename of training config has a similar structure
```console
[TASK]_[POLICY]_[OBS]_[ATTN].yaml
```
and
```console
TASK in ["pack_battery", "hang_mug"]
POLICY in ["dp", "ACT"]
OBS in ["pcd", "rgb"]
ATTN in ["attn","none"]
```

All feasible training combination can be found in main directory.
```console
train_[TASK]_[POLICY]_[OBS]_[ATTN].sh
```


### Config Exploration

To run these training bash in local machines, you need to make sure the path is correct although relative path has been used:
```console
_target_: diffusion_policy.workspace.train_act_workspace.TrainACTWorkspace
is_real: false
robot_name: panda
data_root: diffusion_policy_code/general_dp # path for "general_dp"
trial_name: hang_mug_act_rgb_attn
dataset_dir: ./dataset/rgb_attn_mug
output_dir: ${dataset_dir}/${trial_name}
......
```

For observation settings, you can find it in 
```console
d3fields_2d:
  shape:
    - 4 # C
    - 1600 # N
  type: rgb
  info:
    init_name: mug
    tgt_name: branch
......
```
or
```console
  d3fields:
    shape:
      - 4 # C
      - 1600 # N
    type: spatial
    info:
      init_name: mug
      tgt_name: branch
      task_name: hang_mug
......
```
for RGB and PCD attention respectively.

## :video_game: Infer in Simulation
To run an existing policy in the simulator, use the following command:
```console
python eval.py --checkpoint [ckpt_path] -o [eval_result_path]
```

This command will generate a directory in your argument "eval_result_path".

```console
eval_results/
├── instruction_logs/     # results of attention grounding (if attn is not None)
│   ├── attn_eval.json    # success rate of attention grounding (not whole policy)
│   ├── log_0.txt         # generated code
│   ├── log_1.txt
│   └── ... (more log_[idx].txt)
├── meidia/               # videos for all evaluation episodes (if attn is not None)
│   ├── test_0.mp4
│   ├── test_1.mp4
│   └── ... (more test_[idx].mp4)
└── env_attn.json         # success rate of whole policy
```




## :pray: Acknowledgement

This repository is built upon the following repositories. Thanks for their great work!
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [robomimic](https://github.com/ARISE-Initiative/robomimic)
- [D3Fields](https://github.com/WangYixuan12/d3fields)