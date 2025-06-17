"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import os 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # set device id
    # device_id = cfg.training.device_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # append dataset dir for sapien_env
    if hasattr(cfg.task, 'dataset_dir'):
        dataset_dir = cfg.task.dataset_dir
        sys.path.append(dataset_dir)

    cls = hydra.utils.get_class(cfg._target_)
    os.system(f'mkdir -p {cfg.output_dir}')
    workspace: BaseWorkspace = cls(cfg, output_dir=cfg.output_dir)
    workspace.run()


if __name__ == "__main__":
    main()

"""
python train.py --config-dir=/home/yitong/diffusion/diffusion_policy_code/general_dp/config --config-name=hang_mug_attn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python /root/lyt/diffusion_policy_code/general_dp/train.py --config-dir=/root/lyt/diffusion_policy_code/general_dp/config --config-name=discrete.yaml training.seed=42 training.device=cuda:1 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
"""