#!/bin/bash
#SBATCH --job-name="act_mug_text_180"
#SBATCH --output="/projects/bcfs/ywang41/baseline/slurm_output/%x/%j.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64  
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcyd-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 18:00:00

export HYDRA_FULL_ERROR=1

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate robodiff_bl
python train.py --config-dir=config --config-name=hang_mug_act.yaml training.seed=0 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'