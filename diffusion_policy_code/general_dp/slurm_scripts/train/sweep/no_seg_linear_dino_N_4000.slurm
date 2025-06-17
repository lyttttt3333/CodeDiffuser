#!/bin/bash
#SBATCH --job-name="sweep"
#SBATCH --output="/projects/bcfs/ywang41/general_dp/general_dp/slurm_scripts/slurm_outputs/%x/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcfs-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 48:00:00

export HYDRA_FULL_ERROR=1

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate robodiff_1
cd /projects/bcfs/ywang41/general_dp/general_dp
python train.py --config-dir=config/sweep --config-name=no_seg_linear_dino.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=/scratch/bcfs/ywang41/general_dp
