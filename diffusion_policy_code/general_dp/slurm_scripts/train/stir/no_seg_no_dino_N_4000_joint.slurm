#!/bin/bash
#SBATCH --job-name="stir"
#SBATCH --output="/projects/bcfs/ywang41/general_dp/general_dp/slurm_scripts/slurm_outputs/%x/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcfs-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 48:00:00

export HYDRA_FULL_ERROR=1

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate robodiff_1
cd /projects/bcfs/ywang41/general_dp/general_dp
python train.py --config-dir=config/stir --config-name=no_seg_no_dino_N_4000_joint.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=/scratch/bcfs/ywang41/general_dp
