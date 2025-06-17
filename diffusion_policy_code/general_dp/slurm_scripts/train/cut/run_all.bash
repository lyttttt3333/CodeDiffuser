#!/bin/bash
sbatch original_dp.slurm
# sbatch original_dp_joint.slurm
sbatch no_seg_no_dino_N_4000.slurm
# sbatch no_seg_no_dino_N_4000_joint.slurm
sbatch no_seg_no_dino_N_4000_wo_tool.slurm
# sbatch no_seg_no_dino_N_4000_joint_wo_tool.slurm

