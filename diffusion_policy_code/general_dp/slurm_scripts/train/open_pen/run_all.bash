#!/bin/bash
sbatch original_dp.slurm
sbatch original_dp_rgbd.slurm
sbatch no_seg_no_dino_N_1000.slurm
sbatch no_seg_distilled_dino_N_1000.slurm

