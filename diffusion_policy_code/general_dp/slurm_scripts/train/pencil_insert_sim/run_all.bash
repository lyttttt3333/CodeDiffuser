#!/bin/bash
sbatch original_dp.slurm
sbatch no_seg_no_dino_N_4000.slurm
sbatch no_seg_distilled_dino_N_4000.slurm
sbatch act.slurm
sbatch act_no_dino.slurm
sbatch act_distill_dino.slurm

