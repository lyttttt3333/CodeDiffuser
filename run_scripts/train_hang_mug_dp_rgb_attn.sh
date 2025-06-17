export variable HYDRA_FULL_ERROR=1
python diffusion_policy_code/general_dp/train.py \
        --config-dir=diffusion_policy_code/general_dp/config \
        --config-name=hang_mug_dp_rgb_attn.yaml \
        training.seed=42 \
        training.device=cuda:0 \
        hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'