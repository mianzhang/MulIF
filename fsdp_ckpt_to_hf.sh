python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/single_hard_sft_s100_lr1e5_bz16_epoch1/global_step_43 \
    --target_dir hf_cache/single_hard_sft_s100_lr1e5_bz16_epoch1