python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/single_hard_sft_lr5e6_bz16_epoch1/global_step_153 \
    --target_dir hf_cache/qwen17b_sft2