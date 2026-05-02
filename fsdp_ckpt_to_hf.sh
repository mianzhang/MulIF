python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/qwen17b-sft2/global_step_482 \
    --target_dir hf_cache/qwen17b-sft2
