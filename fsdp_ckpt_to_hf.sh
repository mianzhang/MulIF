python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/qwen17b-mh-sft1/global_step_176 \
    --target_dir hf_cache/qwen17b-mh-sft1
