python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/llama8b-sft1/global_step_108 \
    --target_dir hf_cache/llama8b-sft1
