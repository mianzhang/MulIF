python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/sg_baseline_rule_qwen17b_step125 \
    --target_dir hf_cache/sg_baseline_rule_qwen17b_step125