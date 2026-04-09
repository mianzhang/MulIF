python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/MulIF/sg_rank_w0.5_qwen17b/global_step_25/actor \
    --target_dir hf_cache/test_qwen17b