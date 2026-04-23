python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/IFTrain_noconf_baseline_qwen17b/global_step_500 \
    --target_dir hf_cache/IFTrain_noconf_baseline_qwen17b_step500
