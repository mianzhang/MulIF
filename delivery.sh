python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/IFTrain_noconf_baseline_qwen17b/global_step_500/actor/ \
    --target_dir hf_cache/IFTrain_noconf_baseline_qwen17b_step500
python hf_upload.py hf_cache/IFTrain_noconf_baseline_qwen17b_step500 --repo-id billmianz/IFTrain_noconf_baseline_qwen17b_step500 --repo-type model

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/IFTrain_noconf_linear_base_qwen17b/global_step_500/actor/ \
    --target_dir hf_cache/IFTrain_noconf_linear_base_qwen17b_step500
python hf_upload.py hf_cache/IFTrain_noconf_linear_base_qwen17b_step500 --repo-id billmianz/IFTrain_noconf_linear_base_qwen17b_step500 --repo-type model
