python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/IFTrain_baseline_qwen17b_sft1/global_step_500/actor/ \
    --target_dir hf_cache/IFTrain_baseline_qwen17b_sft1_step500
python hf_upload.py hf_cache/IFTrain_baseline_qwen17b_sft1_step500 --repo-id billmianz/IFTrain_baseline_qwen17b_sft1_step500 --repo-type model

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/IFTrain_baseline_qwen17b_sft1/global_step_200/actor/ \
    --target_dir hf_cache/IFTrain_baseline_qwen17b_sft1_step200
python hf_upload.py hf_cache/IFTrain_baseline_qwen17b_sft1_step200 --repo-id billmianz/IFTrain_baseline_qwen17b_sft1_step200 --repo-type model
