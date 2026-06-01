

for model in stage2_data_ablation_qwen17b_sft_mh_100 stage2_gz8_baseline_qwen17b stage2_gz8_linear_base_pairwise_alpha2_qwen7b_sft_mh_100 stage2_gz32_baseline_qwen17b stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100; do

for step in 200 300; do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${AZURE_STORAGE_ROOT}/checkpoints/${model}/global_step_${step}/actor/ \
    --target_dir ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step}
python hf_upload.py ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step} --repo-id billmianz/${model}_step${step} --repo-type model
done
done
