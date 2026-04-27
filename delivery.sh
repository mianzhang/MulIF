

for model in IFTrain_baseline_qwen17b IFTrain_linear_base_qwen17b IFTrain_baseline_qwen17b_sft1 IFTrain_linear_base_qwen17b_sft1; do

for step in 100 200 300 400; do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${AZURE_STORAGE_ROOT}/checkpoints/${model}/global_step_${step}/actor/ \
    --target_dir ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step}
python hf_upload.py ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step} --repo-id billmianz/${model}_step${step} --repo-type model
done
done
