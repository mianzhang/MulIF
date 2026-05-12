

for model in stage2_baseline_qwen7b_sft_mh_100; do

for step in 100 200 300 400; do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${AZURE_STORAGE_ROOT}/checkpoints/${model}/global_step_${step}/actor/ \
    --target_dir ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step}
python hf_upload.py ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step} --repo-id billmianz/${model}_step${step} --repo-type model
done
done

for model in stage2_linear_base_qwen7b_sft_mh_100; do

for step in 300 400; do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${AZURE_STORAGE_ROOT}/checkpoints/${model}/global_step_${step}/actor/ \
    --target_dir ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step}
python hf_upload.py ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step} --repo-id billmianz/${model}_step${step} --repo-type model
done
done
