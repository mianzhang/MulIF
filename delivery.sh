

for model in rebuttal_SaR_qwen7b rebuttal_SaR_qwen17b_random_sft rebuttal_SaR_qwen17b_bestN_sft; do

for step in 100 200 300 400; do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${AZURE_STORAGE_ROOT}/checkpoints/${model}/global_step_${step}/actor/ \
    --target_dir ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step}
python hf_upload.py ${AZURE_STORAGE_ROOT}/hf_cache/${model}_step${step} --repo-id billmianz/${model}_step${step} --repo-type model
done
done
