# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/qwen17b_sft_hard_100/global_step_18 \
#     --target_dir hf_cache/qwen17b_sft_hard_100

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/qwen17b_sft_hard_500/global_step_87 \
#     --target_dir hf_cache/qwen17b_sft_hard_500

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/qwen17b_sft_mh_100/global_step_50 \
#     --target_dir hf_cache/qwen17b_sft_mh_100

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/qwen17b_sft_mh_500/global_step_233 \
#     --target_dir hf_cache/qwen17b_sft_mh_500

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/qwen17b_sft_all_100/global_step_93 \
#     --target_dir hf_cache/qwen17b_sft_all_100

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/qwen17b_sft_all_500/global_step_451 \
#     --target_dir hf_cache/qwen17b_sft_all_500

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/rebuttal_random_sft/global_step_56 \
#     --target_dir hf_cache/rebuttal_random_sft

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/rebuttal_bestN_sft/global_step_56 \
#     --target_dir hf_cache/rebuttal_bestN_sft