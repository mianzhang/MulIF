

[ -f .env ] && export $(grep -v '^#' .env | xargs)
export ROOT_DIR=${ROOT_DIR:-$(pwd)}
export AZURE_STORAGE_ROOT=${AZURE_STORAGE_ROOT:-}

if [ -n "$AZURE_STORAGE_ROOT" ]; then
    DEFAULT_HF_HOME="$AZURE_STORAGE_ROOT/hf_home"
    DEFAULT_HF_CACHE_DIR="$AZURE_STORAGE_ROOT/hf_cache"
    DEFAULT_DATA_DIR="$AZURE_STORAGE_ROOT/verl_data"
    DEFAULT_RUN_LOG_DIR="$AZURE_STORAGE_ROOT/logs/stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100"
    DEFAULT_WANDB_DIR="$AZURE_STORAGE_ROOT/wandb"
    DEFAULT_CKPT_DIR="$AZURE_STORAGE_ROOT/checkpoints/stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100"
    DEFAULT_PROFILE_DIR="$AZURE_STORAGE_ROOT/profile/stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100"
else
    DEFAULT_HF_HOME="$ROOT_DIR/.hf_home"
    DEFAULT_HF_CACHE_DIR="$ROOT_DIR/hf_cache"
    DEFAULT_DATA_DIR="$ROOT_DIR/verl_data"
    DEFAULT_RUN_LOG_DIR="$ROOT_DIR/log"
    DEFAULT_WANDB_DIR="$ROOT_DIR/wandb"
    DEFAULT_CKPT_DIR="$ROOT_DIR/checkpoints/MulIF/stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100"
    DEFAULT_PROFILE_DIR="$ROOT_DIR/outputs/profile/stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100"
fi

export HF_HOME=${HF_HOME:-$DEFAULT_HF_HOME}
export HF_CACHE_DIR=${HF_CACHE_DIR:-$DEFAULT_HF_CACHE_DIR}
export DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}
export RUN_LOG_DIR=${RUN_LOG_DIR:-$DEFAULT_RUN_LOG_DIR}
export WANDB_DIR=${WANDB_DIR:-$DEFAULT_WANDB_DIR}
export CKPT_DIR=${CKPT_DIR:-$DEFAULT_CKPT_DIR}
export PROFILE_DIR=${PROFILE_DIR:-$DEFAULT_PROFILE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
n_gpus_per_node=8
MODEL_PATH=$HF_CACHE_DIR/billmianz/qwen7b_sft_mh_100
mkdir -p "$RUN_LOG_DIR" "$WANDB_DIR" "$HF_HOME" "$HF_CACHE_DIR" "$DATA_DIR" "$CKPT_DIR" "$PROFILE_DIR"
export DEBUG_SAMPLES=10


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/IFTrain_3c_5c.parquet \
    data.val_files="[$DATA_DIR/IFBench.parquet]" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=ifverify_sg_score \
    reward_model.launch_reward_fn_async=False \
    +reward_model.reward_kwargs.inst_weight_mode=linear \
    +reward_model.reward_kwargs.pairwise_alpha=2.0 \
    trainer.critic_warmup=0.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MulIF' \
    trainer.experiment_name='stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$CKPT_DIR \
    global_profiler.save_path=$PROFILE_DIR \
    trainer.save_freq=10 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.resume_mode=auto \
    trainer.resume_from_path=null \
    trainer.test_freq=25 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=400 > "$RUN_LOG_DIR/stage2_gz32_linear_base_pairwise_alpha2_qwen7b_sft_mh_100.log"
