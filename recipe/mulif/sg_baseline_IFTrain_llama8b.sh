

export $(grep -v '^#' .env | xargs)
export CUDA_VISIBLE_DEVICES=2,3
n_gpus_per_node=2
MODEL_PATH=$HF_CACHE_DIR/Llama-3.1-8B-Instruct
export DEBUG_SAMPLES=10


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$ROOT_DIR/verl_data/IFTrain_3c_5c.parquet \
    data.val_files="[$ROOT_DIR/verl_data/IFBench.parquet]" \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=ifverify_sg_score \
    reward_model.launch_reward_fn_async=False \
    +reward_model.reward_kwargs.inst_weight_mode=none \
    trainer.critic_warmup=0.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MulIF' \
    trainer.experiment_name='sg_baseline_IFTrain_llama8b' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.resume_mode=disable \
    trainer.resume_from_path=null \
    trainer.test_freq=25 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=400 > log/sg_baseline_IFTrain_llama8b.log

    # actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \