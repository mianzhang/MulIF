#!/usr/bin/env bash

# Single-node SFT for Qwen3-1.7B on IFVerify-style parquet data.
# Expected dataset schema:
#   - id: string
#   - messages: list<struct<role: string, content: string>>

export CUDA_VISIBLE_DEVICES=3
torchrun --standalone --nnodes=1 --nproc-per-node=2 -m verl.trainer.sft_trainer \
    data.train_files="verl_data/rebulttal_random_sft_data.parquet" \
    data.val_files=null \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=2048 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.messages_key=messages \
    +data.default_enable_thinking=False \
    model.path=hf_cache/Qwen3-1.7B \
    model.use_remove_padding=True \
    optim.lr=5e-6 \
    optim.weight_decay=0.1 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.clip_grad=1.0 \
    optim.warmup_style=cosine \
    trainer.project_name="MulIF" \
    trainer.experiment_name="rebulttal_random_sft" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="checkpoints/rebulttal_random_sft" \
    trainer.resume_mode="auto" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=5 \
    trainer.logger=["console","wandb"] \
