#!/bin/bash
set -x

# Set environment variable to get full error details
export HYDRA_FULL_ERROR=1

ulimit -n 65536
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_LOG_LEVEL=DEBUG

pip install cydifflib unidiff

# Stop the running ray cluster
# ray stop



MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_PATH="Qwen/Qwen3-14B"
# MODEL_PATH="Qwen/Qwen3-8B"


# TRAIN_FILE=~/persistent/data/github_patches_2k/train.parquet
# TEST_FILE=~/persistent/data/github_patches_2k/test.parquet
# TRAIN_FILE=~/persistent/data/swe_smith_oracle_4k/train.parquet
# TEST_FILE=~/persistent/data/swe_smith_oracle_4k/test.parquet
# TRAIN_FILE=~/persistent/data/r2e-gym-subset-oracle-4k/train.parquet
# TEST_FILE=~/persistent/data/r2e-gym-subset-oracle-4k/test.parquet
TRAIN_FILE=~/persistent/data/swe_rl_8k/train.parquet
TEST_FILE=~/persistent/data/swe_rl_8k/test.parquet

# EXPERIMENT_NAME=deepseek_r1_7b_gh_patches_2k_fixed_reward
# EXPERIMENT_NAME=deepseek_r1_qwen3_8b_swe_smith_oracle_4k
# EXPERIMENT_NAME=qwen3_8b_r2e_gym_subset_oracle_4k
# EXPERIMENT_NAME=qwen3_8b_swe_rl_8k
# EXPERIMENT_NAME=qwen3_14b_swe_rl_8k
EXPERIMENT_NAME=deepseek_r1_7b_swe_rl_8k

# TRAIN_BATCH_SIZE=8
TRAIN_BATCH_SIZE=32
# TRAIN_BATCH_SIZE=32
# TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=8
PPO_MICRO_BATCH_SIZE=8
# PPO_MINI_BATCH_SIZE=16
# PPO_MICRO_BATCH_SIZE=16
# PPO_MAX_TOKEN_LEN_PER_GPU=36864
# PPO_MAX_TOKEN_LEN_PER_GPU=36864
# PPO_MAX_TOKEN_LEN_PER_GPU=40960
PPO_MAX_TOKEN_LEN_PER_GPU=32768
# PPO_MAX_TOKEN_LEN_PER_GPU=16384
# PPO_MAX_TOKEN_LEN_PER_GPU=40000
# MAX_NUM_BATCHED_TOKENS=36864
# MAX_NUM_BATCHED_TOKENS=40000
# MAX_NUM_BATCHED_TOKENS=40960
MAX_NUM_BATCHED_TOKENS=32768
# MAX_NUM_BATCHED_TOKENS=16384


# MAX_PROMPT_LENGTH=2048
# MAX_PROMPT_LENGTH=4096
MAX_PROMPT_LENGTH=8192
# MAX_RESPONSE_LENGTH=16384
# MAX_RESPONSE_LENGTH=8192
# MAX_RESPONSE_LENGTH=32768
MAX_RESPONSE_LENGTH=24576

TP_SIZE=1
SP_SIZE=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=512 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    actor_rollout_ref.rollout.disable_log_stats=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=4 \
    trainer.project_name='swe-rl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=2 \
    trainer.default_local_dir="/root/persistent/checkpoints/$EXPERIMENT_NAME" \
    trainer.validation_data_dir="/root/persistent/rollouts/$EXPERIMENT_NAME/validation" \
    trainer.rollout_data_dir="/root/persistent/rollouts/$EXPERIMENT_NAME/train" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 $@

    # trainer.val_before_train=False \
