# Stop any existing Ray instances and start fresh Ray head node with 2 GPUs
# ray stop
# ray start --head --num-gpus=2 --disable-usage-stats

# MODEL_PATH=Qwen/Qwen3-0.6B
MODEL_PATH=Qwen/Qwen3-1.7B
# EXPERIMENT_NAME=qwen3_0_6b_gh_patches
EXPERIMENT_NAME=qwen3_1_7b_gh_patches
TRAIN_FILE=/root/persistent/data/github_patches/train.parquet
TEST_FILE=/root/persistent/data/github_patches/test.parquet

# MAX_PROMPT_LENGTH=512
MAX_PROMPT_LENGTH=4096
# MAX_PROMPT_LENGTH=-1
# MAX_RESPONSE_LENGTH=1024
# MAX_RESPONSE_LENGTH=16384
MAX_RESPONSE_LENGTH=4096
# MAX_RESPONSE_LENGTH=8192

# TRAIN_BATCH_SIZE=1024
# TRAIN_BATCH_SIZE=128
# TRAIN_BATCH_SIZE=64
# TRAIN_BATCH_SIZE=16
# TRAIN_BATCH_SIZE=4
TRAIN_BATCH_SIZE=2

# PPO_MINI_BATCH_SIZE=80
# PPO_MICRO_BATCH_SIZE_PER_GPU=20
# PPO_MINI_BATCH_SIZE=16
# PPO_MICRO_BATCH_SIZE_PER_GPU=4
# PPO_MINI_BATCH_SIZE=8
# PPO_MICRO_BATCH_SIZE_PER_GPU=2
# PPO_MINI_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=2
# PPO_MICRO_BATCH_SIZE_PER_GPU=2
PPO_MICRO_BATCH_SIZE_PER_GPU=1
# LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=20
# LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=4 \
    trainer.project_name='verl_gh_patches' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.default_local_dir="/root/persistent/checkpoints/$EXPERIMENT_NAME" \
    trainer.test_freq=1 \
    trainer.total_epochs=1 $@
