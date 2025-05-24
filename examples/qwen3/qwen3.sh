# Stop any existing Ray instances and start fresh Ray head node with 2 GPUs
ray stop
ray start --head --num-gpus=2 --disable-usage-stats

MODEL_PATH=Qwen/Qwen3-0.6B
# EXPERIMENT_NAME=qwen3_0_6b_function_rm
EXPERIMENT_NAME=qwen3_0_6b_swe_fixer
# TRAIN_FILE=data/gsm8k/train.parquet
# TEST_FILE=data/gsm8k/test.parquet
TRAIN_FILE=data/swe_fixer/train.parquet
TEST_FILE=data/swe_fixer/test.parquet

# MAX_PROMPT_LENGTH=512
MAX_PROMPT_LENGTH=4096

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=1024 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=80 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=4 \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
