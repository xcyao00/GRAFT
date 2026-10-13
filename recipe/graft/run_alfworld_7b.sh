set -e
ENGINE=${1:-vllm}

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # This is the logical device IDs
# === vLLM-Ascend 开关 ===
export VLLM_USE_V1=1
# export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1

export VLLM_LOGGING_LEVEL=ERROR         # suppress vLLM verbose output
export VLLM_CONFIGURE_LOGGING=0         # disable vLLM's logging configuration entirely (vLLM v1)
export VLLM_DISABLE_TQDM=1              # disable vLLM tqdm progress bars
export RAY_DEBUG=legacy
export PYTHONUNBUFFERED=1
DEVICE="npu"

num_cpus_per_env_worker=0.2

train_data_size=16
val_data_size=128
group_size=8
loss_mode="grpo-c"
clip_low=0.01
clip_high=0.01

# ─────────────── Graft mode switches ───────────────
GRAFT_VALUE_BASED=${GRAFT_VALUE_BASED:-true}
GRAFT_BELLMAN_MODE=${GRAFT_BELLMAN_MODE:-action_level}

# ─────────────── Advantage normalization scope ───────────────
# ``GRAFT_ADV_NORM_SCOPE`` selects the step-level advantage normalization
# group. 
#   - node : group = (task uid, anchor_obs)  (original behaviour). Steps
#            only compete against other steps sharing the exact same graph
#            node within the same task group. 
#   - task : group = task uid only (all steps of all trajectories in the
#            same task group pooled together).
GRAFT_ADV_NORM_SCOPE=${GRAFT_ADV_NORM_SCOPE:-node}

# ─────────────── GAE-based step advantage switches ───────────────
# ``GRAFT_GAE_MODE`` selects how per-step advantages are built:
#   - none      : keep the original TD(0) + episode-advantage pipeline
#   - gae_graph_multihop : multi-hop edge-global GAE. For each
#                 executed edge (s0->s1), TD(0) is the specific edge's TD
#                 error; TD(k) for k>=1 is the GLOBAL uniform mean over ALL
#                 raw edge instances reachable from s1 in exactly k hops
#                 (every edge instance gets equal weight 1/|E_k|, regardless
#                 of node branching structure). The final advantage is
#                 sum_k (gamma*lam)^k * TD(k). 
GRAFT_GAE_MODE=${GRAFT_GAE_MODE:-gae_graph_multihop}
GRAFT_GAE_LAMBDA=${GRAFT_GAE_LAMBDA:-0.95}

# ─────────────── Graph data saving ───────────────
# Save graph nodes/edges/V-values to JSON every N steps (0 = disabled).
GRAFT_GRAPH_SAVE_FREQ=${GRAFT_GRAPH_SAVE_FREQ:-0}
# Directory to save graph JSON files. Empty = <default_local_dir>/graph_data/
GRAFT_GRAPH_SAVE_DIR=${GRAFT_GRAPH_SAVE_DIR:-"$PWD/vis_data/alfworld"}


GRAFT_GAMMA=${GRAFT_GAMMA:-0.99}

if [ "$GRAFT_GAE_MODE" != "none" ]; then
    GRAFT_VALUE_BASED=true
    GRAFT_STEP_NORM_MODE=${GRAFT_STEP_NORM_MODE:-mean_std_norm}
    GRAFT_GAMMA=${GRAFT_GAMMA:-0.99}
else
    GRAFT_STEP_NORM_MODE=${GRAFT_STEP_NORM_MODE:-mean_std_norm}
fi

project_name='alfworld'
experiment_name="graft_qwen2.5_7b"
log_dir="$PWD/exps/${project_name}/${experiment_name}"
mkdir -p "$log_dir"
log_file="${log_dir}/train_log.log"
> "${log_file}"

python3 -m recipe.graft.main_graft \
    algorithm.adv_estimator=graft \
    algorithm.gamma=$GRAFT_GAMMA \
    algorithm.graft.step_advantage_w=1.0 \
    algorithm.graft.episode_advantage_w=0.0 \
    algorithm.graft.mode=$GRAFT_STEP_NORM_MODE \
    algorithm.graft.advantage_norm_scope=$GRAFT_ADV_NORM_SCOPE \
    algorithm.graft.normalize_distance=False \
    algorithm.graft.value_based=$GRAFT_VALUE_BASED \
    algorithm.graft.bellman_mode=$GRAFT_BELLMAN_MODE \
    algorithm.graft.gae_mode=$GRAFT_GAE_MODE \
    algorithm.graft.gae_lambda=$GRAFT_GAE_LAMBDA \
    algorithm.graft.graph_save_freq=$GRAFT_GRAPH_SAVE_FREQ \
    algorithm.graft.graph_save_dir=$GRAFT_GRAPH_SAVE_DIR \
    algorithm.use_kl_in_reward=False \
    data.train_files=$PWD/text/train.parquet \
    data.val_files=$PWD/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$PWD/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.clip_ratio_low=$clip_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_high \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.device=$DEVICE \
    trainer.val_before_train=False \
    trainer.default_local_dir="$PWD/exps/${project_name}/${experiment_name}" \
    ray_init.num_cpus=64 $@ 2>&1 | grep -v "Finished request" | tee -a "$log_file" || true
