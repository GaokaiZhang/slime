#!/bin/bash
# SWE-bench GRPO Training with SLiME
#
# Search-R1 GRPO hyperparameters adapted for SWE-bench training.
# Reference: https://github.com/PeterGriffinJin/Search-R1
#
# Prerequisites:
# 1. Start vLLM server (Terminal 1) - Modal or Local:
#    Modal: modal deploy examples/qwen_swe/modal_inference.py
#    Local: python examples/qwen_swe/start_vllm.py --port 8000 --tp 8
#
# 2. Docker daemon running with SWE-bench images
#
# Usage:
#    # With Modal GPU:
#    export VLLM_URL="https://susvibes-mitigation--qwen-swe-inference-serve-vllm.modal.run"
#    bash examples/qwen_swe/run_grpo.sh
#
#    # With local GPU:
#    bash examples/qwen_swe/run_grpo.sh

set -e

cd /home/gaokaizhang/slime

# ============================================
# Model Configuration
# ============================================
export VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_FILE="${INSTANCE_FILE:-$SCRIPT_DIR/data/train_201_django.txt}"

# ============================================
# Search-R1 GRPO Hyperparameters
# ============================================
# Training parameters (adapted from Search-R1)
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"           # Total rollout iterations
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}" # Prompts per rollout batch
N_SAMPLES="${N_SAMPLES:-5}"                   # Samples per prompt (n_agent in Search-R1)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-20}"  # rollout_batch_size * n_samples

# Learning rate settings (Search-R1: 1e-6 for 7B models)
LR="${LR:-1e-6}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.285}"   # From Search-R1 v0.2

# KL settings (Search-R1 GRPO uses KL loss instead of KL penalty)
USE_KL_LOSS="${USE_KL_LOSS:-true}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"         # From Search-R1
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"    # Low-variance KL approximation

# Generation settings
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"  # Full randomness for GRPO
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-8192}"

# PPO/GRPO settings
EPS_CLIP="${EPS_CLIP:-0.2}"
GAMMA="${GAMMA:-1.0}"                         # No discounting (GRPO)

# Output
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/qwen_swe_grpo}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen_swe_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_grpo_$(date +%Y%m%d_%H%M%S)}"

echo "=============================================="
echo "SWE-bench GRPO Training with SLiME"
echo "(Search-R1 Hyperparameters)"
echo "=============================================="
echo "vLLM URL: $VLLM_URL"
echo "Model: $MODEL"
echo "Instance file: $INSTANCE_FILE"
echo ""
echo "Training Parameters:"
echo "  Rollouts: $NUM_ROLLOUTS"
echo "  Batch size: $ROLLOUT_BATCH_SIZE"
echo "  Samples per prompt: $N_SAMPLES"
echo "  Global batch size: $GLOBAL_BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  LR warmup ratio: $LR_WARMUP_RATIO"
echo ""
echo "GRPO Parameters:"
echo "  Use KL loss: $USE_KL_LOSS"
echo "  KL loss coef: $KL_LOSS_COEF"
echo "  KL loss type: $KL_LOSS_TYPE"
echo "  Temperature: $ROLLOUT_TEMPERATURE"
echo "  Eps clip: $EPS_CLIP"
echo "  Gamma: $GAMMA"
echo ""
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check vLLM server
echo "Checking vLLM server..."
MAX_RETRIES=5
RETRY_COUNT=0
while ! curl -s "$VLLM_URL/v1/models" > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server not responding at $VLLM_URL after $MAX_RETRIES attempts"
        echo "Please start vLLM first:"
        echo "  Modal: modal deploy examples/qwen_swe/modal_inference.py"
        echo "  Local: python examples/qwen_swe/start_vllm.py --port 8000"
        exit 1
    fi
    echo "Waiting for vLLM server... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
done
echo "vLLM server is running"

# Check Docker
echo "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker not available"
    exit 1
fi
echo "Docker is available"

# Build training arguments
TRAIN_ARGS=(
    # Model
    --hf_checkpoint "$MODEL"
    --vllm_url "$VLLM_URL"
    --swe_instance_file "$INSTANCE_FILE"

    # Rollout settings
    --num_rollout "$NUM_ROLLOUTS"
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
    --n_samples_per_prompt "$N_SAMPLES"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --rollout_temperature "$ROLLOUT_TEMPERATURE"
    --rollout_top_p "$ROLLOUT_TOP_P"
    --rollout_max_response_len "$ROLLOUT_MAX_RESPONSE_LEN"
    --rollout_max_prompt_len "$ROLLOUT_MAX_PROMPT_LEN"

    # Training settings
    --train_backend fsdp
    --advantage_estimator grpo
    --lr "$LR"
    --eps_clip "$EPS_CLIP"
    --gamma "$GAMMA"

    # KL settings (Search-R1 style)
    --use_kl_loss
    --kl_loss_coef "$KL_LOSS_COEF"
    --kl_loss_type "$KL_LOSS_TYPE"

    # GPU settings
    --actor_num_nodes 1
    --actor_num_gpus_per_node 8

    # Output
    --save "$OUTPUT_DIR"
)

# Add optional wandb logging
if [ -n "$WANDB_PROJECT" ]; then
    TRAIN_ARGS+=(
        --use_wandb
        --wandb_project "$WANDB_PROJECT"
    )
fi

# Run training with SLiME
echo ""
echo "Starting training..."
python examples/qwen_swe/run_qwen_swe.py "${TRAIN_ARGS[@]}"

echo "=============================================="
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Optional: Stop Modal server after training
if [[ "$VLLM_URL" == *"modal.run"* ]]; then
    echo ""
    echo "To stop Modal server:"
    echo "  modal app stop qwen-swe-inference"
fi
