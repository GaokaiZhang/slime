#!/bin/bash
# =============================================================================
# Harbor + SLiME GRPO Training Script (tmux version)
# =============================================================================
#
# This script:
# 1. Deploys Modal vLLM server with Qwen3-Coder-30B-A3B-Instruct
# 2. Runs GRPO training with Search-R1 hyperparameters on 231 Django instances
# 3. Stops Modal server when training completes
#
# IMPORTANT: Run this in the hb_train conda environment (Python 3.12+)
#
# Usage:
#   tmux new -s grpo
#   conda activate hb_train
#   bash examples/grpo/run_grpo_tmux.sh
#
# To monitor in another terminal:
#   tmux attach -t grpo
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Model - Qwen3-Coder-30B-A3B-Instruct
export MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"
export HF_CHECKPOINT="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Agent configuration
# NOTE: OpenHands does NOT provide completion_token_ids and logprobs needed for GRPO!
#       Use "terminus-2" for proper GRPO training with token IDs and logprobs.
#       Use "openhands" for evaluation/rollouts only (limited RL training).
AGENT_NAME="${AGENT_NAME:-terminus-2}"  # Default to terminus-2 for GRPO

# Data settings - 231 Django instances from SWE-Bench_Verified
DATASET_TYPE="django_train"
NUM_ROLLOUTS=231  # All Django instances
N_SAMPLES=5       # Samples per prompt (n_agent in Search-R1)

# Search-R1 GRPO hyperparameters
LR=1e-6
KL_LOSS_COEF=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0   # Full randomness for GRPO diversity
GAMMA=1.0         # No discounting

# Training settings
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_TURNS=50
EVAL_TIMEOUT=900

# Output directory
OUTPUT_DIR="outputs/grpo_${AGENT_NAME}_$(date +%Y%m%d_%H%M%S)"

# Modal app name
MODAL_APP_NAME="harbor-grpo-vllm"

# =============================================================================
# Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

cleanup() {
    log "Cleaning up..."

    # Stop Modal app
    log "Stopping Modal vLLM server..."
    modal app stop "$MODAL_APP_NAME" 2>/dev/null || true

    log "Cleanup complete."
}

# Set trap to cleanup on exit
trap cleanup EXIT

check_environment() {
    log "Checking environment..."

    # Check Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python version: $PYTHON_VERSION"

    if [[ "$PYTHON_VERSION" < "3.12" ]]; then
        log "ERROR: Python 3.12+ required for Harbor. Current: $PYTHON_VERSION"
        log "Please activate hb_train environment: conda activate hb_train"
        exit 1
    fi

    # Check Modal CLI
    if ! command -v modal &> /dev/null; then
        log "ERROR: Modal CLI not found. Install with: pip install modal"
        exit 1
    fi

    # Check if Harbor is installed
    if ! python -c "import harbor" 2>/dev/null; then
        log "WARNING: Harbor not installed. Installing..."
        pip install -e submodules/harbor
    fi

    log "Environment check passed."
}

deploy_modal() {
    log "Deploying Modal vLLM server..."
    log "  Model: $MODEL_NAME"

    # Deploy the Modal app
    cd "$(dirname "$0")/../.."  # Go to slime root
    modal deploy examples/grpo/modal_vllm.py

    # Get the URL
    log "Waiting for Modal app to be ready..."
    sleep 10

    # Try to get the URL from Modal
    MODAL_URL=$(modal app list 2>/dev/null | grep "$MODAL_APP_NAME" | awk '{print $NF}' || echo "")

    if [[ -z "$MODAL_URL" ]]; then
        # Fallback: construct URL from workspace
        MODAL_URL="https://susvibes-mitigation--${MODAL_APP_NAME}-serve-vllm.modal.run"
    fi

    export VLLM_URL="$MODAL_URL"
    log "Modal vLLM URL: $VLLM_URL"

    # Wait for server to be healthy
    log "Waiting for vLLM server to be ready (this may take 5-10 minutes for model loading)..."
    for i in {1..120}; do
        if curl -s "$VLLM_URL/health" 2>/dev/null | grep -q "healthy"; then
            log "vLLM server is ready!"
            return 0
        fi

        if (( i % 30 == 0 )); then
            log "Still waiting... ${i}s (model may be downloading/loading)"
        fi
        sleep 5
    done

    log "WARNING: vLLM server health check timed out. Proceeding anyway..."
}

run_grpo_training() {
    log "Starting GRPO training..."
    log "============================================================"
    log "Configuration:"
    log "  Agent: $AGENT_NAME"
    log "  Model: $MODEL_NAME"
    log "  Dataset: $DATASET_TYPE ($NUM_ROLLOUTS instances)"
    log "  Samples/prompt: $N_SAMPLES"
    log "  Learning rate: $LR"
    log "  KL coefficient: $KL_LOSS_COEF"
    log "  Temperature: $TEMPERATURE"
    log "  Output: $OUTPUT_DIR"
    log "============================================================"

    mkdir -p "$OUTPUT_DIR"

    # Save configuration
    cat > "$OUTPUT_DIR/config.json" << EOF
{
    "agent": "$AGENT_NAME",
    "model": "$MODEL_NAME",
    "dataset": "$DATASET_TYPE",
    "num_rollouts": $NUM_ROLLOUTS,
    "n_samples_per_prompt": $N_SAMPLES,
    "lr": $LR,
    "kl_loss_coef": $KL_LOSS_COEF,
    "kl_loss_type": "$KL_LOSS_TYPE",
    "temperature": $TEMPERATURE,
    "gamma": $GAMMA,
    "batch_size": $BATCH_SIZE,
    "gradient_accumulation_steps": $GRAD_ACCUM,
    "max_turns": $MAX_TURNS,
    "vllm_url": "$VLLM_URL",
    "started_at": "$(date -Iseconds)"
}
EOF

    # Run training
    python -m slime.train \
        --hf_checkpoint "$HF_CHECKPOINT" \
        --rollout_module "examples.harbor.rollout" \
        --rollout_fn "generate" \
        --data_source_module "examples.harbor.data_source" \
        --data_source_fn "create_data_source" \
        --data_source_kwargs "{\"dataset_type\": \"$DATASET_TYPE\", \"limit\": $NUM_ROLLOUTS}" \
        --n_samples_per_prompt $N_SAMPLES \
        --rollout_temperature $TEMPERATURE \
        --lr $LR \
        --kl_loss_coef $KL_LOSS_COEF \
        --kl_loss_type "$KL_LOSS_TYPE" \
        --gamma $GAMMA \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_turns $MAX_TURNS \
        --eval_timeout $EVAL_TIMEOUT \
        --model_name "$MODEL_NAME" \
        --agent_name "$AGENT_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --trust_remote_code \
        2>&1 | tee "$OUTPUT_DIR/training.log"

    log "Training complete!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "============================================================"
    log "Harbor + SLiME GRPO Training"
    log "============================================================"
    log ""
    log "Architecture:"
    log "  - SLiME: GRPO training (loss computation, model updates)"
    log "  - Harbor: Agent rollouts ($AGENT_NAME), Docker environments"
    log "  - Modal: vLLM inference (A100-80GB)"
    log ""

    if [[ "$AGENT_NAME" == "openhands" ]]; then
        log "WARNING: OpenHands does NOT provide completion_token_ids and logprobs!"
        log "         GRPO training may not work correctly without these."
        log "         Consider using terminus-2 for proper GRPO training."
        log ""
        read -p "Continue with OpenHands anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Switching to terminus-2..."
            AGENT_NAME="terminus-2"
        fi
    fi

    # Run pipeline
    check_environment
    deploy_modal
    run_grpo_training

    log ""
    log "============================================================"
    log "GRPO Training Complete!"
    log "============================================================"
    log "Results saved to: $OUTPUT_DIR"
    log "Modal server will be stopped automatically."
}

# Run main
main "$@"
