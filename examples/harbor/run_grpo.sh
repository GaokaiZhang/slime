#!/bin/bash
# Harbor GRPO Training Script for SWE-bench
#
# Uses Search-R1 hyperparameters with Harbor's ATIF trajectory format
# for RL training on SWE-bench tasks.
#
# Prerequisites:
# - Modal vLLM server deployed (modal deploy examples/harbor/modal_vllm.py)
# - Docker available for SWE-bench environments
# - hb_train conda environment with Python 3.12

set -e

# Configuration
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
HF_CHECKPOINT="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# vLLM URL (from Modal deployment)
export VLLM_URL="${VLLM_URL:-https://susvibes-mitigation--harbor-grpo-vllm-serve-vllm.modal.run}"

# Data settings
DATASET_TYPE="django_train"  # Focus on django for training
NUM_ROLLOUTS=100
N_SAMPLES=5  # Samples per prompt (n_agent in Search-R1)

# Search-R1 GRPO hyperparameters
LR=1e-6
KL_LOSS_COEF=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0  # Full randomness for GRPO diversity
GAMMA=1.0  # No discounting

# Training settings
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_TURNS=50
EVAL_TIMEOUT=900

# Output directory
OUTPUT_DIR="outputs/harbor_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=" * 60
echo "Harbor GRPO Training for SWE-bench"
echo "=" * 60
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  vLLM URL: $VLLM_URL"
echo "  Dataset: $DATASET_TYPE"
echo "  Rollouts: $NUM_ROLLOUTS"
echo "  Samples/prompt: $N_SAMPLES"
echo "  Learning rate: $LR"
echo "  KL coefficient: $KL_LOSS_COEF"
echo "  Temperature: $TEMPERATURE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check vLLM server
echo "Checking vLLM server..."
if curl -s "$VLLM_URL/health" | grep -q "healthy"; then
    echo "vLLM server is healthy"
else
    echo "Warning: vLLM server may not be ready"
    echo "Deploy with: modal deploy examples/harbor/modal_vllm.py"
fi

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
    --vllm_url "$VLLM_URL" \
    --model_name "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --loss_mask_type "qwen3" \
    --trust_remote_code \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete. Results in: $OUTPUT_DIR"
