#!/bin/bash
# Harbor + SLiME GRPO Training Script
#
# Architecture:
# - SLiME: GRPO training (loss computation, model updates, Megatron)
# - Harbor: Agent rollouts (terminus-2 with collect_rollout_details=True)
# - Custom: Loss mask computation from Harbor's RolloutDetail
#
# Prerequisites:
# - Harbor installed: pip install -e submodules/harbor
# - Docker available for SWE-bench environments
# - Model inference endpoint (vLLM, OpenAI API, etc.)

set -e

# Configuration
MODEL="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

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

echo "============================================================"
echo "Harbor + SLiME GRPO Training"
echo "============================================================"
echo ""
echo "Architecture:"
echo "  - SLiME: GRPO training framework"
echo "  - Harbor: Agent rollouts (terminus-2), Docker environments, verification"
echo "  - Custom: Loss mask computation from RolloutDetail"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET_TYPE"
echo "  Rollouts: $NUM_ROLLOUTS"
echo "  Samples/prompt: $N_SAMPLES"
echo "  Learning rate: $LR"
echo "  KL coefficient: $KL_LOSS_COEF"
echo "  Temperature: $TEMPERATURE"
echo "  Output: $OUTPUT_DIR"
echo ""

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
    --model_name "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --trust_remote_code \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete. Results in: $OUTPUT_DIR"
