#!/bin/bash
# SWE-bench GRPO Training with SLiME
#
# Prerequisites:
# 1. Start vLLM server (Terminal 1):
#    python examples/qwen_swe/start_vllm.py --port 8000 --tp 8
#
# 2. Docker daemon running with SWE-bench images
#
# Usage:
#    bash run_grpo.sh

set -e

cd /home/gaokaizhang/slime

# Configuration
export VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
INSTANCE_FILE="${INSTANCE_FILE:-/home/gaokaizhang/SWE-sft/data/raw/splits/train_201_django.txt}"

# Training parameters
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/qwen_swe_grpo}"

echo "=============================================="
echo "SWE-bench GRPO Training with SLiME"
echo "=============================================="
echo "vLLM URL: $VLLM_URL"
echo "Model: $MODEL"
echo "Instance file: $INSTANCE_FILE"
echo "Rollouts: $NUM_ROLLOUTS"
echo "Batch size: $ROLLOUT_BATCH_SIZE"
echo "Samples per prompt: $N_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check vLLM server
echo "Checking vLLM server..."
if ! curl -s "$VLLM_URL/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at $VLLM_URL"
    echo "Please start vLLM first:"
    echo "  python examples/qwen_swe/start_vllm.py --port 8000"
    exit 1
fi
echo "vLLM server is running"

# Check Docker
echo "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker not available"
    exit 1
fi
echo "Docker is available"

# Run training with SLiME
python examples/qwen_swe/run_qwen_swe.py \
    --hf_checkpoint "$MODEL" \
    --vllm_url "$VLLM_URL" \
    --swe_instance_file "$INSTANCE_FILE" \
    --num_rollout "$NUM_ROLLOUTS" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --n_samples_per_prompt "$N_SAMPLES" \
    --save "$OUTPUT_DIR" \
    --train_backend fsdp \
    --advantage_estimator grpo \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8

echo "=============================================="
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "=============================================="
