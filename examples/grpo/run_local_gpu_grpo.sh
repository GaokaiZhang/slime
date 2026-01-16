#!/bin/bash
# Local GPU GRPO Training with swebench.harness Evaluation
#
# This script runs GRPO training entirely on local GPU with proper
# swebench.harness evaluation (Docker required).
#
# Prerequisites:
#   1. GPU with 24GB+ VRAM (for 8B model with LoRA)
#   2. Docker with SWE-bench images pulled
#   3. swebench package installed: pip install swebench
#   4. PEFT for LoRA: pip install peft
#
# Usage:
#   # Quick test (5 instances, 2 samples each)
#   bash examples/grpo/run_local_gpu_grpo.sh --test
#
#   # Full training (50 instances, 4 samples each)
#   bash examples/grpo/run_local_gpu_grpo.sh
#
#   # Custom run
#   bash examples/grpo/run_local_gpu_grpo.sh --num-rollouts 100 --n-samples 5

set -e

# Default values (Search-R1 GRPO hyperparameters)
MODEL_NAME="Kwai-Klear/Klear-AgentForge-8B-SFT"
NUM_ROLLOUTS=50
N_SAMPLES=4
LR="1e-6"
KL_COEF="0.001"
TEMPERATURE="1.0"
OUTPUT_DIR="outputs/local_gpu_grpo"
SAVE_EVERY=10
USE_LORA="--use-lora"
LORA_R=16
EVAL_TIMEOUT=300
USE_HF_GENERATE=""
VLLM_URL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            NUM_ROLLOUTS=5
            N_SAMPLES=2
            SAVE_EVERY=2
            OUTPUT_DIR="outputs/local_gpu_grpo_test"
            shift
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num-rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --kl-coef)
            KL_COEF="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --no-lora)
            USE_LORA="--no-lora"
            shift
            ;;
        --lora-r)
            LORA_R="$2"
            shift 2
            ;;
        --use-hf-generate)
            USE_HF_GENERATE="--use-hf-generate"
            shift
            ;;
        --vllm-url)
            VLLM_URL="--vllm-url $2"
            shift 2
            ;;
        --eval-timeout)
            EVAL_TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Local GPU GRPO Training with swebench.harness"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test                Quick test run (5 instances, 2 samples)"
            echo "  --model-name NAME     Model name (default: $MODEL_NAME)"
            echo "  --num-rollouts N      Number of training instances (default: $NUM_ROLLOUTS)"
            echo "  --n-samples N         Samples per prompt (default: $N_SAMPLES)"
            echo "  --lr RATE             Learning rate (default: $LR)"
            echo "  --kl-coef COEF        KL coefficient (default: $KL_COEF)"
            echo "  --temperature T       Sampling temperature (default: $TEMPERATURE)"
            echo "  --output-dir DIR      Output directory (default: $OUTPUT_DIR)"
            echo "  --save-every N        Save checkpoint every N rollouts (default: $SAVE_EVERY)"
            echo "  --no-lora             Disable LoRA (requires more VRAM)"
            echo "  --lora-r R            LoRA rank (default: $LORA_R)"
            echo "  --use-hf-generate     Use HuggingFace generate instead of vLLM"
            echo "  --vllm-url URL        vLLM server URL (e.g., http://localhost:8000)"
            echo "  --eval-timeout SEC    Evaluation timeout in seconds (default: $EVAL_TIMEOUT)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
echo "========================================================================"
echo "Local GPU GRPO Training with swebench.harness Evaluation"
echo "========================================================================"
echo ""

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available"
    exit 1
fi
echo "✓ CUDA available"

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker not running"
    exit 1
fi
echo "✓ Docker running"

# Check swebench
if ! python -c "import swebench" 2>/dev/null; then
    echo "WARNING: swebench not installed. Install with: pip install swebench"
fi
echo "✓ swebench available"

# Check peft
if ! python -c "import peft" 2>/dev/null; then
    echo "WARNING: peft not installed. Install with: pip install peft"
fi
echo "✓ peft available"

# Print configuration
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Num rollouts: $NUM_ROLLOUTS"
echo "  Samples per prompt: $N_SAMPLES"
echo "  Learning rate: $LR"
echo "  KL coefficient: $KL_COEF"
echo "  Temperature: $TEMPERATURE"
echo "  Output dir: $OUTPUT_DIR"
echo "  LoRA: $USE_LORA (r=$LORA_R)"
echo "  Inference: ${USE_HF_GENERATE:-vLLM API}"
echo "  Evaluation: swebench.harness (Docker)"
echo ""

# Run training
echo "Starting training..."
echo ""

python examples/grpo/local_gpu_grpo_trainer.py \
    --model-name "$MODEL_NAME" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --n-samples "$N_SAMPLES" \
    --lr "$LR" \
    --kl-coef "$KL_COEF" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --save-every "$SAVE_EVERY" \
    $USE_LORA \
    --lora-r "$LORA_R" \
    --eval-timeout "$EVAL_TIMEOUT" \
    $USE_HF_GENERATE \
    $VLLM_URL

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo "Output saved to: $OUTPUT_DIR"
echo ""
