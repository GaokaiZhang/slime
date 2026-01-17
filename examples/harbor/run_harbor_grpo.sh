#!/bin/bash
# Harbor GRPO Training Script
#
# This script runs Harbor + SLiME GRPO training.
# Architecture:
# - Harbor CLI: Agent rollouts (mini-swe-agent, qwen-coder)
# - Modal A100: GRPO weight updates
# - Local Docker: swebench.harness evaluation
#
# Usage:
#   # Test mode (5 instances)
#   bash examples/harbor/run_harbor_grpo.sh --test
#
#   # Full training (201 Django instances)
#   bash examples/harbor/run_harbor_grpo.sh
#
#   # With Modal CLI
#   modal run examples/harbor/harbor_grpo_trainer.py --test

set -e

# Change to slime root
cd "$(dirname "$0")/../.."
SLIME_ROOT=$(pwd)

# Default settings
NUM_ROLLOUTS=50
N_SAMPLES=4
AGENT="mini-swe-agent"
MODEL="Kwai-Klear/Klear-AgentForge-8B-SFT"
OUTPUT_DIR="outputs/harbor_grpo"
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --num-rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================"
echo "Harbor GRPO Training"
echo "============================================"
echo "Slime root: $SLIME_ROOT"
echo "Agent: $AGENT"
echo "Model: $MODEL"
echo "Rollouts: $NUM_ROLLOUTS"
echo "Samples per prompt: $N_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "Test mode: $TEST_MODE"
echo "============================================"

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check Harbor
if ! command -v harbor &> /dev/null; then
    echo "ERROR: Harbor not found. Install with: uv tool install harbor"
    exit 1
fi
echo "  Harbor: $(harbor --version)"

# Check Docker
if ! docker info &> /dev/null; then
    echo "ERROR: Docker not running"
    exit 1
fi
echo "  Docker: Running"

# Check SWE-bench images
SWEBENCH_IMAGES=$(docker images --format "{{.Repository}}" | grep -c swebench || true)
echo "  SWE-bench images: $SWEBENCH_IMAGES"

# Check Modal
if ! command -v modal &> /dev/null; then
    echo "WARNING: Modal CLI not found. Install with: pip install modal"
fi

# Run tests first
echo ""
echo "Running integration tests..."
PYTHONPATH=$SLIME_ROOT TOKENIZERS_PARALLELISM=false python examples/harbor/test_harbor_slime.py

if [ $? -ne 0 ]; then
    echo "ERROR: Integration tests failed"
    exit 1
fi

# Start training
echo ""
echo "Starting Harbor GRPO training..."

# Build command
CMD="PYTHONPATH=$SLIME_ROOT python examples/harbor/harbor_grpo_trainer.py"
CMD="$CMD --num-rollouts $NUM_ROLLOUTS"
CMD="$CMD --n-samples $N_SAMPLES"
CMD="$CMD --agent $AGENT"
CMD="$CMD --model $MODEL"
CMD="$CMD --output-dir $OUTPUT_DIR"

if [ "$TEST_MODE" = true ]; then
    CMD="$CMD --test"
fi

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
