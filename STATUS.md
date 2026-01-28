# Harbor GRPO Testing Status

## Goal
Test Harbor GRPO with both SWE-bench and C2Bug data sources using Qwen/Qwen3-Coder-30B-A3B-Instruct model with qwen-coder agent.

## Environment
- **GPU**: 8x NVIDIA H100 80GB HBM3
- **Docker**: Available
- **Harbor CLI**: v0.1.42
- **Model**: Qwen/Qwen3-Coder-30B-A3B-Instruct

## Architecture

### vLLM Server (Inference - GPUs 0-3)
- **Status**: ✓ Running
- **Endpoint**: http://localhost:8000/v1
- **Log**: vllm_4gpu.log

### GRPO Training (GPUs 4-7)
- **Agent**: qwen-coder
- **Group size**: 4
- **LR**: 1e-6, KL: 0.001

## Tests

### 1. SWE-bench (Running)
- **PID**: 60376
- **Log**: grpo_swebench.log
- **Instances**: 3 from train_instances_id.txt
- **Output**: outputs/test_swebench

### 2. C2Bug (Pending)
- **Script**: run_c2bug_grpo.sh
- **Dataset**: TwelfthStar/c2bug_tasks_django_Jan-22-2026
- **Instances**: 3
- **Output**: outputs/test_c2bug

## Status
- SWE-bench: 🔄 In progress
- C2Bug: ⏳ Waiting
