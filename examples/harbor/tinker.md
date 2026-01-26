# Harbor GRPO with Tinker GPU

This guide covers using [Tinker](https://thinkingmachines.ai/tinker/) (Thinking Machines Lab) for GRPO training on SWE-bench tasks.

## Overview

Tinker provides cloud GPU for both **inference** and **training**:
- **Inference**: Fast sampling with logprobs via `sample_async()`
- **Training**: Native PPO loss via `forward_backward(loss_fn="ppo")`

```
Tinker Cloud (GPU)
├── Inference: sample_async() with logprobs
└── Training: forward_backward(loss_fn="ppo") + optim_step()

Local Machine
├── Harbor agent rollouts (qwen-coder)
├── Docker containers for evaluation
└── Reward computation
```

## Setup

### 1. Get Tinker API Key

1. Go to https://tinker-console.thinkingmachines.ai/
2. Create an API key
3. Set environment variable:

```bash
export TINKER_API_KEY="tml-..."
```

### 2. Install Dependencies

```bash
pip install tinker openai
```

### 3. Verify Connection

```bash
python -c "
import tinker
client = tinker.ServiceClient()
caps = client.get_server_capabilities()
print('Available models:')
for m in caps.supported_models:
    print(f'  - {m.model_name}')
"
```

## Supported Models

| Model | Type | Size | Notes |
|-------|------|------|-------|
| `Qwen/Qwen3-8B` | Dense | 8B | Default, fast |
| `Qwen/Qwen3-30B-A3B` | MoE | 30B | Larger capacity |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | 30B | Instruction-tuned |
| `Qwen/Qwen3-32B` | Dense | 32B | High quality |
| `meta-llama/Llama-3.1-8B-Instruct` | Dense | 8B | Good for code |
| `meta-llama/Llama-3.3-70B-Instruct` | Dense | 70B | Best quality |
| `deepseek-ai/DeepSeek-V3.1` | MoE | 671B | State-of-the-art |
| `moonshotai/Kimi-K2-Thinking` | MoE | 1T | Reasoning focused |

## Basic Usage

### Test Mode (5 instances)

```bash
export TINKER_API_KEY="tml-..."
python examples/harbor/harbor_grpo_tinker.py --test
```

### Full Training

```bash
# With default Qwen3-8B
python examples/harbor/harbor_grpo_tinker.py --num-rollouts 50

# With larger model
python examples/harbor/harbor_grpo_tinker.py \
    --tinker-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --num-rollouts 50

# With specific agent
python examples/harbor/harbor_grpo_tinker.py \
    --tinker-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --agent qwen-coder \
    --num-rollouts 50
```

### Rollouts Only (No Training)

```bash
python examples/harbor/harbor_grpo_tinker.py --skip-training --test
```

## Quick Start

**Step 1: Start the Tinker proxy server (Terminal 1)**
```bash
export TINKER_API_KEY="tml-..."
python examples/harbor/tinker_proxy.py \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --port 8000
```

**Step 2: Run training with Harbor agent (Terminal 2)**
```bash
python examples/harbor/harbor_grpo_tinker.py \
    --agent qwen-coder \
    --tinker-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --test
```

## Architecture

```
Harbor Agent (qwen-coder, swe-agent, etc.)
    │ OpenAI API calls
    ▼
Tinker Proxy Server (localhost:8000)
    │ - Translates OpenAI API → Tinker SDK
    │ - Stores logprobs for PPO training
    │ Tinker SDK calls
    ▼
Tinker Cloud (GPU)
    ├── sample_async() → Generate + logprobs
    └── forward_backward(loss_fn="ppo") → Train
```

**Key Benefits:**
- Works with any Harbor agent (qwen-coder, swe-agent, openhands, etc.)
- Logprobs automatically stored for PPO training
- Same model weights for inference and training
- Training triggered via `/v1/train/ppo` endpoint

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tinker-model` | `Qwen/Qwen3-8B` | Tinker model for inference/training |
| `--proxy-url` | `http://localhost:8000/v1` | Tinker proxy server URL |
| `--lora-rank` | `32` | LoRA rank for training |
| `--agent` | `qwen-coder` | Harbor agent (qwen-coder, swe-agent, etc.) |
| `--env` | `docker` | Environment (docker, daytona) |
| `--num-rollouts` | `50` | Number of instances |
| `--n-samples` | `4` | GRPO group size |
| `--lr` | `1e-6` | Learning rate |
| `--output-dir` | `outputs/harbor_grpo_tinker` | Output directory |
| `--test` | - | Test mode (5 instances) |
| `--skip-training` | - | Only run rollouts |

### Proxy Server Options (`tinker_proxy.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3-8B` | Tinker model to load |
| `--port` | `8000` | Port to bind to |
| `--lora-rank` | `32` | LoRA rank for training |
| `--no-training` | - | Inference only (no PPO training) |

## Hyperparameters (Matching SLiME)

Tinker's PPO uses the same algorithm as SLiME's `ppo_utils.py`:

| SLiME Parameter | Value | Tinker Equivalent |
|-----------------|-------|-------------------|
| `eps_clip` | 0.2 | `clip_low_threshold = 0.8` |
| `eps_clip_high` | 0.28 | `clip_high_threshold = 1.28` |
| `lr` | 1e-6 | `learning_rate = 1e-6` |
| `kl_loss_type` | low_var_kl | (tracked in metrics) |

The PPO clipping formula:
```
ratio = exp(new_logprobs - old_logprobs)
clipped_ratio = clamp(ratio, clip_low_threshold, clip_high_threshold)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

## GRPO Training Flow

1. **Sample**: Generate responses with logprobs via Tinker
2. **Evaluate**: Run tests in Docker, get rewards (+1/-1)
3. **Advantages**: Compute group-relative advantages
   ```
   advantages = (rewards - mean) / std
   ```
4. **Train**: PPO update with Tinker's `forward_backward(loss_fn="ppo")`

## Data Sources

### SWE-bench (Default)

```bash
python examples/harbor/harbor_grpo_tinker.py \
    --data-source swebench \
    --num-rollouts 50
```

### C2Bug

```bash
python examples/harbor/harbor_grpo_tinker.py \
    --data-source c2bug \
    --c2bug-dataset TwelfthStar/c2bug_tasks_django_Jan-22-2026 \
    --num-rollouts 50
```

## Example Output

```
============================================================
Harbor GRPO Training - Tinker GPU
============================================================
  Tinker model: Qwen/Qwen3-30B-A3B-Instruct-2507
  Data source: swebench
  Agent: qwen-coder
  n_samples_per_prompt: 4
============================================================

[1/5] django__django-11951
  Sample 1/4: Status: completed, Reward: -1.0
  Sample 2/4: Status: completed, Reward: 1.0
  Sample 3/4: Status: completed, Reward: -1.0
  Sample 4/4: Status: completed, Reward: -1.0
  GRPO advantages: mean_reward=-0.500, std=0.866
  Training on Tinker GPU (PPO loss)...
  Loss: 0.0234
```

## Tinker API Reference

- **Docs**: https://tinker-docs.thinkingmachines.ai/
- **Console**: https://tinker-console.thinkingmachines.ai/
- **Loss Functions**: https://tinker-docs.thinkingmachines.ai/losses
- **PPO Details**: https://tinker-docs.thinkingmachines.ai/losses#ppo

## Troubleshooting

### "TINKER_API_KEY required"

```bash
export TINKER_API_KEY="tml-..."
```

### "Model not supported"

Check available models:
```python
import tinker
client = tinker.ServiceClient()
for m in client.get_server_capabilities().supported_models:
    print(m.model_name)
```

### "input sequence ... must have the same length"

This error occurs when `model_input` length doesn't match `loss_fn_inputs` arrays. For PPO, `model_input` should contain **only response tokens**, not the full prompt+response sequence.

## Files

| File | Description |
|------|-------------|
| `harbor_grpo_tinker.py` | Main training script |
| `harbor_core.py` | Shared GRPO utilities |
| `tinker.md` | This documentation |
