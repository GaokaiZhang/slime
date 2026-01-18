# Harbor GRPO Training

Train coding agents using **Harbor** for rollouts and **GRPO** for RL training.

## Prerequisites

Before running Harbor GRPO training, ensure SLiME is set up:

```bash
cd /path/to/slime
pip install -e .
```

## Quick Start

```bash
# Install dependencies
pip install torch transformers peft accelerate

# Install Harbor CLI
uv tool install harbor

# Run training
python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --agent qwen-coder \
    --num-rollouts 10
```

## Environments

Choose where agent rollouts execute:

| Environment | Flag | Requirements |
|-------------|------|--------------|
| **Local Docker** | `--env docker` (default) | Docker + SWE-bench images |
| **Daytona Cloud** | `--env daytona` | Daytona API key (no Docker needed) |

```bash
# Local Docker (default)
python examples/harbor/harbor_grpo_local.py --env docker ...

# Daytona Cloud
export DAYTONA_API_KEY="your_key"
export DAYTONA_API_URL="https://app.daytona.io/api"
python examples/harbor/harbor_grpo_local.py --env daytona ...
```

## Harbor Agents

All Harbor built-in agents work with both docker and daytona environments:

| Agent | Flag | Description |
|-------|------|-------------|
| `qwen-coder` | `--agent qwen-coder` | Qwen Code agent (default) |
| `mini-swe-agent` | `--agent mini-swe-agent` | Mini SWE Agent |
| `claude-code` | `--agent claude-code` | Claude Code CLI |
| `openhands` | `--agent openhands` | OpenHands |
| `aider` | `--agent aider` | Aider |
| `swe-agent` | `--agent swe-agent` | SWE-Agent |
| `oracle` | `--agent oracle` | Oracle (applies gold solution) |

See all agents: [Harbor docs](https://github.com/laude-institute/harbor)

### About mini-swe-agent

Harbor's `mini-swe-agent` uses the [mini-swe-agent-plus](https://github.com/mini-swe-agent-plus) tool internally. To use it:

```bash
# Install mini-swe-agent-plus (required for --agent mini-swe-agent)
pip install minisweagent
# Or install from slime submodule:
pip install -e submodules/mini-swe-agent-plus

# Then use with Harbor
python examples/harbor/harbor_grpo_local.py \
    --agent mini-swe-agent \
    --model openai/gpt-4o \
    --num-rollouts 10
```

## Examples

```bash
# Any model + any agent + local Docker
python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --agent qwen-coder \
    --env docker \
    --num-rollouts 50

# Daytona cloud (HPC clusters without Docker)
python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --agent mini-swe-agent \
    --env daytona \
    --num-rollouts 50

# Custom instance list
python examples/harbor/harbor_grpo_local.py \
    --instances train_instances_id.txt \
    --model your-model-name \
    --agent openhands
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-Coder-7B-Instruct | HuggingFace model name |
| `--agent` | qwen-coder | Harbor agent (see table above) |
| `--agent-import-path` | None | Custom agent import path |
| `--env` | docker | Environment: `docker` or `daytona` |
| `--dataset` | swebench-verified@1.0 | Harbor dataset |
| `--num-rollouts` | 50 | Number of instances |
| `--instances` | None | Path to instance ID file |
| `--n-samples` | 4 | GRPO group size |
| `--lr` | 1e-6 | Learning rate |
| `--kl-coef` | 0.001 | KL coefficient |
| `--output-dir` | outputs/harbor_grpo | Output directory |
| `--test` | - | Test mode (5 instances) |
| `--no-lora` | - | Disable LoRA |

## Training Data

Django training instances (201): `train_instances_id.txt` in repo root.

## GRPO Hyperparameters (Search-R1)

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| KL coefficient | 0.001 |
| KL loss type | low_var_kl |
| PPO clip | 0.2 |
| PPO clip high | 0.28 |
| Group size | 4 |

## File Structure

```
examples/harbor/
├── harbor_grpo_local.py   # Main trainer (local GPU)
├── harbor_grpo_modal.py   # Modal GPU trainer
├── harbor_core.py         # Shared GRPO implementation
└── __init__.py
```

## Modal GPU Training

For users without local GPU.

```bash
pip install modal
modal setup

modal run examples/harbor/harbor_grpo_modal.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --num-rollouts 50
```
