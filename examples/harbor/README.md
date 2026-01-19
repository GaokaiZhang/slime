# Harbor GRPO Training

Train coding agents with GRPO using Harbor for rollouts and Daytona for cloud sandboxes.

## Quick Start

```bash
# Set Daytona credentials
export DAYTONA_API_KEY="dtn_..."
export DAYTONA_API_URL="https://app.daytona.io/api"

# Run GRPO training
python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --agent qwen-coder \
    --env daytona \
    --num-rollouts 50 \
    --n-samples 4
```

## Running Commands

### Local Docker
```bash
python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --agent qwen-coder \
    --env docker \
    --num-rollouts 50
```

### Daytona Cloud (no Docker required)
```bash
export DAYTONA_API_KEY="your_key"
export DAYTONA_API_URL="https://app.daytona.io/api"

python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --agent qwen-coder \
    --env daytona \
    --num-rollouts 50
```

### Modal GPU (no local GPU required)
```bash
modal run examples/harbor/harbor_grpo_modal.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --num-rollouts 50
```

### Test Mode (5 instances)
```bash
python examples/harbor/harbor_grpo_local.py --test
```

---

## Setup

### Prerequisites
```bash
pip install torch transformers peft accelerate datasets
pip install -e .  # Install SLiME
uv tool install harbor  # Install Harbor CLI
```

### Daytona Setup

1. Get API key from [Daytona](https://app.daytona.io)
2. Set environment variables:
```bash
export DAYTONA_API_KEY="dtn_..."
export DAYTONA_API_URL="https://app.daytona.io/api"
```

### Docker Setup (alternative to Daytona)

Pull SWE-bench images:
```bash
docker pull swebench/django__django:latest
```

---

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-Coder-7B-Instruct | HuggingFace model |
| `--agent` | qwen-coder | Harbor agent |
| `--agent-model` | None | Model for agent inference |
| `--env` | docker | `docker` or `daytona` |
| `--dataset` | swebench-verified@1.0 | Dataset |
| `--num-rollouts` | 50 | Number of instances |
| `--instances` | None | Instance ID file path |
| `--n-samples` | 4 | GRPO group size |
| `--lr` | 1e-6 | Learning rate |
| `--kl-coef` | 0.001 | KL coefficient |
| `--output-dir` | outputs/harbor_grpo | Output path |
| `--save-every` | 10 | Checkpoint frequency |
| `--test` | - | Test mode (5 instances) |
| `--no-lora` | - | Full fine-tuning |

### Harbor Agents

| Agent | Description |
|-------|-------------|
| `qwen-coder` | Qwen Code CLI (default) |
| `mini-swe-agent` | Mini SWE Agent |
| `claude-code` | Claude Code CLI |
| `openhands` | OpenHands |
| `aider` | Aider |
| `swe-agent` | SWE-Agent |
| `oracle` | Applies gold solution |

### GRPO Hyperparameters (Search-R1)

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| KL coefficient | 0.001 |
| KL loss type | low_var_kl |
| PPO clip | 0.2 / 0.28 |
| Group size | 4 |

---

## File Structure

```
examples/harbor/
├── harbor_grpo_local.py   # Local GPU trainer
├── harbor_grpo_modal.py   # Modal GPU trainer
├── harbor_core.py         # Shared GRPO implementation
└── README.md
```

## Training Data

Django instances: `train_instances_id.txt` (201 instances)
