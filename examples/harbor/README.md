# Harbor GRPO Training

Train coding agents with GRPO using Harbor for rollouts. Supports both SWE-bench and C2Bug data sources.

## Quick Start

```bash
# Local GPU + Docker (default)
python examples/harbor/harbor_grpo_local.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --agent qwen-coder \
    --num-rollouts 10

# Modal GPU + Docker
modal run examples/harbor/harbor_grpo_modal.py \
    --num-rollouts 10

# C2Bug data source
python examples/harbor/harbor_grpo_local.py \
    --data-source c2bug \
    --num-rollouts 10
```

## Data Sources

### SWE-bench (default)
```bash
python examples/harbor/harbor_grpo_local.py \
    --data-source swebench \
    --dataset swebench-verified@1.0
```

### C2Bug (HuggingFace)
```bash
python examples/harbor/harbor_grpo_local.py \
    --data-source c2bug \
    --c2bug-dataset TwelfthStar/c2bug_tasks_django_Jan-22-2026
```

## Environment Options

| Environment | Flag | Internet | Requirements |
|-------------|------|----------|--------------|
| **Docker** (default) | `--env docker` | Full | Local Docker |
| **Modal** | `--env modal` | Full | Modal account |
| **Daytona** | `--env daytona` | Limited* | Daytona API key |

*Daytona free tier has network restrictions. Use `--env docker` or `--env modal` for LLM-based agents.

### Docker Setup (Recommended)
```bash
# Pull SWE-bench images
docker pull swebench/sweb.eval.x86_64.django_1776_django-13810:latest
```

### Modal Setup
```bash
pip install modal
modal setup
```

### Daytona Setup (Optional)
```bash
export DAYTONA_API_KEY="dtn_..."
export DAYTONA_API_URL="https://app.daytona.io/api"
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-Coder-7B-Instruct | Training model |
| `--agent` | qwen-coder | Harbor agent |
| `--env` | docker | Environment: docker, modal, daytona |
| `--data-source` | swebench | Data: swebench or c2bug |
| `--num-rollouts` | 50 | Number of instances |
| `--n-samples` | 4 | GRPO group size |
| `--lr` | 1e-6 | Learning rate |
| `--kl-coef` | 0.001 | KL coefficient |
| `--test` | - | Test mode (5 instances) |
| `--skip-training` | - | Only run rollouts |

## Harbor Agents

| Agent | API Required | Description |
|-------|--------------|-------------|
| `oracle` | None | Applies gold solution (testing) |
| `qwen-coder` | OpenAI/Qwen | Qwen Code CLI |
| `claude-code` | Anthropic | Claude Code CLI |
| `aider` | Various | Aider |
| `openhands` | Various | OpenHands |

## File Structure

```
examples/harbor/
├── harbor_grpo_local.py     # Local GPU trainer
├── harbor_grpo_modal.py     # Modal GPU trainer
├── harbor_core.py           # Shared GRPO implementation
├── c2bug_adapter.py         # C2Bug data adapter
├── c2bug_template/          # Harbor task templates for C2Bug
│   ├── Dockerfile
│   ├── instruction.md
│   ├── task.toml
│   └── test.sh
├── simple_openai_server.py  # Local model server (optional)
├── modal_openai_server.py   # Modal model server (optional)
└── README.md
```

## Verified Configurations

| Config | Status | Notes |
|--------|--------|-------|
| Local GPU + Docker + oracle | Tested | 100% success |
| Modal GPU + Docker + claude-code | Tested | 100% success |
| Modal GPU + Modal env + claude-code | Tested | 100% success |
| C2Bug + Docker + oracle | Tested | 100% success |

## GRPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| KL coefficient | 0.001 |
| KL loss type | low_var_kl |
| PPO clip | 0.2 / 0.28 |
| Group size | 4 |
