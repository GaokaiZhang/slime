# Harbor GRPO Training

Train coding agents with GRPO using Harbor for rollouts. Supports both SWE-bench and C2Bug data sources.

## Prerequisites

### 1. Install Dependencies
```bash
# Install SLiME
git clone https://github.com/your-repo/slime.git
cd slime
pip install -e .

# Install Harbor via uv
pip install uv
uv tool install harbor
```

### 2. Fix Harbor's qwen-coder Agent (REQUIRED)

Harbor's qwen-coder agent has a bug that prevents it from interacting with the environment. You must manually patch it:

```bash
# Locate Harbor's qwen-coder agent
HARBOR_AGENT_FILE=~/.local/share/uv/tools/harbor/lib/python3.*/site-packages/harbor/agents/installed/qwen_code.py

# Edit the file and find the line (around line 62-64):
#   f"echo {escaped_instruction} | qwen -y "
# Replace it with:
#   f"qwen --approval-mode yolo -p {escaped_instruction} "
```

**The exact change:**
```python
# OLD (broken - qwen only generates text, doesn't modify code):
return [
    ExecInput(
        command=(
            f"echo {escaped_instruction} | qwen -y "
            f"2>&1 | tee /logs/agent/qwen-code.txt"
        ),
        env=env,
    )
]

# NEW (fixed - qwen actually modifies files):
return [
    ExecInput(
        command=(
            f"qwen --approval-mode yolo -p {escaped_instruction} "
            f"2>&1 | tee /logs/agent/qwen-code.txt"
        ),
        env=env,
    )
]
```

**Why this is needed:** The original command uses an invalid `-y` flag and pipes input via `echo`, which prevents qwen from using tools and modifying code. The fixed version uses the correct `--approval-mode yolo` flag and `-p` (prompt) flag, allowing qwen to actually interact with the environment.

### 3. Setup Docker (for SWE-bench)
```bash
# Ensure Docker is installed and running
docker --version

# Add user to docker group (if needed)
sudo usermod -aG docker $USER
newgrp docker
```

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

### Local Model Server (Optional)

For local GPU inference, start vLLM server:
```bash
# Example: Qwen3-Coder-30B-A3B on 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

# Then pass the URL to Harbor
python examples/harbor/harbor_grpo_local.py \
    --openai-base-url http://172.17.0.1:8000/v1 \
    --openai-api-key local \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct
```

**Note:** Use `172.17.0.1` (Docker bridge IP) instead of `localhost` so agents inside containers can reach the host server.

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
