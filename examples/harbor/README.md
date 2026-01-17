# Harbor + SLiME Integration

This module integrates **Harbor** (agent evaluation framework) with **SLiME** (RL training framework) for training coding agents using GRPO.

## Key Insight

**Harbor handles trajectory generation, SLiME handles training.**

Log probabilities are NOT collected during Harbor rollouts - they are recomputed at training time via forward pass. This enables using any Harbor agent (mini-swe-agent-plus, qwen-code, etc.) for RL training.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Harbor CLI     │───▶│  Trajectories    │───▶│  SLiME GRPO     │
│ (mini-swe-agent)│    │  (text + reward) │    │  Training       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
       │                        │
       ▼                        ▼
  Local Docker            swebench.harness
  (execution)             (evaluation)
```

## File Structure

```
examples/harbor/
├── harbor_core.py          # ★ SHARED: Config, Harbor rollouts, GRPO training
├── harbor_grpo_local.py    # Local GPU trainer
├── harbor_grpo_modal.py    # Modal GPU trainer
├── harbor_rollout.py       # Harbor CLI wrapper
├── trajectory_converter.py # Harbor output → SLiME Sample
├── harbor_slime_trainer.py # Basic trainer
└── test_harbor_slime.py    # Integration tests
```

## Quick Start

### Local GPU Training

```bash
# Test mode (5 instances)
python examples/harbor/harbor_grpo_local.py --test

# Full training (201 Django instances)
python examples/harbor/harbor_grpo_local.py --num-rollouts 201 --n-samples 4
```

### Modal GPU Training

```bash
# Test mode
modal run examples/harbor/harbor_grpo_modal.py --test

# Full training
modal run examples/harbor/harbor_grpo_modal.py --num-rollouts 201 --n-samples 4
```

### Custom Agent

```bash
python examples/harbor/harbor_grpo_local.py --agent qwen-coder --num-rollouts 50
```

## Search-R1 GRPO Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 1e-6 | Learning rate |
| `kl_coef` | 0.001 | KL divergence coefficient |
| `kl_loss_type` | low_var_kl | Low-variance KL approximation |
| `eps_clip` | 0.2 | PPO clip lower bound |
| `eps_clip_high` | 0.28 | PPO clip upper bound (DAPO) |

## Installation

### 1. Install Harbor via uv tool (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install harbor as a CLI tool
uv tool install harbor

# Verify installation
harbor --help
```

### 2. Install SLiME dependencies

```bash
# In the slime conda environment
pip install transformers peft torch
```

## Quick Start

### Option 1: Use the integrated trainer

```bash
python examples/harbor/harbor_slime_trainer.py \
    --model Qwen/Qwen3-Coder-30B-A3B \
    --agent qwen-coder \
    --dataset swebench-verified@1.0 \
    --n-rollouts 50 \
    --n-samples 2 \
    --output-dir outputs/harbor_training
```

### Option 2: Generate trajectories, then train separately

```bash
# Step 1: Generate trajectories with Harbor
harbor run \
    --agent qwen-coder \
    --model Qwen/Qwen3-Coder-30B-A3B \
    --dataset swebench-verified@1.0 \
    --n-concurrent 4 \
    --jobs-dir jobs \
    --export-traces

# Step 2: Convert to SLiME format
python examples/harbor/trajectory_converter.py \
    jobs/your-job-name \
    --tokenizer Qwen/Qwen3-Coder-30B-A3B \
    --output samples.json

# Step 3: Train with SLiME (using converted samples)
```

## Architecture

### Why No Log Probs from Harbor?

Traditional RL training requires log probabilities at inference time. However:

1. **Harbor agents are CLI wrappers** - They invoke external tools (qwen CLI, openhands, etc.) that don't expose log probs
2. **Log probs can be imprecise** - Inference-time log probs may differ from training due to optimizations
3. **On-policy training recomputes anyway** - GRPO does a forward pass at training time

**Solution**: Only collect (tokens, response, reward) from Harbor, recompute log probs at training time.

### Data Flow

```
Harbor Rollout:
    Input: Task instruction
    Output: Sample {
        tokens: [token IDs],
        response: "generated text",
        reward: float (from verifier),
        // NO log_probs!
    }

SLiME Training:
    1. Forward pass on policy model → policy_log_probs
    2. Forward pass on ref model → ref_log_probs
    3. Compute GRPO loss
    4. Backward + optimizer step
```

## Components

### `trajectory_converter.py`

Converts Harbor job output to SLiME Sample format.

```python
from trajectory_converter import HarborTrajectoryConverter

converter = HarborTrajectoryConverter(tokenizer)
samples = converter.load_job("jobs/my-job")
# or
samples = converter.load_traces_parquet("jobs/my-job/traces.parquet")
```

### `harbor_rollout.py`

Generates rollouts using Harbor CLI.

```python
from harbor_rollout import HarborRolloutGenerator, HarborConfig

config = HarborConfig(
    model="Qwen/Qwen3-Coder-30B-A3B",
    agent="qwen-coder",
    n_concurrent=4,
)

generator = HarborRolloutGenerator(config=config)
samples = generator.generate(dataset="swebench-verified@1.0", n_tasks=10)
```

### `harbor_slime_trainer.py`

Full training pipeline integrating Harbor + SLiME.

```python
from harbor_slime_trainer import HarborSlimeTrainer, TrainerConfig

config = TrainerConfig(
    model_name="Qwen/Qwen3-Coder-30B-A3B",
    agent="qwen-coder",
    dataset="swebench-verified@1.0",
    n_rollouts=50,
    n_samples_per_prompt=2,
    use_lora=True,
)

trainer = HarborSlimeTrainer(config)
trainer.train()
```

## Configuration

### Model Options

| Model | Description |
|-------|-------------|
| `Qwen/Qwen3-Coder-30B-A3B` | **Default**. Qwen3 Coder 30B (A3B variant) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | Qwen2.5 Coder 32B |
| `Kwai-Klear/Klear-AgentForge-8B-SFT` | Smaller 8B model for testing |

### Agent Options

| Agent | Description |
|-------|-------------|
| `qwen-coder` | **Default**. Qwen Code CLI agent |
| `openhands` | OpenHands agent |
| `aider` | Aider coding assistant |
| `mini-swe-agent` | Mini SWE Agent |
| `claude-code` | Claude Code agent |

### Dataset Options

| Dataset | Description |
|---------|-------------|
| `swebench-verified@1.0` | **Default**. SWE-bench Verified |
| `swebench@1.0` | Full SWE-bench |
| `terminal-bench@2.0` | Terminal Bench |

## Training Parameters

```python
TrainerConfig(
    # GRPO hyperparameters
    n_samples_per_prompt=2,      # Group size for GRPO
    lr=1e-6,                     # Learning rate
    kl_coef=0.001,               # KL penalty coefficient
    eps_clip=0.2,                # PPO clip lower bound
    eps_clip_high=0.28,          # PPO clip upper bound (DAPO style)

    # LoRA
    use_lora=True,               # Use LoRA for efficient training
    lora_r=16,                   # LoRA rank

    # Output
    save_every=10,               # Checkpoint frequency
)
```

## Tips

### Memory Optimization

For large models, use LoRA (enabled by default):

```bash
python harbor_slime_trainer.py --model Qwen/Qwen3-Coder-30B-A3B --lora-r 16
```

### Debugging

Test with a smaller model first:

```bash
python harbor_slime_trainer.py \
    --model Kwai-Klear/Klear-AgentForge-8B-SFT \
    --n-rollouts 5 \
    --n-concurrent 2
```

### Resume from Harbor Job

If Harbor job already completed:

```python
generator = HarborRolloutGenerator(config=config)
samples = generator.generate_from_existing_job("jobs/existing-job")
```

## Comparison with Direct GRPO

| Approach | Log Probs | Agent Flexibility | Complexity |
|----------|-----------|-------------------|------------|
| **Harbor + SLiME** | Recomputed | Any Harbor agent | Medium |
| Direct GRPO | At inference | Custom only | High |
| SFT on filtered | Not needed | Any | Low |

Choose Harbor + SLiME when:
- You want to use existing Harbor agents (qwen-code, openhands)
- You need evaluation infrastructure (Docker environments, verification)
- You prefer modular rollout/training separation

## License

Same as SLiME repository.
