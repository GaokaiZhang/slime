# Harbor GRPO Training

Train coding agents using **Harbor** for rollouts and **SLiME GRPO** for RL training.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Harbor Agent   │───▶│  Trajectories    │───▶│  GRPO Training  │
│ (mini-swe-agent)│    │  (text + reward) │    │  (GPU)          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
       │                        │
       ▼                        ▼
  Local Docker            swebench.harness
  (agent execution)       (evaluation)
```

**Key**: Log probs are recomputed at training time, not captured from Harbor.

---

## Prerequisites (All Users)

### Step 1: Clone SLiME Repository

```bash
git clone https://github.com/GaokaiZhang/slime.git
cd slime
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
conda create -n slime python=3.11 -y
conda activate slime
```

### Step 3: Install SLiME and Dependencies

```bash
# Install SLiME (required for ppo_utils)
pip install -e .

# Install training dependencies
pip install torch transformers peft accelerate datasets huggingface_hub

# Install Harbor CLI
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install harbor

# Verify installations
python -c "from slime.utils.ppo_utils import compute_policy_loss; print('SLiME OK')"
harbor --help
```

### Step 4: Setup Docker with SWE-bench Images

```bash
# Verify Docker is running
docker ps

# Pull SWE-bench images (example for Django)
docker pull ghcr.io/swe-bench/django__django:latest

# Or pull all images (takes time and disk space)
# See: https://github.com/princeton-nlp/SWE-bench
```

---

## Option 1: Local GPU Training

**Additional Requirements:**
- Local GPU (24GB+ VRAM recommended)

### Run Training

```bash
cd slime

# Test mode (5 instances, quick validation)
python examples/harbor/harbor_grpo_local.py --test

# Full training (201 Django instances)
python examples/harbor/harbor_grpo_local.py \
    --num-rollouts 201 \
    --n-samples 4 \
    --output-dir outputs/harbor_grpo_local
```

### CLI Options

```bash
python examples/harbor/harbor_grpo_local.py --help

Options:
  --num-rollouts    Number of SWE-bench instances (default: 50)
  --n-samples       GRPO group size (default: 4)
  --agent           Harbor agent: mini-swe-agent-plus, qwen-coder (default: mini-swe-agent-plus)
  --model           Model name (default: Kwai-Klear/Klear-AgentForge-8B-SFT)
  --lr              Learning rate (default: 1e-6)
  --kl-coef         KL coefficient (default: 0.001)
  --output-dir      Output directory (default: outputs/harbor_grpo_local)
  --test            Test mode with 5 instances
  --no-lora         Disable LoRA (use full fine-tuning)
```

---

## Option 2: Modal GPU Training

**Additional Requirements:**
- Modal account

### Setup Modal (One-time)

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Create HF token secret
modal secret create hf-token-swe HF_TOKEN=your_huggingface_token

# Verify
modal profile list
```

### Run Training

```bash
cd slime

# Test mode (5 instances)
modal run examples/harbor/harbor_grpo_modal.py --test

# Full training (201 Django instances)
modal run examples/harbor/harbor_grpo_modal.py \
    --num-rollouts 201 \
    --n-samples 4 \
    --output-dir outputs/harbor_grpo_modal
```

### CLI Options

```bash
modal run examples/harbor/harbor_grpo_modal.py --help

Options:
  --num-rollouts    Number of SWE-bench instances (default: 50)
  --n-samples       GRPO group size (default: 4)
  --agent           Harbor agent (default: mini-swe-agent-plus)
  --model-name      Model name (default: Kwai-Klear/Klear-AgentForge-8B-SFT)
  --lr              Learning rate (default: 1e-6)
  --kl-coef         KL coefficient (default: 0.001)
  --output-dir      Output directory (default: outputs/harbor_grpo_modal)
  --test            Test mode with 5 instances
```

---

## GRPO Parameters (Search-R1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 1e-6 | Learning rate |
| `kl_coef` | 0.001 | KL divergence coefficient |
| `kl_loss_type` | low_var_kl | Low-variance KL approximation |
| `eps_clip` | 0.2 | PPO clip lower bound |
| `eps_clip_high` | 0.28 | PPO clip upper bound (DAPO) |
| `n_samples` | 4 | Group size for GRPO |

---

## File Structure

```
examples/harbor/
├── harbor_core.py          # ★ Shared: Config, rollouts, GRPO training
├── harbor_grpo_local.py    # Local GPU trainer
├── harbor_grpo_modal.py    # Modal GPU trainer
├── harbor_rollout.py       # Harbor CLI wrapper
├── trajectory_converter.py # Harbor output → SLiME format
└── test_harbor_slime.py    # Integration tests
```

---

## Troubleshooting

### Harbor agent fails

```bash
# Check Harbor is installed
harbor --help

# Check Docker is running
docker ps

# Test Harbor manually
harbor run --agent oracle --dataset hello-world@1.0 --n-concurrent 1
```

### Modal authentication error

```bash
# Re-authenticate
modal setup

# Check secret exists
modal secret list | grep hf-token
```

### CUDA out of memory

```bash
# Use LoRA (enabled by default)
python examples/harbor/harbor_grpo_local.py --test

# Or reduce batch size by using fewer samples
python examples/harbor/harbor_grpo_local.py --n-samples 2
```

### swebench.harness evaluation fails

```bash
# Check Docker images
docker images | grep swe-bench

# Pull missing image
docker pull ghcr.io/swe-bench/django__django:latest
```

---

## Output

Training outputs are saved to `--output-dir`:

```
outputs/harbor_grpo_local/
├── metrics.json          # Per-instance training metrics
├── summary.json          # Training summary
├── checkpoint_10/        # Intermediate checkpoint
├── checkpoint_20/
└── final/                # Final trained model
```

---

## Comparison: Local vs Modal

| Feature | Local GPU | Modal GPU |
|---------|-----------|-----------|
| GPU | Your machine | Modal A100-80GB |
| Rollouts | Local (Harbor + Docker) | Local (Harbor + Docker) |
| Training | Local GPU | Modal GPU |
| Cost | Your electricity | Modal credits |
| Setup | Simpler | Requires Modal account |

**Recommendation:**
- Use **Local GPU** if you have 24GB+ VRAM
- Use **Modal GPU** if you need A100 or don't have local GPU
