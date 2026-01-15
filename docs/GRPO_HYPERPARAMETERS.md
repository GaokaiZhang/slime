# GRPO Hyperparameters: Our Implementation vs Search-R1 vs SLiME

This document provides a detailed comparison of our GRPO implementation against Search-R1 and SLiME.

## 1. Core Hyperparameters

| Parameter | Our Implementation | Search-R1 | SLiME Default | Notes |
|-----------|-------------------|-----------|---------------|-------|
| **Learning Rate** | `1e-6` | `1e-6` | `1e-6` | ✅ Match |
| **KL Coefficient** | `0.001` | `0.001` | `0.0` | ✅ Match Search-R1 |
| **KL Loss Type** | `low_var_kl` | `low_var_kl` | `k1` | ✅ Match Search-R1 |
| **Samples per Prompt** | `4-5` | `5` | `1` | ✅ Match Search-R1 |
| **Temperature** | `1.0` | `1.0` | (varies) | ✅ Match |
| **Gamma** | `1.0` | `1.0` | `1.0` | ✅ Match |
| **Lambda (GAE)** | `1.0` | N/A | `1.0` | ✅ Standard GRPO |
| **eps_clip (lower)** | `0.2` | `0.2` | `0.2` | ✅ Match |
| **eps_clip_high (upper)** | `0.28` | `0.28` | `= eps_clip` | ✅ Match Search-R1 (DAPO-style) |

## 2. KL Divergence Implementation

### Our Implementation (`grpo_trainer.py`, `modal_grpo_trainer.py`)
```python
# Low-variance KL from Search-R1
ratio = torch.exp(log_probs_policy - log_probs_ref)
kl = 0.5 * (ratio - 1) ** 2
```

### SLiME Implementation (`slime/utils/ppo_utils.py:30-51`)
```python
if kl_loss_type in ["k3", "low_var_kl"]:
    # http://joschu.net/blog/kl-approx.html
    log_ratio = -log_ratio  # Note: SLiME uses negative
    kl = log_ratio.exp() - 1 - log_ratio  # Different formula!
```

### Analysis
| Aspect | Our Implementation | SLiME `low_var_kl` |
|--------|-------------------|-------------------|
| **Formula** | `0.5 * (ratio - 1)^2` | `exp(-r) - 1 + r` where `r = log_π - log_π_ref` |
| **Taylor Expansion** | ≈ `0.5 * r^2` | ≈ `0.5 * r^2` (same at 2nd order) |
| **Range** | Always ≥ 0 | Always ≥ 0 |
| **Source** | Search-R1 paper | Schulman KL blog |

**Both are valid low-variance KL approximations** - they're equivalent to 2nd order Taylor expansion.

## 3. Policy Loss (PPO Clipping)

### Our Implementation
```python
# Importance sampling ratio
ratio = torch.exp(log_probs_policy - log_probs_old)

# DAPO-style asymmetric clipping
clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)
# eps_clip = 0.2, eps_clip_high = 0.28

# Policy loss
loss1 = -advantages * ratio
loss2 = -advantages * clipped_ratio
policy_loss = torch.max(loss1, loss2).mean()
```

### SLiME Implementation (`slime/utils/ppo_utils.py:125-148`)
```python
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()  # ppo_kl = log_π_old - log_π
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # Optional: Dual-clip PPO (eps_clip_c)
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1
```

### Analysis
| Aspect | Our Implementation | SLiME |
|--------|-------------------|-------|
| **Asymmetric Clipping** | ✅ `[1-0.2, 1+0.28]` | ✅ Supported via `eps_clip_high` |
| **Dual-clip PPO** | ❌ Not implemented | ✅ Optional via `eps_clip_c` |
| **Ratio Computation** | `exp(log_π - log_π_old)` | `exp(-(log_π_old - log_π))` = same |

## 4. Advantage Computation

### Our Implementation (GRPO)
```python
def compute_grpo_advantages(rewards, normalize=True):
    mean_reward = sum(rewards) / len(rewards)
    std_reward = sqrt(sum((r - mean) ** 2) / len(rewards))
    std_reward = max(std_reward, 1e-8)
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    return advantages
```

### SLiME Implementation (`slime/utils/ppo_utils.py:200-263`)

**Standard GRPO** (`get_grpo_returns`):
```python
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
# Note: Normalization done separately
```

**SSR-GRPO** (`get_ssr_grpo_returns`):
```python
def get_ssr_grpo_returns(rewards, kl, response_lengths, use_weighted_mean=False):
    """
    SSR Differences from GRPO:
    1. Uses returns Ri (sum of rewards) instead of terminal reward ri
    2. No σ normalization: Aˆi = (Ri − µ) instead of (ri − µ)/σ
    3. Optional: Weighted mean return (length-weighted)
    """
    mu = rewards.mean().item()
    centered_rewards = rewards - mu  # No std normalization!

    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * centered_rewards[i])
    return returns, centered_rewards
```

### Analysis
| Aspect | Our Implementation | SLiME GRPO | SLiME SSR-GRPO |
|--------|-------------------|------------|----------------|
| **Mean Centering** | ✅ `r - μ` | ✅ | ✅ |
| **Std Normalization** | ✅ `/ σ` | ✅ (via args) | ❌ (SSR-specific) |
| **Length Weighting** | ❌ | ❌ | ✅ Optional |
| **Gibberish Detection** | ❌ | ❌ | ✅ Available |

## 5. SLiME Advantage Estimators

SLiME supports multiple advantage estimators (`--advantage-estimator`):

| Estimator | Description | Our Use |
|-----------|-------------|---------|
| `grpo` | Standard GRPO with σ normalization | ✅ Used |
| `ssr_grpo` | SSR variant without σ normalization | ❌ Not used |
| `gspo` | Generalized Sequence Policy Optimization | ❌ Not used |
| `reinforce_plus_plus` | REINFORCE++ with discounted returns | ❌ Not used |
| `reinforce_plus_plus_baseline` | REINFORCE++ with baseline | ❌ Not used |
| `ppo` | Standard PPO with GAE | ❌ Not used |
| `on_policy_distillation` | Distillation-based | ❌ Not used |

## 6. Optimizer Settings

| Parameter | Our Implementation | Search-R1 | SLiME |
|-----------|-------------------|-----------|-------|
| **Optimizer** | AdamW | AdamW | AdamW |
| **β1** | `0.9` | `0.9` | Megatron default |
| **β2** | `0.98` | `0.98` | Megatron default |
| **Weight Decay** | `0.1` | `0.1` | `0.1` |
| **Grad Clip** | `1.0` | `1.0` | `1.0` |

## 7. SLiME-Specific Features We Don't Use

### Off-Policy Correction (TIS)
```python
# SLiME supports Truncated Importance Sampling
--use-tis                 # Enable TIS
--tis-clip 2.0           # Upper clip for IS ratio
--tis-clip-low 0.1       # Lower clip for IS ratio
```

### Off-Policy Sequence Masking (OPSM)
```python
# SLiME supports OPSM for handling off-policy data
--use-opsm               # Enable OPSM
--opsm-delta 0.1        # KL threshold for masking
```

### Unbiased KL Estimation
```python
# SLiME supports IS-weighted KL (DeepSeek-V3.2 style)
--use-unbiased-kl       # Enable unbiased KL
```

### Dr.GRPO σ Normalization
```python
# SLiME supports Dr.GRPO variant
--disable-grpo-std-normalization  # Disable σ normalization
```

## 8. Data Format Comparison

### Our Sample Format
```python
@dataclass
class GRPOSample:
    prompt_tokens: list[int]      # Encoded prompt
    response_tokens: list[int]    # Generated response tokens
    tokens: list[int]             # prompt + response (full sequence)
    logprobs: list[float]         # π_old(token) from generation
    reward: float                 # Scalar reward
```

### SLiME Sample Format (`slime/utils/types.py`)
```python
@dataclass
class Sample:
    prompt: str | list[dict]      # Raw prompt or chat messages
    tokens: list[int]             # Full token sequence
    response_length: int          # Length of response
    loss_mask: list[int]          # Which tokens to train on
    rollout_log_probs: list[float]# π_old from rollout engine
    reward: float | dict          # Scalar or multi-reward
```

## 9. Reward Functions

**We use swebench.harness ONLY - no heuristics.**

| Script | Reward Function | Docker Required |
|--------|----------------|-----------------|
| `local_gpu_grpo_trainer.py` | swebench.harness | ✅ Yes |
| `local_grpo_trainer.py` | swebench.harness | ✅ Yes |
| `modal_grpo_trainer.py` | ⚠️ Heuristic (deprecated) | ❌ |

**Note**: Modal cannot easily run Docker containers, so Modal-based training uses heuristic rewards.
For accurate training, use `local_gpu_grpo_trainer.py` with local GPU + Docker.

## 10. Summary: Our Implementation Status

### ✅ Correctly Implemented
- Learning rate: `1e-6`
- KL coefficient: `0.001`
- KL loss type: Low-variance (equivalent to SLiME's)
- Samples per prompt: `4-5`
- Temperature: `1.0`
- Asymmetric PPO clipping: `[0.8, 1.28]`
- Group-relative advantage with σ normalization
- AdamW optimizer with correct β values
- Gradient clipping at `1.0`

### ⚠️ Minor Differences
- KL formula: Uses `0.5*(r-1)^2` vs SLiME's `exp(-r)-1+r` (mathematically equivalent)
- No Dual-clip PPO support
- No TIS/OPSM off-policy correction

### ❌ Not Implemented (Advanced SLiME Features)
- `ssr_grpo` advantage estimator (no σ normalization)
- Length-weighted mean for SSR
- Gibberish detection
- Off-policy correction (TIS, OPSM)
- Unbiased KL estimation
- Dr.GRPO variant

## 11. Commands Comparison

### Our Implementation (RECOMMENDED: Local GPU + swebench.harness)
```bash
# Prerequisites: Local GPU (24GB+ VRAM), Docker with SWE-bench images

# Quick test
bash examples/harbor/run_local_gpu_grpo.sh --test

# Full training with swebench.harness evaluation
python examples/harbor/local_gpu_grpo_trainer.py \
    --num-rollouts 100 \
    --n-samples 4 \
    --lr 1e-6 \
    --kl-coef 0.001 \
    --temperature 1.0 \
    --use-hf-generate  # Or --vllm-url http://localhost:8000

# With local vLLM for faster inference (requires 2 GPUs or careful memory management)
# Terminal 1: Start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Kwai-Klear/Klear-AgentForge-8B-SFT \
    --port 8000 --dtype bfloat16

# Terminal 2: Run training
python examples/harbor/local_gpu_grpo_trainer.py \
    --num-rollouts 100 \
    --n-samples 4 \
    --vllm-url http://localhost:8000
```

### SLiME
```bash
python -m slime.train \
    --hf_checkpoint "model_name" \
    --rollout_module "examples.harbor.rollout" \
    --rollout_fn "generate" \
    --n_samples_per_prompt 5 \
    --rollout_temperature 1.0 \
    --lr 1e-6 \
    --kl_loss_coef 0.001 \
    --kl_loss_type "low_var_kl" \
    --advantage_estimator "grpo" \
    --eps_clip 0.2 \
    --eps_clip_high 0.28 \
    --gamma 1.0 \
    --use_kl_loss
```

## 12. References

- **Search-R1**: https://arxiv.org/abs/2503.09516
- **GRPO**: Group Relative Policy Optimization
- **DAPO**: https://arxiv.org/abs/2503.14476 (asymmetric clipping)
- **Dr.GRPO**: https://arxiv.org/abs/2503.20783 (no σ normalization)
- **Schulman KL Blog**: http://joschu.net/blog/kl-approx.html
- **PPO**: https://arxiv.org/abs/1707.06347
