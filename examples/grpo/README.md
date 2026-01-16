# GRPO Training for SWE-bench

Multi-turn agent GRPO training using **mini-swe-agent-plus** with swebench.harness evaluation.

**Agent Framework:** [mini-swe-agent-plus](https://github.com/GaokaiZhang/mini-swe-agent-plus) (modal branch)

## Quick Start

```bash
# 1. Deploy Modal vLLM (inference server)
modal deploy examples/grpo/modal_vllm.py

# 2. Run GRPO training
python examples/grpo/mini_swe_grpo_trainer.py \
    --model-name Kwai-Klear/Klear-AgentForge-8B-SFT \
    --vllm-url https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run \
    --num-rollouts 50 \
    --n-samples 2 \
    --agent-step-limit 30

# 3. Stop Modal when done
modal app stop slime-grpo-vllm
```

## Prerequisites

- **Local GPU**: 24GB+ VRAM (for LoRA training)
- **Docker**: With SWE-bench images (`docker images | grep swebench`)
- **Modal**: Account configured (`modal token new`)

## Architecture

```
Modal vLLM                    Local Machine
├── Model inference           ├── mini-swe-agent-plus (multi-turn)
├── Token IDs + logprobs      ├── Docker (swebench containers)
└── Completions API           ├── swebench.harness (evaluation)
                              └── GRPO training (GPU)
```

## Parameters (Search-R1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 1e-6 | Learning rate |
| `kl_coef` | 0.001 | KL divergence coefficient |
| `temperature` | 1.0 | Sampling temperature |
| `eps_clip` | 0.2 | Lower PPO clip bound |
| `eps_clip_high` | 0.28 | Upper PPO clip bound (DAPO) |
| `n_samples` | 2-4 | Samples per instance for GRPO |
| `agent_step_limit` | 30 | Max agent turns |

## Reward Function

```
reward = +1.0  if swebench.harness tests PASS (resolved)
         -1.0  if swebench.harness tests FAIL (not resolved)
```

Evaluation uses `swebench.harness.run_evaluation` - no heuristics.

## GRPO Loss

```python
# 1. Group-relative advantages
advantages = (reward - mean(rewards)) / std(rewards)

# 2. Policy loss (PPO-style with DAPO clipping)
ratio = exp(policy_logprobs - old_logprobs)
clipped = clamp(ratio, 1-0.2, 1+0.28)
policy_loss = max(-adv * ratio, -adv * clipped)

# 3. KL loss (low-variance)
kl = 0.5 * (exp(policy - ref) - 1)^2

# 4. Total
loss = policy_loss + 0.001 * kl_loss
```

## Key Files

| File | Description |
|------|-------------|
| `mini_swe_grpo_trainer.py` | Main training script using mini-swe-agent-plus |
| `grpo_agent.py` | GRPO-compatible wrapper for mini-swe-agent-plus |
| `grpo_core.py` | Shared GRPO implementation (SLiME ppo_utils) |
| `modal_vllm.py` | Modal vLLM deployment |
| `swebench_utils.py` | Docker container management |

**Submodule:** `submodules/mini-swe-agent-plus/` (multi-turn agent with bash + edit tools)

## Agent Config

Uses `swebench_add_edit_tool.yaml` from mini-swe-agent-plus:
- Bash commands for exploration
- `edit_via_str_replace` for safe code editing
- THOUGHT + command format

## CLI Arguments

```
--model-name        Model name (default: Kwai-Klear/Klear-AgentForge-8B-SFT)
--vllm-url          vLLM server URL
--num-rollouts      Number of instances to train on
--n-samples         Samples per instance for GRPO (default: 2)
--lr                Learning rate (default: 1e-6)
--kl-coef           KL coefficient (default: 0.001)
--temperature       Sampling temperature (default: 1.0)
--agent-step-limit  Max agent turns (default: 30)
--output-dir        Output directory for checkpoints
--save-every        Save checkpoint every N rollouts (default: 10)
--eval-timeout      Evaluation timeout in seconds (default: 300)
```
