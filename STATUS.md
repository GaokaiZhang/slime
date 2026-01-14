# SSR on SLiME: Status

## Current State: Harbor Integration (2026-01-14)

Added Harbor framework as submodule for improved agent evaluation and RL rollouts.
Uses ATIF (Agent Trajectory Interchange Format) for trajectory logging.

### Submodules Setup

```
submodules/
├── qwen-code/    # Qwen CLI tool (Node.js)
└── harbor/       # Harbor evaluation framework
```

### Search-R1 GRPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 1e-6 | Learning rate (for 7-8B models) |
| `kl_loss_coef` | 0.001 | KL loss coefficient |
| `kl_loss_type` | low_var_kl | Low-variance KL approximation |
| `n_samples_per_prompt` | 5 | Samples per prompt (n_agent) |
| `temperature` | 1.0 | Full randomness for GRPO diversity |
| `gamma` | 1.0 | No discounting |

### Search-R1 Reward/Loss Functions

**Reward (SWE-bench):**
```python
reward = +1.0 if all_tests_pass else -1.0
```

**GRPO Advantage (per prompt group):**
```python
μ = mean(rewards_in_group)
σ = std(rewards_in_group)
advantage = (reward - μ) / (σ + ε)
```

**Loss:**
```python
loss = pg_loss - entropy_coeff * entropy_loss + kl_loss_coef * kl_loss
```

### Architecture

```
Modal Cloud (GPU)
├── vLLM Server (A100-80GB)
│   └── Qwen/Qwen3-Coder-30B-A3B-Instruct
│
│         ▲ HTTP API (with logprobs for RL)
│         │
Local Machine
├── rl_agent.py
│   ├── Calls Modal vLLM API
│   ├── Records ATIF trajectory
│   ├── Includes token_ids + logprobs
│   └── Parses JSON tool calls
│
└── Docker Containers (swebench)
    ├── Execute tools (read, write, grep, shell)
    └── Run tests for evaluation
```

### Harbor Integration

Harbor provides:
- **ATIF Format**: Standard trajectory format with `completion_token_ids` and `logprobs`
- **Agent Evaluation**: Run agents on SWE-bench tasks
- **Modal Environment**: Built-in Modal support for parallel execution
- **SWE-bench Adapter**: Pre-built adapter for SWE-Bench_Verified

### File Structure

```
examples/harbor/
├── __init__.py          # Package init
├── rl_agent.py          # RL agent with ATIF trajectory
├── rollout.py           # SLiME rollout function
├── data_source.py       # SWE-Bench_Verified DataSource
├── swebench_env.py      # Docker container management
├── rewards.py           # Patch evaluation
├── modal_vllm.py        # Modal vLLM deployment
├── run_grpo.sh          # GRPO training script
└── test_pipeline.py     # Pipeline test

examples/qwen_swe/       # Previous implementation (deprecated)
├── simple_agent.py
├── rollout.py
└── ...
```

### Usage

```bash
# Activate hb_train environment (Python 3.12)
conda activate hb_train

# Deploy Modal vLLM server
modal deploy examples/harbor/modal_vllm.py

# Test the pipeline
cd examples/harbor
python test_pipeline.py

# Run GRPO training
export VLLM_URL="https://susvibes-mitigation--harbor-grpo-vllm-serve-vllm.modal.run"
bash examples/harbor/run_grpo.sh

# Stop Modal when done
modal app stop harbor-grpo-vllm
```

### ATIF Trajectory Format

The RL agent records ATIF-compatible trajectories:

```json
{
  "schema_version": "ATIF-v1.5",
  "session_id": "instance_123_timestamp",
  "agent": {
    "name": "slime-rl-agent",
    "version": "1.0.0",
    "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "tool_definitions": [...]
  },
  "steps": [
    {
      "step_id": 1,
      "source": "user",
      "message": "..."
    },
    {
      "step_id": 2,
      "source": "agent",
      "message": "...",
      "tool_calls": [...],
      "observation": {"results": [...]},
      "metrics": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "completion_token_ids": [...],  // For RL training
        "logprobs": [...]               // For RL training
      }
    }
  ],
  "final_metrics": {...}
}
```

### Loss Mask Verification

Previous test results (simple_agent):
```
Total tokens: 3493
Train (mask=1): 374   ← assistant outputs only
Skip (mask=0): 3119   ← system, user, tool responses
Train ratio: 10.7%
```

### Requirements

- **Python**: 3.12+ (hb_train conda environment)
- **Inference**: Modal A100-80GB (vLLM)
- **Rollouts**: Local Docker + swebench images
- **Training**: Modal or local GPU

### Notes

- Harbor uses ATIF-v1.5 for trajectory logging
- Token IDs and logprobs are recorded for RL training
- qwen-code CLI has streaming issues with vLLM (use rl_agent.py instead)
