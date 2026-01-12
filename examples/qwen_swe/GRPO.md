# GRPO Training for SWE-bench

## Quick Start

### Option A: Local GPU

```bash
# Terminal 1: Start local vLLM server (requires 8x A100-80GB)
python examples/qwen_swe/start_vllm.py --port 8000 --tp 8

# Terminal 2: Run training
cd /home/gaokaizhang/slime
bash examples/qwen_swe/run_grpo.sh
```

### Option B: Modal GPU

```bash
# Terminal 1: Deploy Modal vLLM server
modal deploy examples/qwen_swe/modal_inference.py

# Terminal 2: Run training with Modal URL
export VLLM_URL="https://susvibes-mitigation--qwen-swe-inference-serve-vllm.modal.run"
bash examples/qwen_swe/run_grpo.sh

# When done, stop Modal to save costs
modal app stop qwen-swe-inference
```

---

## Architecture

This implementation uses SLiME's full training infrastructure for correctness:

```
vLLM Server (local OR Modal)
         |
Local Docker + qwen-code  -->  vLLM inference
         |
         v
swebench.harness (reward evaluation)
         |
         v
SLiME train.py --> FSDP Training (multi-GPU)
```

Key SLiME integrations:
- `DataSource` base class for data loading
- `train.py` for distributed training
- `MultiTurnLossMaskGenerator` for proper loss masking
- FSDP backend for efficient multi-GPU training

---

## Key Settings

### GRPO Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Group size | 2 | `--n_samples_per_prompt` per instance |
| Learning rate | 1e-6 | Default for fine-tuning |
| Advantage estimator | grpo | Group Relative Policy Optimization |
| Train backend | fsdp | Fully Sharded Data Parallel |

### Rollout Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rollout batch size | 4 | `--rollout_batch_size` |
| Max response length | 16384 | Token limit for response |
| Agent timeout | 1800s | Per rollout |
| Max turns | 50 | Agent conversation limit |

### Hardware Requirements

| Component | Resources |
|-----------|-----------|
| vLLM Inference | 8x A100-80GB (TP=8) |
| FSDP Training | 8x A100-80GB |
| Rollouts | Local Docker containers |

---

## Files

| File | Description |
|------|-------------|
| `run_grpo.sh` | Training launch script |
| `run_qwen_swe.py` | Training entry point (configures SLiME's train.py) |
| `start_vllm.py` | Local vLLM server launcher |
| `modal_inference.py` | Modal vLLM server deployment |
| `rollout.py` | Rollout function with loss masking |
| `qwen_agent.py` | qwen-code CLI wrapper |
| `rewards.py` | swebench.harness evaluation |
| `data_source.py` | SWE-bench data loader (extends SLiME DataSource) |
| `prompts.py` | Bug solving prompts |

---

## Loss Masking

Proper loss masking is implemented using SLiME's `MultiTurnLossMaskGenerator`:

| Message Type | Loss Mask | Training |
|--------------|-----------|----------|
| Assistant (model output) | 1 | ✓ Train |
| Tool responses | 0 | ✗ Skip |
| User messages | 0 | ✗ Skip |

This ensures we only train on model-generated content, not tool outputs.

---

## Reward Function

| Outcome | Reward |
|---------|--------|
| All FAIL_TO_PASS tests pass | +1.0 |
| Tests fail / No patch | -1.0 |

Uses official `swebench.harness.run_evaluation` for accurate rewards.

---

## GRPO Advantage Computation

For each instance group:
```
advantage_i = reward_i - mean(group_rewards)
```

Example with group_size=2:
```
instance_A: sample_0=+1, sample_1=-1 → mean=0 → advantages: +1, -1
```

---

## Data

- Dataset: `princeton-nlp/SWE-bench_Verified`
- Train: 201 Django instances
- Instance list: `/home/gaokaizhang/SWE-sft/data/raw/splits/train_201_django.txt`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://localhost:8000` | vLLM server URL |
| `MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Model name |
| `NUM_ROLLOUTS` | 10 | Total rollout iterations |
| `ROLLOUT_BATCH_SIZE` | 4 | Instances per batch |
| `N_SAMPLES` | 2 | Samples per instance (group size) |
| `OUTPUT_DIR` | `/tmp/qwen_swe_grpo` | Checkpoint directory |

---

## Notes

1. **Loss masking**: Implemented via `MultiTurnLossMaskGenerator` (trains only on assistant messages)
2. **Evaluation**: Uses official `swebench.harness.run_evaluation`
3. **Checkpoints**: Saved to `OUTPUT_DIR` with SLiME's checkpoint format
4. **SLiME Integration**: Uses SLiME's DataSource, train.py, and FSDP backend
