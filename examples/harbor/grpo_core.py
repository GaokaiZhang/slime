"""
GRPO Core Implementation - Using SLiME's ppo_utils.py.

This module contains the shared GRPO implementation used by both:
- hybrid_grpo_trainer.py (Modal GPU + local Docker)
- local_gpu_grpo_trainer.py (local GPU + local Docker)

IMPORTANT: This module uses SLiME's battle-tested implementations from
slime/utils/ppo_utils.py for:
- KL divergence computation (compute_approx_kl)
- Policy loss computation (compute_policy_loss)

Both training paths use IDENTICAL GRPO implementation with:
- Search-R1 hyperparameters
- swebench.harness evaluation ONLY (no heuristics)

Search-R1 GRPO Hyperparameters:
- lr: 1e-6
- kl_coef: 0.001
- kl_loss_type: low_var_kl
- n_samples_per_prompt: 4-5
- temperature: 1.0
- eps_clip: 0.2 (lower bound)
- eps_clip_high: 0.28 (upper bound, DAPO-style)
- gamma: 1.0 (no discounting)
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Import SLiME's GRPO utilities
from slime.utils.ppo_utils import (
    compute_approx_kl as slime_compute_kl,
    compute_policy_loss as slime_compute_policy_loss,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class GRPOConfig:
    """
    GRPO training configuration with Search-R1 defaults.

    These are the EXACT hyperparameters from the Search-R1 paper.
    """
    # Model
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"

    # GRPO hyperparameters (Search-R1)
    lr: float = 1e-6
    kl_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"  # SLiME's kl_loss_type
    n_samples_per_prompt: int = 4
    temperature: float = 1.0
    gamma: float = 1.0  # No discounting
    eps_clip: float = 0.2  # Lower clip bound
    eps_clip_high: float = 0.28  # Upper clip bound (DAPO-style)

    # Training
    gradient_accumulation_steps: int = 4
    max_new_tokens: int = 2048
    max_prompt_len: int = 4096

    # LoRA (for memory efficiency)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Evaluation
    eval_timeout: int = 300  # seconds per swebench evaluation


@dataclass
class RolloutSample:
    """Single rollout sample with all data needed for GRPO."""
    response_text: str = ""
    token_ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    reward: float = -1.0
    instance_id: str = ""
    patch: str = ""


# ==============================================================================
# Patch Extraction
# ==============================================================================

def extract_patch(response: str) -> str:
    """
    Extract git diff patch from model response.

    Tries multiple patterns to find the patch:
    1. Code fence with diff annotation
    2. Code fence without annotation
    3. Raw diff format
    """
    # Pattern 1: Code fence with diff
    diff_pattern = r'```(?:diff)?\n((?:---|\+\+\+|@@|[-+ ].*\n?)+)```'
    match = re.search(diff_pattern, response, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Pattern 2: Raw diff
    lines = response.split('\n')
    diff_lines = []
    in_diff = False

    for line in lines:
        if line.startswith('--- ') or line.startswith('+++ ') or line.startswith('@@'):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
            # End diff at empty line after content
            if not line.strip() and len(diff_lines) > 5:
                break

    if diff_lines:
        return '\n'.join(diff_lines)

    return ""


# ==============================================================================
# swebench.harness Evaluation (NO HEURISTICS)
# ==============================================================================

def evaluate_with_swebench(
    instance_id: str,
    patch: str,
    timeout: int = 300,
    run_id_prefix: str = "grpo",
) -> float:
    """
    Evaluate patch using swebench.harness.

    THIS IS THE ONLY REWARD FUNCTION - NO HEURISTICS.

    Requires:
    - Docker running locally
    - SWE-bench images pulled
    - swebench package installed

    Args:
        instance_id: SWE-bench instance ID (e.g., "django__django-12345")
        patch: Git diff patch to evaluate
        timeout: Timeout in seconds for evaluation
        run_id_prefix: Prefix for run ID

    Returns:
        +1.0 if patch resolves the issue (all tests pass)
        -1.0 otherwise
    """
    if not patch or not patch.strip():
        logger.info(f"[{instance_id}] No patch provided -> reward=-1.0")
        return -1.0

    try:
        # Create temporary prediction file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            prediction = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": f"{run_id_prefix}-model",
            }
            f.write(json.dumps(prediction) + "\n")
            pred_file = f.name

        # Run swebench.harness evaluation
        run_id = f"{run_id_prefix}_{instance_id.replace('/', '_')}_{int(time.time())}"

        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--split", "test",
            "--predictions_path", pred_file,
            "--max_workers", "1",
            "--timeout", str(timeout),
            "--run_id", run_id,
            "--instance_ids", instance_id,
        ]

        logger.info(f"[{instance_id}] Running swebench.harness evaluation...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 120,  # Extra time for setup
            cwd=os.getcwd(),
        )

        if result.returncode != 0:
            logger.warning(f"[{instance_id}] swebench returned non-zero: {result.stderr[:500]}")

        # Find evaluation result file
        eval_file = None
        model_name = f"{run_id_prefix}-model"
        for pattern in [
            f"{model_name}.{run_id}.json",
            f"*{run_id}*.json",
        ]:
            matches = list(Path('.').glob(pattern))
            if matches:
                eval_file = matches[0]
                break

        if eval_file and eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)

            resolved_ids = eval_data.get("resolved_ids", [])

            # Clean up
            try:
                eval_file.unlink()
            except:
                pass
            try:
                os.unlink(pred_file)
            except:
                pass

            if instance_id in resolved_ids:
                logger.info(f"[{instance_id}] RESOLVED! -> reward=+1.0")
                return 1.0
            else:
                logger.info(f"[{instance_id}] Not resolved -> reward=-1.0")
                return -1.0
        else:
            logger.warning(f"[{instance_id}] No evaluation result file found")
            try:
                os.unlink(pred_file)
            except:
                pass
            return -1.0

    except subprocess.TimeoutExpired:
        logger.error(f"[{instance_id}] Evaluation timeout")
        return -1.0
    except Exception as e:
        logger.error(f"[{instance_id}] Evaluation error: {e}")
        return -1.0


# ==============================================================================
# GRPO Advantage Computation
# ==============================================================================

def compute_grpo_advantages(
    rewards: list[float],
    normalize: bool = True,
) -> tuple[list[float], float, float]:
    """
    Compute GRPO group-relative advantages.

    For each reward r_i in the group:
        advantage_i = (r_i - mean) / std

    Args:
        rewards: List of rewards for samples in the group
        normalize: Whether to normalize by std (True for standard GRPO)

    Returns:
        Tuple of (advantages, mean_reward, std_reward)
    """
    mean_reward = sum(rewards) / len(rewards)

    if normalize and len(rewards) > 1:
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance ** 0.5
        std_reward = max(std_reward, 1e-8)  # Avoid division by zero
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        std_reward = 0.0
        advantages = [r - mean_reward for r in rewards]

    return advantages, mean_reward, std_reward


# ==============================================================================
# GRPO Loss Computation (Using SLiME's ppo_utils.py)
# ==============================================================================

def compute_kl_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_loss_type: str = "low_var_kl",
) -> torch.Tensor:
    """
    Compute KL divergence loss using SLiME's implementation.

    Uses slime.utils.ppo_utils.compute_approx_kl which supports:
    - k1: Simple KL approximation
    - k2: Squared KL approximation
    - k3/low_var_kl: Low-variance KL (Schulman's blog)

    Args:
        policy_log_probs: Log probs from current policy
        ref_log_probs: Log probs from reference policy
        kl_loss_type: Type of KL approximation (default: "low_var_kl")

    Returns:
        KL loss tensor (mean over tokens)
    """
    # SLiME's compute_approx_kl expects (log_probs, log_probs_base, kl_loss_type)
    # where log_probs is current policy and log_probs_base is reference
    kl = slime_compute_kl(
        log_probs=policy_log_probs,
        log_probs_base=ref_log_probs,
        kl_loss_type=kl_loss_type,
    )
    return kl.mean()


def compute_policy_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PPO-style clipped policy loss using SLiME's implementation.

    Uses slime.utils.ppo_utils.compute_policy_loss which supports:
    - Asymmetric clipping (DAPO-style)
    - Optional dual-clip PPO

    Args:
        policy_log_probs: Log probs from current policy
        old_log_probs: Log probs from rollout policy (π_old)
        advantages: Advantages tensor (same shape as log_probs)
        eps_clip: Lower clip bound (default: 0.2)
        eps_clip_high: Upper clip bound (default: 0.28, DAPO-style)

    Returns:
        Tuple of (policy_loss, clip_fraction)
    """
    # SLiME's compute_policy_loss expects ppo_kl = log_probs_old - log_probs
    ppo_kl = old_log_probs - policy_log_probs

    # Call SLiME's implementation
    pg_losses, clipfrac = slime_compute_policy_loss(
        ppo_kl=ppo_kl,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
        eps_clip_c=None,  # No dual-clip
    )

    return pg_losses.mean(), clipfrac.mean()


def compute_grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantage: float,
    config: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Compute full GRPO loss for a single sample using SLiME's utilities.

    Total loss = policy_loss + kl_coef * kl_loss

    Args:
        policy_log_probs: Log probs from current policy
        old_log_probs: Log probs from rollout policy
        ref_log_probs: Log probs from reference policy
        advantage: Group-relative advantage (scalar)
        config: GRPO configuration

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Expand advantage to match log_probs shape
    advantages_tensor = torch.full_like(policy_log_probs, advantage)

    # Policy loss using SLiME's implementation
    policy_loss, clipfrac = compute_policy_loss(
        policy_log_probs,
        old_log_probs,
        advantages_tensor,
        eps_clip=config.eps_clip,
        eps_clip_high=config.eps_clip_high,
    )

    # KL loss using SLiME's implementation
    kl_loss = compute_kl_loss(
        policy_log_probs,
        ref_log_probs,
        kl_loss_type=config.kl_loss_type,
    )

    # Total loss
    total_loss = policy_loss + config.kl_coef * kl_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "clipfrac": clipfrac.item(),
    }

    return total_loss, metrics


# ==============================================================================
# Prompt Template
# ==============================================================================

def create_swebench_prompt(instance: dict) -> str:
    """
    Create prompt for SWE-bench instance.

    Args:
        instance: SWE-bench instance dict with 'repo' and 'problem_statement'

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert software engineer. Fix this bug and provide a git diff patch.

## Repository: {instance["repo"]}

## Problem Statement
{instance["problem_statement"][:3000]}

## Instructions
Analyze the problem and provide a git diff patch that fixes the issue.

Format your patch as:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
-old line
+new line
```

Provide ONLY the patch, no explanations.

Your patch:"""


# ==============================================================================
# Utility Functions
# ==============================================================================

def setup_lora(model, config: GRPOConfig):
    """Apply LoRA to model for memory-efficient training."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def compute_response_log_probs(
    model,
    tokenizer,
    prompt: str,
    response_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute log probabilities for response tokens.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        response_ids: Response token IDs
        device: Torch device

    Returns:
        Tensor of log probabilities for each response token
    """
    # Tokenize prompt
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)
    prompt_length = prompt_inputs["input_ids"].shape[1]

    # Build full sequence
    full_ids = torch.cat([
        prompt_inputs["input_ids"][0],
        response_ids.to(device)
    ]).unsqueeze(0)

    # Forward pass
    outputs = model(full_ids, return_dict=True)
    logits = outputs.logits

    # Compute log probs for response tokens
    response_start = prompt_length - 1
    response_end = full_ids.shape[1] - 1

    log_probs = F.log_softmax(logits[0, response_start:response_end], dim=-1)
    token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs
