#!/usr/bin/env python
"""
Harbor GRPO Core: Shared Implementation

This module contains shared code for Harbor-based GRPO training:
- Configuration with Search-R1 parameters
- Harbor agent rollout functions
- swebench.harness evaluation
- GRPO training step using SLiME's ppo_utils

Used by:
- harbor_grpo_local.py (local GPU training)
- harbor_grpo_modal.py (Modal GPU training)
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class HarborGRPOConfig:
    """Configuration for Harbor GRPO training with Search-R1 parameters."""

    # Model
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"

    # Harbor agent and environment
    agent: str = "qwen-coder"  # Built-in: qwen-coder, mini-swe-agent, claude-code, openhands, etc.
    agent_model: str = None  # Model for agent (e.g., openai/gpt-4o, openai/local-model). Defaults to training model.
    agent_import_path: str = None  # Custom agent import path (e.g., for mini-swe-agent-plus)
    env: str = "docker"  # Environment: "docker" (local) or "daytona" (cloud)
    dataset: str = "swebench-verified@1.0"  # Dataset to use
    n_concurrent: int = 1

    # GRPO training (Search-R1 parameters)
    n_samples_per_prompt: int = 4  # GRPO group size
    lr: float = 1e-6               # Search-R1
    kl_coef: float = 0.001         # Search-R1
    kl_loss_type: str = "low_var_kl"  # Search-R1
    eps_clip: float = 0.2          # PPO clip
    eps_clip_high: float = 0.28    # DAPO asymmetric

    # Generation
    temperature: float = 1.0
    max_new_tokens: int = 2048

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    gradient_checkpointing: bool = True

    # Evaluation
    eval_timeout: int = 300

    # OpenAI API (for agents like qwen-coder)
    openai_base_url: str = None
    openai_api_key: str = "local"

    # Output
    output_dir: str = "outputs/harbor_grpo"
    jobs_dir: str = "jobs"
    save_every: int = 10


# ==============================================================================
# Harbor Rollout Functions
# ==============================================================================

def run_harbor_agent(
    instance: dict,
    config: HarborGRPOConfig,
    timeout: int = 1800,
) -> dict:
    """
    Run Harbor agent on a single SWE-bench instance.

    Args:
        instance: SWE-bench instance dict with instance_id
        config: Training configuration
        timeout: Timeout in seconds

    Returns:
        {"response": str, "patch": str, "status": str}
    """
    instance_id = instance["instance_id"]
    job_name = f"grpo-{instance_id.replace('/', '_')}-{int(time.time())}"

    # Build Harbor command
    cmd = [
        "harbor", "run",
        "--env", config.env,
        "--agent", config.agent,
        "--dataset", config.dataset,
        "--task-name", instance_id,
        "--n-concurrent", str(config.n_concurrent),
        "--jobs-dir", config.jobs_dir,
        "--job-name", job_name,
        "--export-traces",
    ]

    # Add model for agent if specified (e.g., openai/gpt-4o or openai/local-model)
    if config.agent_model:
        cmd.extend(["--model", config.agent_model])

    # Add custom agent import path if specified
    if config.agent_import_path:
        cmd.extend(["--agent-import-path", config.agent_import_path])

    # Add OpenAI API configuration for agents that need it (e.g., qwen-coder)
    if config.openai_base_url:
        cmd.extend(["--ak", f"base_url={config.openai_base_url}"])
    if config.openai_api_key:
        cmd.extend(["--ak", f"api_key={config.openai_api_key}"])

    # For qwen-coder, pass the model name via --model (not --ak model=...)
    # The agent expects model_name parameter which is set by --model CLI arg
    if config.agent == "qwen-coder" and config.model_name and not config.agent_model:
        cmd.extend(["--model", config.model_name])

    logger.info(f"  Running Harbor: {config.agent} on {instance_id}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        job_dir = Path(config.jobs_dir) / job_name

        if result.returncode != 0:
            logger.warning(f"  Harbor failed: {result.stderr[:200]}")
            return {"response": "", "patch": "", "status": "failed", "job_dir": str(job_dir)}

        # Parse trajectory from job directory (if available)
        # Note: Some agents like qwen-coder modify code directly without trajectory files
        response, patch = parse_harbor_trajectory(job_dir)

        # Harbor succeeded - trust its evaluation (reward.txt)
        # Status is "completed" even if we can't parse trajectory files
        return {
            "response": response,
            "patch": patch,
            "status": "completed",
            "job_dir": str(job_dir),
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"  Harbor timeout for {instance_id}")
        return {"response": "", "patch": "", "status": "timeout", "job_dir": ""}
    except Exception as e:
        logger.error(f"  Harbor error: {e}")
        return {"response": "", "patch": "", "status": "error", "job_dir": ""}


def parse_harbor_trajectory(job_dir: Path) -> tuple[str, str]:
    """Parse Harbor trajectory to extract response and patch."""
    import re

    response = ""
    patch = ""

    # Look for trajectory file
    for traj_file in job_dir.glob("**/trajectory*.json"):
        try:
            with open(traj_file) as f:
                data = json.load(f)

            # Extract messages
            messages = data.get("messages", [])
            if not messages and "trajectory" in data:
                # ATIF format
                for step in data["trajectory"]:
                    if "action" in step:
                        response += step["action"] + "\n"
            else:
                for msg in messages:
                    if msg.get("role") == "assistant":
                        response += msg.get("content", "") + "\n"

            # Extract patch from response
            patch = extract_patch_from_response(response)
            break

        except Exception as e:
            logger.debug(f"  Error parsing {traj_file}: {e}")

    return response.strip(), patch


def extract_patch_from_response(response: str) -> str:
    """Extract git diff patch from response text."""
    import re

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
            if not line.strip() and len(diff_lines) > 5:
                break

    if diff_lines:
        return '\n'.join(diff_lines)

    return ""


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_with_swebench(
    instance_id: str,
    patch: str,
    timeout: int = 300,
) -> float:
    """
    Evaluate patch using swebench.harness.

    Returns +1.0 if resolved, -1.0 otherwise.
    """
    if not patch or not patch.strip():
        logger.info(f"  [{instance_id}] No patch -> reward=-1.0")
        return -1.0

    try:
        # Create prediction file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            prediction = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "harbor-grpo",
            }
            f.write(json.dumps(prediction) + "\n")
            pred_file = f.name

        run_id = f"harbor_grpo_{instance_id.replace('/', '_')}_{int(time.time())}"

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

        logger.info(f"  [{instance_id}] Running swebench.harness...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 120,
        )

        # Find evaluation result
        eval_file = None
        for pattern in [f"harbor-grpo.{run_id}.json", f"*{run_id}*.json"]:
            matches = list(Path('.').glob(pattern))
            if matches:
                eval_file = matches[0]
                break

        if eval_file and eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)

            resolved_ids = eval_data.get("resolved_ids", [])

            # Cleanup
            try:
                eval_file.unlink()
                os.unlink(pred_file)
            except:
                pass

            if instance_id in resolved_ids:
                logger.info(f"  [{instance_id}] RESOLVED! -> reward=+1.0")
                return 1.0
            else:
                logger.info(f"  [{instance_id}] Not resolved -> reward=-1.0")
                return -1.0

        logger.warning(f"  [{instance_id}] No eval result file")
        return -1.0

    except subprocess.TimeoutExpired:
        logger.error(f"  [{instance_id}] Eval timeout")
        return -1.0
    except Exception as e:
        logger.error(f"  [{instance_id}] Eval error: {e}")
        return -1.0


# ==============================================================================
# Data Loading
# ==============================================================================

def load_training_instances(
    num_instances: int = 50,
    test_mode: bool = False,
) -> list[dict]:
    """
    Load SWE-bench training instances.

    Args:
        num_instances: Maximum number of instances to load
        test_mode: If True, load only 5 instances

    Returns:
        List of instance dicts with instance_id, problem_statement, repo
    """
    from datasets import load_dataset

    if test_mode:
        num_instances = min(5, num_instances)

    # Try to load from file first
    train_file = Path("/home/ubuntu/slime/train_instances_id.txt")
    if train_file.exists():
        with open(train_file) as f:
            instance_ids = [line.strip() for line in f if line.strip()]
        instance_ids = instance_ids[:num_instances]

        # Load full dataset to get problem statements
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        id_to_instance = {item["instance_id"]: item for item in dataset}

        instances = []
        for iid in instance_ids:
            if iid in id_to_instance:
                item = id_to_instance[iid]
                instances.append({
                    "instance_id": iid,
                    "problem_statement": item["problem_statement"],
                    "repo": item["repo"],
                })

        logger.info(f"Loaded {len(instances)} instances from {train_file}")
        return instances

    # Fall back to loading from HuggingFace
    logger.info("Loading from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    django_instances = [
        {
            "instance_id": item["instance_id"],
            "problem_statement": item["problem_statement"],
            "repo": item["repo"],
        }
        for item in dataset
        if "django" in item["instance_id"].lower()
    ]

    logger.info(f"Found {len(django_instances)} Django instances")
    return django_instances[:num_instances]


def create_swebench_prompt(instance: dict) -> str:
    """Create prompt for SWE-bench instance."""
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
# GRPO Training Step
# ==============================================================================

def train_grpo_step(
    prompt: str,
    responses: list[str],
    rewards: list[float],
    model,
    ref_model,
    tokenizer,
    optimizer,
    config: HarborGRPOConfig,
    device,
) -> dict:
    """
    Perform one GRPO training step using SLiME's ppo_utils.

    Key Insight:
    Responses come from Harbor agents (text only, no log probs).
    We compute log probs at training time via forward pass.

    Args:
        prompt: The SWE-bench prompt
        responses: List of response texts from Harbor agent
        rewards: List of rewards from swebench.harness evaluation
        model: Policy model (trainable)
        ref_model: Reference model (frozen)
        tokenizer: Tokenizer
        optimizer: Optimizer
        config: Training configuration
        device: Device to use

    Returns:
        Training metrics dict
    """
    import torch
    import torch.nn.functional as F

    # Import SLiME's GRPO utilities
    from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss

    # Compute group-relative advantages
    mean_reward = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = max(variance ** 0.5, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        std_reward = 0.0
        advantages = [r - mean_reward for r in rewards]

    logger.info(f"  Mean reward: {mean_reward:.3f}, Std: {std_reward:.3f}")

    # Training
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    n_valid = 0

    # Max response tokens to prevent OOM
    max_response_tokens = 4096

    for response_text, reward, advantage in zip(responses, rewards, advantages):
        if not response_text.strip():
            continue

        # Tokenize prompt + response
        full_text = prompt + response_text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(device)

        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        full_ids = inputs["input_ids"]
        response_length = full_ids.shape[1] - prompt_length

        if response_length <= 0:
            continue

        # Truncate if too long
        if response_length > max_response_tokens:
            logger.info(f"  Truncating response from {response_length} to {max_response_tokens}")
            full_ids = full_ids[:, :prompt_length + max_response_tokens]
            response_length = max_response_tokens

        # Forward pass through policy model
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            policy_outputs = model(full_ids, return_dict=True)
            policy_logits = policy_outputs.logits

            # Forward pass through reference model
            with torch.no_grad():
                ref_outputs = ref_model(full_ids, return_dict=True)
                ref_logits = ref_outputs.logits

        # Compute log probs for response tokens only
        response_logits = policy_logits[0, prompt_length - 1:-1]  # Shifted for next-token prediction
        ref_response_logits = ref_logits[0, prompt_length - 1:-1]
        response_ids = full_ids[0, prompt_length:]

        policy_log_probs = F.log_softmax(response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)

        # Gather log probs for actual tokens
        policy_token_log_probs = policy_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        # For GRPO without old log probs, we use current policy as "old"
        # This is equivalent to the first iteration of PPO
        old_log_probs = policy_token_log_probs.detach()

        # Compute PPO-style KL for policy loss
        ppo_kl = old_log_probs - policy_token_log_probs
        advantages_tensor = torch.full_like(policy_token_log_probs, advantage)

        # Use SLiME's policy loss
        pg_losses, clipfrac = compute_policy_loss(
            ppo_kl=ppo_kl,
            advantages=advantages_tensor,
            eps_clip=config.eps_clip,
            eps_clip_high=config.eps_clip_high,
            eps_clip_c=None,
        )
        policy_loss = pg_losses.mean()

        # Use SLiME's KL loss
        kl = compute_approx_kl(
            log_probs=policy_token_log_probs,
            log_probs_base=ref_token_log_probs,
            kl_loss_type=config.kl_loss_type,
        )
        kl_loss = kl.mean()

        # Total loss
        loss = policy_loss + config.kl_coef * kl_loss
        loss.backward()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()
        n_valid += 1

    # Update weights
    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logger.info(f"  Updated weights with {n_valid} samples")

    return {
        "loss": total_loss / max(n_valid, 1),
        "policy_loss": total_policy_loss / max(n_valid, 1),
        "kl_loss": total_kl_loss / max(n_valid, 1),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_valid": n_valid,
    }


# ==============================================================================
# Model Loading Utilities
# ==============================================================================

def load_model_and_tokenizer(
    model_name: str,
    device,
    use_lora: bool = True,
    lora_r: int = 16,
    gradient_checkpointing: bool = True,
):
    """Load model and tokenizer with optional LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def load_reference_model(model_name: str, device):
    """Load frozen reference model for KL computation."""
    import torch
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading reference model: {model_name}")

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model
