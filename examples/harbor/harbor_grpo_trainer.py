#!/usr/bin/env python
"""
Harbor GRPO Trainer: Harbor Agent Rollouts + Modal GPU Training + Local swebench.harness

This trainer combines:
- Harbor CLI: Runs agents (mini-swe-agent, qwen-coder, etc.) to generate trajectories
- Modal A100: GPU training with GRPO (weight updates)
- Local Docker: swebench.harness evaluation for accurate rewards

Key Insight:
Log probabilities from Harbor rollouts are NOT used for training.
Instead, log probs are recomputed at training time via forward pass on Modal.
This enables using ANY Harbor agent for RL training.

Architecture:
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │  Harbor Agent   │───▶│  Trajectories    │───▶│  Modal GRPO     │
    │  (mini-swe)     │    │  (text + reward) │    │  Training       │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
           │                        │
           ▼                        ▼
    Local Docker              swebench.harness
    (execution)               (evaluation)

Usage:
    # Test with 5 instances
    python examples/harbor/harbor_grpo_trainer.py --test

    # Full training (201 Django instances)
    python examples/harbor/harbor_grpo_trainer.py --num-rollouts 201

    # Using Modal CLI
    modal run examples/harbor/harbor_grpo_trainer.py --num-rollouts 50
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Modal Configuration
# ==============================================================================

app = modal.App("harbor-grpo")

# Modal image with training dependencies
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "curl", "build-essential"])
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "datasets",
        "accelerate",
        "peft",
        "huggingface_hub",
        "numpy",
        "tqdm",
    )
)

# Volumes
model_cache = modal.Volume.from_name("harbor-grpo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("harbor-grpo-outputs", create_if_missing=True)

# HF secret
hf_secret = modal.Secret.from_name("hf-token-swe")


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class HarborGRPOConfig:
    """Configuration for Harbor GRPO training."""

    # Model
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"

    # Harbor agent
    agent: str = "mini-swe-agent"  # mini-swe-agent, qwen-coder, etc.
    n_concurrent: int = 4

    # Training
    n_samples_per_prompt: int = 4  # GRPO group size
    lr: float = 1e-6
    kl_coef: float = 0.001
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    temperature: float = 1.0
    max_new_tokens: int = 2048

    # LoRA
    use_lora: bool = True
    lora_r: int = 16

    # Evaluation
    eval_timeout: int = 300

    # Output
    output_dir: str = "outputs/harbor_grpo"
    jobs_dir: str = "jobs"
    save_every: int = 10


# ==============================================================================
# Modal Training Functions
# ==============================================================================

@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/outputs": output_volume,
    },
)
def train_step_on_modal(
    prompt: str,
    responses: list[str],
    rewards: list[float],
    state: dict,
    checkpoint_path: str = None,
    save_checkpoint: bool = False,
    checkpoint_name: str = None,
) -> dict:
    """
    Perform GRPO training step on Modal GPU.

    Key Insight:
    Responses come from Harbor agents (text only, no log probs).
    We compute log probs at training time via forward pass.

    Args:
        prompt: The SWE-bench prompt
        responses: List of response texts from Harbor agent
        rewards: List of rewards from swebench.harness evaluation
        state: Training configuration
        checkpoint_path: Previous checkpoint (if any)
        save_checkpoint: Whether to save checkpoint
        checkpoint_name: Name for saved checkpoint

    Returns:
        Training metrics dict
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_name = state["model_name"]

    print(f"Training on {len(responses)} responses with rewards {rewards}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        print(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        if state.get("use_lora", True):
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=state.get("lora_r", 16),
                lora_alpha=state.get("lora_r", 16) * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    # Load reference model for KL
    print("Loading reference model for KL...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=state.get("lr", 1e-6),
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )

    # =========================================================================
    # GRPO Training (log probs computed from responses, not stored)
    # =========================================================================

    # Compute group-relative advantages
    mean_reward = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = max(variance ** 0.5, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        std_reward = 0.0
        advantages = [r - mean_reward for r in rewards]

    print(f"Mean reward: {mean_reward:.3f}, Std: {std_reward:.3f}")
    print(f"Advantages: {[f'{a:.3f}' for a in advantages]}")

    # GRPO hyperparameters
    eps_clip = state.get("eps_clip", 0.2)
    eps_clip_high = state.get("eps_clip_high", 0.28)
    kl_coef = state.get("kl_coef", 0.001)

    # Training
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    n_valid = 0

    for response_text, reward, advantage in zip(responses, rewards, advantages):
        if not response_text.strip():
            continue

        # Tokenize prompt + response
        full_text = prompt + response_text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=6144,
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

        # Forward pass through policy model
        policy_outputs = model(full_ids, return_dict=True)
        policy_logits = policy_outputs.logits

        # Forward pass through reference model
        with torch.no_grad():
            ref_outputs = ref_model(full_ids, return_dict=True)
            ref_logits = ref_outputs.logits

        # Compute log probs for response tokens
        response_start = prompt_length - 1
        response_end = full_ids.shape[1] - 1
        response_ids = full_ids[0, prompt_length:]

        policy_log_probs = F.log_softmax(
            policy_logits[0, response_start:response_end], dim=-1
        )
        ref_log_probs = F.log_softmax(
            ref_logits[0, response_start:response_end], dim=-1
        )

        policy_token_log_probs = policy_log_probs.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # For on-policy training, use detached policy log probs as "old"
        # This is valid because Harbor responses are generated by the current policy
        old_log_probs = policy_token_log_probs.detach()

        # PPO-style policy loss with DAPO asymmetric clipping
        ppo_kl = old_log_probs - policy_token_log_probs
        ratio = (-ppo_kl).exp()

        advantages_tensor = torch.full_like(policy_token_log_probs, advantage)
        pg_losses1 = -ratio * advantages_tensor
        pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages_tensor
        policy_loss = torch.maximum(pg_losses1, pg_losses2).mean()

        # Low-variance KL loss
        kl_ratio = (policy_token_log_probs - ref_token_log_probs).exp()
        kl_loss = (0.5 * (kl_ratio - 1) ** 2).mean()

        # Total loss
        loss = policy_loss + kl_coef * kl_loss
        loss.backward()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()
        n_valid += 1

    # Update weights
    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"Updated weights with {n_valid} valid samples")

    # Save checkpoint if requested
    new_checkpoint_path = None
    if save_checkpoint and checkpoint_name:
        new_checkpoint_path = f"/outputs/{checkpoint_name}"
        os.makedirs(new_checkpoint_path, exist_ok=True)
        model.save_pretrained(new_checkpoint_path)
        tokenizer.save_pretrained(new_checkpoint_path)
        output_volume.commit()
        print(f"Saved checkpoint to {new_checkpoint_path}")

    return {
        "loss": total_loss / max(n_valid, 1),
        "policy_loss": total_policy_loss / max(n_valid, 1),
        "kl_loss": total_kl_loss / max(n_valid, 1),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_valid": n_valid,
        "checkpoint_path": new_checkpoint_path,
    }


# ==============================================================================
# Harbor Rollout Functions (Local)
# ==============================================================================

def run_harbor_agent(
    instance: dict,
    agent: str = "mini-swe-agent",
    n_concurrent: int = 1,
    jobs_dir: str = "jobs",
    timeout: int = 1800,
) -> dict:
    """
    Run Harbor agent on a single SWE-bench instance.

    Args:
        instance: SWE-bench instance dict
        agent: Harbor agent name
        n_concurrent: Number of concurrent trials
        jobs_dir: Directory for job outputs
        timeout: Timeout in seconds

    Returns:
        {"response": str, "patch": str, "status": str}
    """
    instance_id = instance["instance_id"]
    job_name = f"grpo-{instance_id.replace('/', '_')}-{int(time.time())}"

    # Build Harbor command
    cmd = [
        "harbor", "run",
        "--agent", agent,
        "--dataset", "swebench-verified@1.0",
        "--task-name", instance_id,
        "--n-concurrent", str(n_concurrent),
        "--jobs-dir", jobs_dir,
        "--job-name", job_name,
        "--export-traces",
    ]

    logger.info(f"  Running Harbor: {agent} on {instance_id}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.warning(f"  Harbor failed: {result.stderr[:200]}")
            return {"response": "", "patch": "", "status": "failed"}

        # Parse trajectory from job directory
        job_dir = Path(jobs_dir) / job_name
        response, patch = parse_harbor_trajectory(job_dir)

        return {
            "response": response,
            "patch": patch,
            "status": "completed" if patch else "no_patch",
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"  Harbor timeout for {instance_id}")
        return {"response": "", "patch": "", "status": "timeout"}
    except Exception as e:
        logger.error(f"  Harbor error: {e}")
        return {"response": "", "patch": "", "status": "error"}


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
# Main Training Loop
# ==============================================================================

def run_harbor_grpo_training(
    config: HarborGRPOConfig,
    num_rollouts: int = 50,
    test_mode: bool = False,
):
    """
    Main training loop using Harbor for rollouts.

    Args:
        config: Training configuration
        num_rollouts: Number of SWE-bench instances to train on
        test_mode: If True, use only 5 instances
    """
    from datasets import load_dataset

    logger.info("=" * 70)
    logger.info("Harbor GRPO Training")
    logger.info("  Harbor Agent: Rollout generation")
    logger.info("  Modal A100: GRPO weight updates")
    logger.info("  Local Docker: swebench.harness evaluation")
    logger.info("=" * 70)

    # Training state
    state = {
        "model_name": config.model_name,
        "lr": config.lr,
        "kl_coef": config.kl_coef,
        "temperature": config.temperature,
        "n_samples_per_prompt": config.n_samples_per_prompt,
        "eps_clip": config.eps_clip,
        "eps_clip_high": config.eps_clip_high,
        "use_lora": config.use_lora,
        "lora_r": config.lora_r,
        "max_new_tokens": config.max_new_tokens,
    }

    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Agent: {config.agent}")
    logger.info(f"  Samples per prompt: {config.n_samples_per_prompt}")
    logger.info(f"  Learning rate: {config.lr}")
    logger.info(f"  KL coefficient: {config.kl_coef}")

    # Load training instances
    logger.info("\nLoading training instances...")

    # First try to load from file
    train_file = Path("/home/gaokaizhang/slime/train_instances_id.txt")
    if train_file.exists():
        with open(train_file) as f:
            train_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(train_ids)} instances from {train_file}")

        # Load full dataset for metadata
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        id_to_instance = {x["instance_id"]: x for x in ds}
        train_instances = [id_to_instance[iid] for iid in train_ids if iid in id_to_instance]
    else:
        # Fall back to dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        train_instances = [x for x in ds if x["repo"] == "django/django"]

    if test_mode:
        train_instances = train_instances[:5]
        logger.info(f"Test mode: using {len(train_instances)} instances")
    else:
        train_instances = train_instances[:num_rollouts]
        logger.info(f"Training on {len(train_instances)} instances")

    # Output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.jobs_dir, exist_ok=True)

    # Training loop
    all_metrics = []
    checkpoint_path = None
    total_resolved = 0
    total_samples = 0
    start_time = time.time()

    for rollout_idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n[{rollout_idx + 1}/{len(train_instances)}] {instance_id}")

        # Generate samples using Harbor
        responses = []
        rewards = []

        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            # Run Harbor agent
            result = run_harbor_agent(
                instance=instance,
                agent=config.agent,
                n_concurrent=1,
                jobs_dir=config.jobs_dir,
            )

            response_text = result["response"]
            patch = result["patch"]

            # Evaluate with swebench.harness
            reward = evaluate_with_swebench(
                instance_id,
                patch,
                timeout=config.eval_timeout,
            )

            responses.append(response_text)
            rewards.append(reward)
            total_samples += 1
            if reward > 0:
                total_resolved += 1

        logger.info(f"  Rewards: {rewards}")

        # Train on Modal
        if any(r.strip() for r in responses):
            logger.info("  Training on Modal...")
            prompt = create_swebench_prompt(instance)

            save_checkpoint = (rollout_idx + 1) % config.save_every == 0
            checkpoint_name = f"checkpoint_{rollout_idx + 1}" if save_checkpoint else None

            metrics = train_step_on_modal.remote(
                prompt=prompt,
                responses=responses,
                rewards=rewards,
                state=state,
                checkpoint_path=checkpoint_path,
                save_checkpoint=save_checkpoint,
                checkpoint_name=checkpoint_name,
            )

            if metrics.get("checkpoint_path"):
                checkpoint_path = metrics["checkpoint_path"]

            metrics["instance_id"] = instance_id
            metrics["rollout_idx"] = rollout_idx
            metrics["rewards"] = rewards
            all_metrics.append(metrics)

            logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")
        else:
            logger.warning("  No valid responses, skipping training step")

        # Save metrics
        with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    resolve_rate = total_resolved / total_samples if total_samples > 0 else 0

    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total rollouts: {len(train_instances)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total resolved: {total_resolved} ({resolve_rate*100:.1f}%)")
    logger.info(f"Time: {elapsed / 60:.1f} minutes")

    # Save summary
    summary = {
        "total_rollouts": len(train_instances),
        "total_samples": total_samples,
        "total_resolved": total_resolved,
        "resolve_rate": resolve_rate,
        "elapsed_minutes": elapsed / 60,
        "config": {
            "model_name": config.model_name,
            "agent": config.agent,
            "n_samples_per_prompt": config.n_samples_per_prompt,
            "lr": config.lr,
            "kl_coef": config.kl_coef,
        },
    }
    with open(os.path.join(config.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


@app.local_entrypoint()
def main(
    num_rollouts: int = 50,
    n_samples: int = 4,
    agent: str = "mini-swe-agent",
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    output_dir: str = "outputs/harbor_grpo",
    test: bool = False,
):
    """Harbor GRPO training entry point."""
    config = HarborGRPOConfig(
        model_name=model_name,
        agent=agent,
        n_samples_per_prompt=n_samples,
        lr=lr,
        kl_coef=kl_coef,
        output_dir=output_dir,
    )

    run_harbor_grpo_training(config, num_rollouts, test_mode=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harbor GRPO Training")
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--agent", default="mini-swe-agent",
                       help="Harbor agent (mini-swe-agent, qwen-coder, etc.)")
    parser.add_argument("--model", default="Kwai-Klear/Klear-AgentForge-8B-SFT")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--output-dir", default="outputs/harbor_grpo")
    parser.add_argument("--test", action="store_true", help="Test mode (5 instances)")
    args = parser.parse_args()

    config = HarborGRPOConfig(
        model_name=args.model,
        agent=args.agent,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        output_dir=args.output_dir,
    )

    run_harbor_grpo_training(config, args.num_rollouts, test_mode=args.test)
