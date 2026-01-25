#!/usr/bin/env python
"""
Harbor GRPO Trainer - Modal GPU Version

This trainer runs on Modal A100 GPUs with:
- Harbor CLI for agent rollouts (locally)
- Modal for GRPO training (GPU)
- Local Docker for swebench.harness evaluation

Uses shared code from harbor_core.py.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Modal Cloud (A100)                        │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Training Function                                     │   │
    │  │ - Model loading                                       │   │
    │  │ - GRPO updates using SLiME ppo_utils                  │   │
    │  │ - Gradient computation                                │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                              ▲
                              │ responses + rewards
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                    Local Machine                             │
    │  ┌──────────────────┐    ┌──────────────────────────────┐   │
    │  │ Harbor Agent     │    │ swebench.harness             │   │
    │  │ (mini-swe-agent) │    │ (evaluation)                 │   │
    │  └──────────────────┘    └──────────────────────────────┘   │
    │           │                           │                     │
    │           ▼                           ▼                     │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Docker Containers                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Test mode (5 instances)
    modal run examples/harbor/harbor_grpo_modal.py --test

    # Full training (201 Django instances)
    modal run examples/harbor/harbor_grpo_modal.py --num-rollouts 201

    # Python entrypoint
    python examples/harbor/harbor_grpo_modal.py --test
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import modal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Modal Configuration
# ==============================================================================

app = modal.App("harbor-grpo-modal")

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
    # Install SLiME for ppo_utils
    .run_commands(
        "pip install git+https://github.com/OpenRLHF/SLiME.git || pip install slime || true"
    )
)

# Volumes
model_cache = modal.Volume.from_name("harbor-grpo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("harbor-grpo-outputs", create_if_missing=True)

# HF secret
hf_secret = modal.Secret.from_name("hf-token-swe")


# ==============================================================================
# Modal Training Function
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
    config_dict: dict,
    checkpoint_path: str = None,
    save_checkpoint: bool = False,
    checkpoint_name: str = None,
) -> dict:
    """
    Perform GRPO training step on Modal GPU using SLiME's ppo_utils.

    Key Insight:
    Responses come from Harbor agents (text only, no log probs).
    We compute log probs at training time via forward pass.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_name = config_dict["model_name"]

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

        if config_dict.get("use_lora", True):
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config_dict.get("lora_r", 16),
                lora_alpha=config_dict.get("lora_r", 16) * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    model.gradient_checkpointing_enable()

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
        lr=config_dict.get("lr", 1e-6),
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )

    # GRPO hyperparameters (Search-R1)
    eps_clip = config_dict.get("eps_clip", 0.2)
    eps_clip_high = config_dict.get("eps_clip_high", 0.28)
    kl_coef = config_dict.get("kl_coef", 0.001)
    kl_loss_type = config_dict.get("kl_loss_type", "low_var_kl")
    max_response_tokens = config_dict.get("max_response_tokens", 4096)

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

    # Try to import SLiME's ppo_utils
    try:
        from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
        use_slime = True
        print("Using SLiME's ppo_utils")
    except ImportError:
        use_slime = False
        print("SLiME not available, using inline GRPO")

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

        # Tokenize
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

        # Truncate if needed
        if response_length > max_response_tokens:
            print(f"Truncating from {response_length} to {max_response_tokens}")
            full_ids = full_ids[:, :prompt_length + max_response_tokens]
            response_length = max_response_tokens

        # Forward passes
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            policy_outputs = model(full_ids, return_dict=True)
            with torch.no_grad():
                ref_outputs = ref_model(full_ids, return_dict=True)

        # Compute log probs
        response_logits = policy_outputs.logits[0, prompt_length - 1:-1]
        ref_logits = ref_outputs.logits[0, prompt_length - 1:-1]
        response_ids = full_ids[0, prompt_length:]

        policy_log_probs = F.log_softmax(response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        policy_token_log_probs = policy_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        old_log_probs = policy_token_log_probs.detach()
        ppo_kl = old_log_probs - policy_token_log_probs
        advantages_tensor = torch.full_like(policy_token_log_probs, advantage)

        if use_slime:
            # Use SLiME's functions
            pg_losses, clipfrac = compute_policy_loss(
                ppo_kl=ppo_kl,
                advantages=advantages_tensor,
                eps_clip=eps_clip,
                eps_clip_high=eps_clip_high,
                eps_clip_c=None,
            )
            policy_loss = pg_losses.mean()

            kl = compute_approx_kl(
                log_probs=policy_token_log_probs,
                log_probs_base=ref_token_log_probs,
                kl_loss_type=kl_loss_type,
            )
            kl_loss = kl.mean()
        else:
            # Inline GRPO
            ratio = torch.exp(policy_token_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)
            policy_loss = torch.max(
                -advantages_tensor * ratio,
                -advantages_tensor * clipped_ratio
            ).mean()

            # Low-variance KL
            kl_ratio = torch.exp(policy_token_log_probs - ref_token_log_probs)
            kl_loss = 0.5 * ((kl_ratio - 1) ** 2).mean()

        loss = policy_loss + kl_coef * kl_loss
        loss.backward()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()
        n_valid += 1

    # Update
    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"Updated weights with {n_valid} samples")

    # Save checkpoint
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
# Local Entrypoint
# ==============================================================================

def parse_harbor_reward(job_dir: Path) -> float:
    """Parse reward from Harbor job directory (reward.txt)."""
    for reward_file in job_dir.glob("**/reward.txt"):
        try:
            reward_text = reward_file.read_text().strip()
            return 1.0 if reward_text == "1" else -1.0
        except Exception:
            pass
    return -1.0


@app.local_entrypoint()
def main(
    num_rollouts: int = 50,
    n_samples: int = 4,
    agent: str = "qwen-coder",
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    output_dir: str = "outputs/harbor_grpo_modal",
    jobs_dir: str = "jobs",
    save_every: int = 10,
    test: bool = False,
    data_source: str = "swebench",
    c2bug_dataset: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    c2bug_docker_image: str = "swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
    env: str = "docker",
    daytona_target: str = None,
    skip_training: bool = False,
    eval_method: str = "harbor",
):
    """
    Run Harbor GRPO training with Modal GPU.

    Rollouts run locally (Harbor + Docker/Daytona), training runs on Modal A100.
    Supports both SWE-bench and C2Bug data sources.

    Args:
        eval_method: "harbor" (default, uses Harbor's built-in verifier) or "swebench" (uses swebench.harness)
    """
    import subprocess

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from examples.harbor.harbor_core import (
        HarborGRPOConfig,
        run_harbor_agent,
        evaluate_with_swebench,
        load_training_instances,
        create_swebench_prompt,
    )

    logger.info("=" * 70)
    logger.info("Harbor GRPO Training - Modal GPU")
    logger.info(f"  - Data source: {data_source}")
    logger.info(f"  - Agent: {agent}")
    logger.info(f"  - Environment: {env}")
    logger.info(f"  - Eval method: {eval_method}")
    logger.info("  - Modal A100: GRPO training")
    logger.info("=" * 70)

    # Update output dir for c2bug
    if output_dir == "outputs/harbor_grpo_modal" and data_source == "c2bug":
        output_dir = "outputs/harbor_grpo_modal_c2bug"

    config = HarborGRPOConfig(
        model_name=model_name,
        agent=agent,
        env=env,
        n_samples_per_prompt=n_samples,
        lr=lr,
        kl_coef=kl_coef,
        output_dir=output_dir,
        jobs_dir=jobs_dir,
        save_every=save_every,
    )

    config_dict = {
        "model_name": config.model_name,
        "lr": config.lr,
        "kl_coef": config.kl_coef,
        "kl_loss_type": config.kl_loss_type,
        "eps_clip": config.eps_clip,
        "eps_clip_high": config.eps_clip_high,
        "use_lora": config.use_lora,
        "lora_r": config.lora_r,
        "max_response_tokens": 4096,
    }

    # Load training instances based on data source
    if data_source == "c2bug":
        from examples.harbor.c2bug_adapter import (
            load_c2bug_from_hf,
            C2BugToHarbor,
            C2BugLoader,
        )

        if test:
            num_rollouts = min(5, num_rollouts)

        logger.info(f"Loading c2bug data from {c2bug_dataset}...")
        collection = load_c2bug_from_hf(c2bug_dataset)
        run_meta_override = {"docker_image": c2bug_docker_image, "workdir": "/testbed"}

        task_root = Path("/tmp/c2bug_harbor_tasks")
        converter = C2BugToHarbor(
            collection_source=collection,
            task_root=task_root,
            max_timeout_sec=3000.0,
            run_meta_override=run_meta_override,
        )
        converter.generate_many(limit=num_rollouts, overwrite=True)

        train_instances = []
        loader = C2BugLoader(collection)
        loader.apply_run_meta(run_meta_override)
        for record in list(loader.iter_records())[:num_rollouts]:
            task_name = record.task_uid or record.instance_id
            task_dir = task_root / task_name
            if task_dir.exists():
                train_instances.append({
                    "instance_id": task_name,
                    "task_dir": str(task_dir),
                    "problem_statement": record.issue_text,
                    "repo": record.repo,
                })
        logger.info(f"Loaded {len(train_instances)} c2bug instances")
    else:
        train_instances = load_training_instances(
            num_instances=num_rollouts,
            test_mode=test,
        )

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.jobs_dir, exist_ok=True)

    all_metrics = []
    checkpoint_path = None
    total_resolved = 0
    total_samples = 0
    start_time = time.time()

    for idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n[{idx + 1}/{len(train_instances)}] {instance_id}")

        # Generate rollouts locally using Harbor
        responses = []
        rewards = []

        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            if data_source == "c2bug":
                # C2Bug: run Harbor with task_dir, reward from verifier
                task_dir = instance["task_dir"]
                job_name = f"c2bug-{instance_id.replace('/', '_').replace('__', '_')[:50]}-{int(time.time())}"
                cmd = [
                    "harbor", "run", "-p", task_dir,
                    "--env", env, "--agent", agent,
                    "--n-concurrent", "1", "--jobs-dir", jobs_dir,
                    "--job-name", job_name, "--export-traces",
                ]
                # Add Daytona target if using daytona env
                _daytona_target = daytona_target or os.environ.get("DAYTONA_TARGET")
                if env == "daytona" and _daytona_target:
                    cmd.extend(["--ek", f"target={_daytona_target}"])
                try:
                    # Pass environment explicitly for DAYTONA_API_KEY
                    proc_env = os.environ.copy()
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=proc_env)
                    if proc.returncode != 0:
                        logger.warning(f"Harbor returned {proc.returncode}: {proc.stderr[:200] if proc.stderr else ''}")
                    # Parse reward from job directory
                    job_dir = Path(jobs_dir) / job_name
                    reward = -1.0
                    for reward_file in job_dir.glob("**/reward.txt"):
                        try:
                            reward = 1.0 if reward_file.read_text().strip() == "1" else -1.0
                            break
                        except Exception:
                            pass
                    result = {"response": "", "status": "completed" if reward > 0 else "failed"}
                except Exception as e:
                    logger.error(f"Harbor error: {e}")
                    reward = -1.0
                    result = {"response": "", "status": "error"}
            else:
                # SWE-bench: run Harbor with dataset
                result = run_harbor_agent(instance=instance, config=config, timeout=1800)

                # Choose evaluation method
                if eval_method == "swebench":
                    # Use swebench.harness for evaluation
                    reward = evaluate_with_swebench(
                        instance_id=instance_id,
                        patch=result.get("patch", ""),
                        timeout=config.eval_timeout,
                    )
                else:
                    # Default: use Harbor's built-in verifier (reward.txt)
                    job_dir = result.get("job_dir")
                    if job_dir:
                        reward = parse_harbor_reward(Path(job_dir))
                    else:
                        # Fallback: try to find reward from jobs_dir
                        reward = -1.0
                        for job_path in Path(jobs_dir).glob(f"*{instance_id.replace('/', '_')[:30]}*"):
                            reward = parse_harbor_reward(job_path)
                            if reward > 0:
                                break

            responses.append(result.get("response", ""))
            rewards.append(reward)
            total_samples += 1
            if reward > 0:
                total_resolved += 1

            logger.info(f"    Status: {result['status']}, Reward: {reward}")

        # Skip training if requested
        if skip_training:
            logger.info(f"  Skipping training (skip_training=True, rewards={rewards})")
            all_metrics.append({"instance_id": instance_id, "rewards": rewards, "skipped": True})
            continue

        # Train on Modal
        logger.info("  Training on Modal A100...")
        if data_source == "c2bug":
            prompt = f"Fix this bug in {instance.get('repo', 'the codebase')}:\n\n{instance.get('problem_statement', '')[:4000]}"
        else:
            prompt = create_swebench_prompt(instance)

        save_checkpoint = (idx + 1) % config.save_every == 0
        checkpoint_name = f"checkpoint_{idx + 1}" if save_checkpoint else None

        metrics = train_step_on_modal.remote(
            prompt=prompt,
            responses=responses,
            rewards=rewards,
            config_dict=config_dict,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            checkpoint_name=checkpoint_name,
        )

        if metrics.get("checkpoint_path"):
            checkpoint_path = metrics["checkpoint_path"]

        metrics["instance_id"] = instance_id
        all_metrics.append(metrics)

        logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")

        # Save metrics locally
        with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    resolve_rate = total_resolved / total_samples if total_samples > 0 else 0

    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total instances: {len(train_instances)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total resolved: {total_resolved} ({resolve_rate * 100:.1f}%)")
    logger.info(f"Time: {elapsed / 60:.1f} minutes")

    summary = {
        "total_instances": len(train_instances),
        "total_samples": total_samples,
        "total_resolved": total_resolved,
        "resolve_rate": resolve_rate,
        "elapsed_minutes": elapsed / 60,
        "final_checkpoint": checkpoint_path,
    }
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
