#!/usr/bin/env python
"""
Agentic GRPO Trainer - Modal GPU Version

This trainer runs training on Modal A100 GPUs with:
- vLLM server on Modal for inference
- Local Docker for agent tool execution
- Local Docker for swebench.harness evaluation

Uses shared code from agentic_grpo_core.py for identical behavior.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Modal Cloud (A100)                        │
    │  ┌──────────────────┐    ┌──────────────────────────────┐   │
    │  │ vLLM Server      │    │ Training Function            │   │
    │  │ (deployed)       │    │ - Model loading              │   │
    │  │                  │    │ - GRPO updates               │   │
    │  └──────────────────┘    │ - Gradient computation       │   │
    │                          └──────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                   │                          │
                   ▼                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    Local Machine                             │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Docker Containers                        │   │
    │  │  - Agent tool execution (via API calls to Modal)      │   │
    │  │  - swebench.harness evaluation                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Deploy vLLM server first
    modal deploy examples/grpo/modal_vllm.py

    # Run training
    modal run examples/grpo/agentic_grpo_modal.py --test
    modal run examples/grpo/agentic_grpo_modal.py --num-rollouts 50 --n-samples 4
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import modal

# Modal app
app = modal.App("agentic-grpo-trainer")

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
        "requests",
    )
    # Install SLiME for ppo_utils
    .pip_install("slime @ git+https://github.com/your-org/slime.git")
)

# Volumes
model_cache = modal.Volume.from_name("agentic-grpo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("agentic-grpo-outputs", create_if_missing=True)

# Secrets
hf_secret = modal.Secret.from_name("hf-token-swe")


# ==============================================================================
# Modal Training Function
# ==============================================================================

@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=7200,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/outputs": output_volume,
    },
)
def train_grpo_step_modal(
    prompt: str,
    rollouts_data: list[dict],
    model_name: str,
    checkpoint_path: str = None,
    save_checkpoint: bool = False,
    checkpoint_name: str = None,
    config_dict: dict = None,
) -> dict:
    """
    Perform GRPO training step on Modal GPU.

    Args:
        prompt: Training prompt
        rollouts_data: List of serialized rollout data
        model_name: Model name
        checkpoint_path: Previous checkpoint path
        save_checkpoint: Whether to save checkpoint
        checkpoint_name: Name for checkpoint
        config_dict: Configuration dict

    Returns:
        Training metrics
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Import SLiME's GRPO utilities
    from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss

    device = torch.device("cuda")
    config = config_dict or {}

    print(f"Training on {len(rollouts_data)} rollouts")

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

        if config.get("use_lora", True):
            lora_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_r", 16) * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    model.gradient_checkpointing_enable()

    # Load reference model
    print("Loading reference model...")
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
        lr=config.get("lr", 1e-6),
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )

    # Compute advantages
    rewards = [r["reward"] for r in rollouts_data]
    mean_reward = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = max(var ** 0.5, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        std_reward = 0.0
        advantages = [r - mean_reward for r in rewards]

    print(f"Rewards: {rewards}, Mean: {mean_reward:.3f}, Std: {std_reward:.3f}")

    # Training
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    n_valid = 0

    # Config values
    eps_clip = config.get("eps_clip", 0.2)
    eps_clip_high = config.get("eps_clip_high", 0.28)
    kl_coef = config.get("kl_coef", 0.001)
    kl_loss_type = config.get("kl_loss_type", "low_var_kl")
    max_response_tokens = config.get("max_response_tokens", 4096)

    # Tokenize prompt
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    prompt_ids = prompt_inputs["input_ids"][0].to(device)

    for rollout_data, advantage in zip(rollouts_data, advantages):
        model_token_ids = rollout_data["token_ids"]
        model_logprobs = rollout_data["logprobs"]

        if len(model_token_ids) == 0:
            continue

        # Truncate
        if len(model_token_ids) > max_response_tokens:
            print(f"Truncating from {len(model_token_ids)} to {max_response_tokens}")
            model_token_ids = model_token_ids[:max_response_tokens]
            model_logprobs = model_logprobs[:max_response_tokens]

        response_ids = torch.tensor(model_token_ids, device=device)
        old_logprobs = torch.tensor(model_logprobs, device=device, dtype=torch.float32)

        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        prompt_len = len(prompt_ids)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            policy_outputs = model(full_ids, return_dict=True)
            with torch.no_grad():
                ref_outputs = ref_model(full_ids, return_dict=True)

        response_logits = policy_outputs.logits[0, prompt_len - 1:-1]
        ref_logits = ref_outputs.logits[0, prompt_len - 1:-1]

        policy_log_probs = F.log_softmax(response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        policy_token_log_probs = policy_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        if len(old_logprobs) != len(policy_token_log_probs):
            min_len = min(len(old_logprobs), len(policy_token_log_probs))
            old_logprobs = old_logprobs[:min_len]
            policy_token_log_probs = policy_token_log_probs[:min_len]
            ref_token_log_probs = ref_token_log_probs[:min_len]

        ppo_kl = old_logprobs - policy_token_log_probs
        advantages_tensor = torch.full_like(policy_token_log_probs, advantage)

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

        loss = policy_loss + kl_coef * kl_loss
        loss.backward()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()
        n_valid += 1

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

@app.local_entrypoint()
def main(
    num_rollouts: int = 50,
    n_samples: int = 4,
    vllm_url: str = "https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run",
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    output_dir: str = "outputs/agentic_grpo_modal",
    save_every: int = 10,
    test: bool = False,
):
    """
    Run agentic GRPO training with Modal GPU.

    Rollouts run locally (Docker), training runs on Modal A100.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from examples.grpo.agentic_grpo_core import (
        AgenticGRPOConfig,
        run_agent_rollout,
        evaluate_patch,
        load_training_instances,
        create_training_prompt,
    )
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Agentic GRPO Training - Modal GPU")
    logger.info("  - Rollouts: Local Docker")
    logger.info("  - Training: Modal A100")
    logger.info("  - SLiME ppo_utils with Search-R1 parameters")
    logger.info("=" * 70)

    config = AgenticGRPOConfig(
        model_name=model_name,
        vllm_url=vllm_url,
        n_samples_per_prompt=n_samples,
        lr=lr,
        kl_coef=kl_coef,
    )

    config_dict = {
        "lr": config.lr,
        "kl_coef": config.kl_coef,
        "kl_loss_type": config.kl_loss_type,
        "eps_clip": config.eps_clip,
        "eps_clip_high": config.eps_clip_high,
        "use_lora": config.use_lora,
        "lora_r": config.lora_r,
        "max_response_tokens": config.max_response_tokens,
    }

    # Load tokenizer locally for rollouts
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training instances
    train_instances = load_training_instances(
        num_instances=num_rollouts,
        test_mode=test,
    )

    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    checkpoint_path = None
    total_resolved = 0
    total_samples = 0
    start_time = time.time()

    for idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n[{idx + 1}/{len(train_instances)}] {instance_id}")

        # Generate rollouts locally
        rollouts = []
        rollouts_data = []

        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            rollout = run_agent_rollout(
                instance_id=instance_id,
                problem_statement=instance["problem_statement"],
                config=config,
                tokenizer=tokenizer,
            )

            # Evaluate locally
            rollout.reward = evaluate_patch(
                instance_id,
                rollout.patch,
                timeout=config.eval_timeout,
            )

            rollouts.append(rollout)
            total_samples += 1
            if rollout.reward > 0:
                total_resolved += 1

            # Serialize for Modal
            rollouts_data.append({
                "token_ids": rollout.get_model_token_ids(),
                "logprobs": rollout.get_model_logprobs(),
                "reward": rollout.reward,
            })

            logger.info(f"    Turns: {len(rollout.turns)}, Tokens: {rollout.total_model_tokens()}, Reward: {rollout.reward}")

        # Train on Modal
        logger.info("  Training on Modal A100...")
        prompt = create_training_prompt(instance)

        save_checkpoint = (idx + 1) % save_every == 0
        checkpoint_name = f"checkpoint_{idx + 1}" if save_checkpoint else None

        metrics = train_grpo_step_modal.remote(
            prompt=prompt,
            rollouts_data=rollouts_data,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            checkpoint_name=checkpoint_name,
            config_dict=config_dict,
        )

        if metrics.get("checkpoint_path"):
            checkpoint_path = metrics["checkpoint_path"]

        metrics["instance_id"] = instance_id
        all_metrics.append(metrics)

        logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")

        # Save metrics locally
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
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
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
