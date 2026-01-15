#!/usr/bin/env python
"""
Hybrid GRPO Trainer: Modal GPU + Local swebench.harness Evaluation.

Architecture:
- Modal A100 GPU: Model loading, inference, weight updates
- Local Machine: swebench.harness evaluation (Docker, no GPU needed)

This is the RECOMMENDED approach when:
- You don't have local GPU
- You want accurate swebench.harness evaluation (not heuristics)

How it works:
1. Modal generates responses and extracts patches
2. Patches are sent back to local machine
3. Local machine evaluates with swebench.harness (Docker)
4. Rewards are sent back to Modal
5. Modal computes GRPO loss and updates weights

Uses IDENTICAL GRPO implementation as local_gpu_grpo_trainer.py via grpo_core.py.

Usage:
    # Run training (this script runs locally, calls Modal for GPU work)
    python examples/harbor/hybrid_grpo_trainer.py \
        --num-rollouts 50 \
        --n-samples 4
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import modal

# NOTE: grpo_core imports are done inside run_hybrid_training()
# because Modal functions don't have access to local files.
# Modal functions use SLiME's ppo_utils.py directly.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Modal app
app = modal.App("harbor-hybrid-grpo")

# Modal image with training dependencies (including SLiME for ppo_utils)
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
    # Install SLiME for ppo_utils.py (GRPO loss computation)
    .pip_install("git+https://github.com/THUDM/slime.git")
)

# Volumes for caching
model_cache = modal.Volume.from_name("harbor-hybrid-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("harbor-hybrid-outputs", create_if_missing=True)

# Secrets
hf_secret = modal.Secret.from_name("hf-token-swe")


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
def generate_samples_on_modal(
    instance_data: dict,
    state: dict,
    checkpoint_path: str = None,
) -> list[dict]:
    """
    Generate samples on Modal GPU.

    Args:
        instance_data: {"instance_id": str, "prompt": str}
        state: Training state dict with GRPO config
        checkpoint_path: Path to checkpoint (None = base model)

    Returns:
        List of {"response": str, "patch": str, "token_ids": list, "logprobs": list}
    """
    import re
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_name = state["model_name"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (from checkpoint or base)
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

    model.eval()

    # Generate samples
    prompt = instance_data["prompt"]
    n_samples = state.get("n_samples_per_prompt", 4)
    temperature = state.get("temperature", 1.0)
    max_new_tokens = state.get("max_new_tokens", 2048)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # Local extract_patch function (grpo_core not available on Modal)
    def _extract_patch(response: str) -> str:
        diff_pattern = r'```(?:diff)?\n((?:---|\+\+\+|@@|[-+ ].*\n?)+)```'
        match = re.search(diff_pattern, response, re.MULTILINE)
        if match:
            return match.group(1).strip()
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

    samples = []
    for i in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract response
        response_ids = outputs.sequences[0, prompt_length:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Extract logprobs
        logprobs = []
        for j, score in enumerate(outputs.scores):
            if j < len(response_ids):
                probs = F.log_softmax(score[0], dim=-1)
                token_id = response_ids[j].item()
                logprobs.append(probs[token_id].item())

        # Extract patch
        patch = _extract_patch(response_text)

        samples.append({
            "response": response_text,
            "patch": patch,
            "token_ids": response_ids.tolist(),
            "logprobs": logprobs,
        })

        print(f"  Sample {i+1}: {len(response_ids)} tokens, patch_len={len(patch)}")

    return samples


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
    samples: list[dict],
    rewards: list[float],
    state: dict,
    checkpoint_path: str = None,
    save_checkpoint: bool = False,
    checkpoint_name: str = None,
) -> dict:
    """
    Perform GRPO training step on Modal GPU.

    Uses the SAME GRPO formulas as local_gpu_grpo_trainer.py:
    - Group-relative advantages: (r_i - mean) / std
    - PPO-style clipping with DAPO asymmetric bounds [0.8, 1.28]
    - Low-variance KL: 0.5 * (ratio - 1)^2

    Args:
        prompt: The prompt used for generation
        samples: List of generated samples with token_ids and logprobs
        rewards: List of rewards from swebench.harness evaluation
        state: Training state dict with GRPO config
        checkpoint_path: Path to load checkpoint from
        save_checkpoint: Whether to save checkpoint after this step
        checkpoint_name: Name for saved checkpoint

    Returns:
        {"loss": float, "policy_loss": float, "kl_loss": float, "checkpoint_path": str}
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Import SLiME's GRPO utilities (installed in Modal image)
    from slime.utils.ppo_utils import (
        compute_approx_kl as slime_compute_kl,
        compute_policy_loss as slime_compute_policy_loss,
    )

    device = torch.device("cuda")
    model_name = state["model_name"]

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
    # GRPO Implementation (IDENTICAL to local_gpu_grpo_trainer.py)
    # =========================================================================

    # Compute GRPO group-relative advantages
    mean_reward = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = max(variance ** 0.5, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        std_reward = 0.0
        advantages = [r - mean_reward for r in rewards]

    print(f"Rewards: {rewards}")
    print(f"Mean: {mean_reward:.3f}, Std: {std_reward:.3f}")
    print(f"Advantages: {[f'{a:.3f}' for a in advantages]}")

    # Tokenize prompt
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)
    prompt_length = prompt_inputs["input_ids"].shape[1]

    # Training
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    n_valid = 0

    # GRPO hyperparameters (Search-R1)
    eps_clip = state.get("eps_clip", 0.2)
    eps_clip_high = state.get("eps_clip_high", 0.28)
    kl_coef = state.get("kl_coef", 0.001)

    for sample, reward, advantage in zip(samples, rewards, advantages):
        token_ids = sample["token_ids"]
        old_logprobs = sample["logprobs"]

        if len(token_ids) == 0 or len(old_logprobs) == 0:
            continue

        # Build full sequence
        response_ids = torch.tensor(token_ids, device=device)
        full_ids = torch.cat([
            prompt_inputs["input_ids"][0],
            response_ids
        ]).unsqueeze(0)

        # Forward through policy model
        policy_outputs = model(full_ids, return_dict=True)
        policy_logits = policy_outputs.logits

        # Forward through reference model
        with torch.no_grad():
            ref_outputs = ref_model(full_ids, return_dict=True)
            ref_logits = ref_outputs.logits

        # Compute log probs
        response_start = prompt_length - 1
        response_end = full_ids.shape[1] - 1

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

        old_log_probs_tensor = torch.tensor(old_logprobs, device=device)

        # Align lengths
        min_len = min(len(policy_token_log_probs), len(old_log_probs_tensor))
        policy_token_log_probs = policy_token_log_probs[:min_len]
        ref_token_log_probs = ref_token_log_probs[:min_len]
        old_log_probs_tensor = old_log_probs_tensor[:min_len]

        # Policy loss using SLiME's implementation (PPO-style with DAPO clipping)
        # Expand advantage to match log_probs shape
        advantages_tensor = torch.full_like(policy_token_log_probs, advantage)

        # SLiME's compute_policy_loss expects ppo_kl = log_probs_old - log_probs
        ppo_kl = old_log_probs_tensor - policy_token_log_probs
        pg_losses, clipfrac = slime_compute_policy_loss(
            ppo_kl=ppo_kl,
            advantages=advantages_tensor,
            eps_clip=eps_clip,
            eps_clip_high=eps_clip_high,
            eps_clip_c=None,  # No dual-clip
        )
        policy_loss = pg_losses.mean()

        # KL loss using SLiME's implementation (low-variance KL)
        kl = slime_compute_kl(
            log_probs=policy_token_log_probs,
            log_probs_base=ref_token_log_probs,
            kl_loss_type="low_var_kl",
        )
        kl_loss = kl.mean()

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


def run_hybrid_training(
    num_rollouts: int = 50,
    n_samples: int = 4,
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    temperature: float = 1.0,
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    output_dir: str = "outputs/hybrid_grpo",
    save_every: int = 10,
    eval_timeout: int = 300,
):
    """
    Main training loop that coordinates Modal GPU and local evaluation.

    Modal GPU: inference + weight updates
    Local: swebench.harness evaluation (Docker)
    """
    from datasets import load_dataset

    # Import grpo_core for local evaluation (Modal functions have their own implementations)
    sys.path.insert(0, str(Path(__file__).parent))
    from grpo_core import (
        evaluate_with_swebench,
        create_swebench_prompt,
    )

    logger.info("=" * 70)
    logger.info("Hybrid GRPO Training")
    logger.info("  Modal A100: Model inference + weight updates")
    logger.info("  Local: swebench.harness evaluation (Docker)")
    logger.info("=" * 70)

    # Training state (Search-R1 GRPO hyperparameters)
    state = {
        "model_name": model_name,
        "lr": lr,
        "kl_coef": kl_coef,
        "temperature": temperature,
        "n_samples_per_prompt": n_samples,
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "use_lora": True,
        "lora_r": 16,
        "max_new_tokens": 2048,
    }

    logger.info(f"\nSearch-R1 GRPO Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Rollouts: {num_rollouts}")
    logger.info(f"  Samples per prompt: {n_samples}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  KL coefficient: {kl_coef}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  eps_clip: [0.8, 1.28] (DAPO asymmetric)")
    logger.info(f"  Evaluation: swebench.harness ONLY (no heuristics)")

    # Load data
    logger.info("\nLoading SWE-bench data...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
    django_instances = [x for x in ds if x["repo"] == "django/django"]
    train_instances = django_instances[:num_rollouts]
    logger.info(f"Training on {len(train_instances)} Django instances")

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    all_metrics = []
    checkpoint_path = None
    total_resolved = 0
    total_samples = 0
    start_time = time.time()

    for rollout_idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        prompt = create_swebench_prompt(instance)

        logger.info(f"\n[{rollout_idx + 1}/{len(train_instances)}] {instance_id}")

        # Step 1: Generate samples on Modal
        logger.info("  Generating samples on Modal...")
        instance_data = {"instance_id": instance_id, "prompt": prompt}

        samples = generate_samples_on_modal.remote(
            instance_data=instance_data,
            state=state,
            checkpoint_path=checkpoint_path,
        )

        logger.info(f"  Generated {len(samples)} samples")

        # Step 2: Evaluate locally with swebench.harness (NO HEURISTICS)
        logger.info("  Evaluating with swebench.harness (local Docker)...")
        rewards = []
        for i, sample in enumerate(samples):
            patch = sample["patch"]
            # Use shared evaluate_with_swebench from grpo_core.py
            reward = evaluate_with_swebench(instance_id, patch, eval_timeout)
            rewards.append(reward)
            if reward > 0:
                total_resolved += 1
            total_samples += 1

        logger.info(f"  Rewards: {rewards}")

        # Step 3: Train on Modal with rewards
        logger.info("  Training on Modal...")
        save_checkpoint = (rollout_idx + 1) % save_every == 0
        checkpoint_name = f"checkpoint_{rollout_idx + 1}" if save_checkpoint else None

        metrics = train_step_on_modal.remote(
            prompt=prompt,
            samples=samples,
            rewards=rewards,
            state=state,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            checkpoint_name=checkpoint_name,
        )

        if metrics.get("checkpoint_path"):
            checkpoint_path = metrics["checkpoint_path"]

        # Log metrics
        metrics["instance_id"] = instance_id
        metrics["rollout_idx"] = rollout_idx
        metrics["rewards"] = rewards
        all_metrics.append(metrics)

        logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")
        logger.info(f"  Reward: mean={metrics['mean_reward']:.3f}, std={metrics['std_reward']:.3f}")

        # Save metrics locally
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
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

    # Save final summary
    summary = {
        "total_rollouts": len(train_instances),
        "total_samples": total_samples,
        "total_resolved": total_resolved,
        "resolve_rate": resolve_rate,
        "elapsed_minutes": elapsed / 60,
        "config": state,
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


@app.local_entrypoint()
def main(
    num_rollouts: int = 50,
    n_samples: int = 4,
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    temperature: float = 1.0,
    output_dir: str = "outputs/hybrid_grpo",
    save_every: int = 10,
    eval_timeout: int = 300,
):
    """
    Hybrid GRPO training: Modal GPU + local swebench.harness.
    """
    run_hybrid_training(
        num_rollouts=num_rollouts,
        n_samples=n_samples,
        lr=lr,
        kl_coef=kl_coef,
        temperature=temperature,
        output_dir=output_dir,
        save_every=save_every,
        eval_timeout=eval_timeout,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid GRPO Training")
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid_grpo")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-timeout", type=int, default=300)
    args = parser.parse_args()

    run_hybrid_training(
        num_rollouts=args.num_rollouts,
        n_samples=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
        output_dir=args.output_dir,
        save_every=args.save_every,
        eval_timeout=args.eval_timeout,
    )
