"""
Modal GRPO Trainer with Proper Weight Updates.

This implements actual GRPO training on Modal with:
1. Search-R1 GRPO hyperparameters
2. Proper policy loss + KL loss computation
3. Group-relative advantage computation
4. Online rollouts using the training model
5. Weight updates after each batch

Architecture:
- Model loaded on Modal GPU (A100-80GB)
- Inference and training on same GPU
- Heuristic rewards initially (valid patch detection)
- Optional: swebench.harness evaluation for accurate rewards

Usage:
    # Run GRPO training
    modal run examples/harbor/modal_grpo_trainer.py --num-rollouts 50

    # Run with more samples per prompt
    modal run examples/harbor/modal_grpo_trainer.py --num-rollouts 50 --n-samples 5
"""

import modal
import os

app = modal.App("harbor-grpo-trainer")

# Training image with all dependencies
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

# Secrets
hf_secret = modal.Secret.from_name("hf-token-swe")


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=43200,  # 12 hours
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/outputs": output_volume,
    },
    cpu=8,
    memory=32768,
)
def train_grpo(
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    num_rollouts: int = 50,
    n_samples_per_prompt: int = 4,
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    gradient_accumulation_steps: int = 4,
    save_every: int = 10,
    output_dir: str = "/outputs/grpo_training",
    use_lora: bool = True,
    lora_r: int = 16,
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """
    Run GRPO training with proper weight updates.

    Search-R1 GRPO Algorithm:
    1. For each prompt, generate n_samples_per_prompt responses
    2. Compute rewards for each response
    3. Compute group-relative advantages: A_i = r_i - mean(r_group)
    4. Compute policy loss with PPO clipping
    5. Add KL penalty to prevent drift from reference
    6. Update weights

    Args:
        model_name: HuggingFace model to train
        num_rollouts: Number of prompts to train on
        n_samples_per_prompt: Group size for GRPO (samples per prompt)
        lr: Learning rate (Search-R1 default: 1e-6)
        kl_coef: KL divergence coefficient (Search-R1 default: 0.001)
        temperature: Sampling temperature (Search-R1 default: 1.0)
        max_new_tokens: Max tokens to generate per response
        gradient_accumulation_steps: Gradient accumulation steps
        save_every: Save checkpoint every N rollouts
        output_dir: Output directory for checkpoints
        use_lora: Whether to use LoRA for efficient training
        lora_r: LoRA rank
        push_to_hub: Whether to push final model to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID for pushing
    """
    import json
    import time
    from dataclasses import dataclass
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("GRPO Training with Proper Weight Updates")
    print("=" * 70)
    print(f"\nSearch-R1 GRPO Hyperparameters:")
    print(f"  Learning rate: {lr}")
    print(f"  KL coefficient: {kl_coef}")
    print(f"  Temperature: {temperature}")
    print(f"  Samples per prompt: {n_samples_per_prompt}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    print(f"Loading model: {model_name}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        print(f"\nApplying LoRA with r={lora_r}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Create reference model for KL computation
    print("Creating reference model for KL...")
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )

    # Load SWE-bench data
    print("\nLoading SWE-bench data...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
    django_instances = [x for x in ds if x["repo"] == "django/django"]
    print(f"Found {len(django_instances)} Django instances")

    train_instances = django_instances[:min(num_rollouts, len(django_instances))]
    print(f"Using {len(train_instances)} instances for training")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prompt template
    def create_prompt(instance):
        return f"""You are an expert software engineer. Fix this bug and provide a git diff patch.

## Repository: {instance["repo"]}
## Problem Statement
{instance["problem_statement"][:2000]}

## Instructions
Provide ONLY a git diff patch that fixes this issue. Format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
-old line
+new line
```

Your patch:"""

    # Training metrics
    all_metrics = []
    total_samples = 0
    start_time = time.time()

    # ==================== GRPO Training Loop ====================
    print("\n" + "=" * 70)
    print("Starting GRPO Training Loop")
    print("=" * 70)

    for rollout_idx, instance in enumerate(tqdm(train_instances, desc="Training")):
        instance_id = instance["instance_id"]
        prompt = create_prompt(instance)

        print(f"\n[{rollout_idx + 1}/{len(train_instances)}] {instance_id}")

        # Tokenize prompt
        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        # ==================== Generate Group of Samples ====================
        group_responses = []
        group_logprobs = []
        group_rewards = []

        model.eval()
        with torch.no_grad():
            for sample_idx in range(n_samples_per_prompt):
                # Generate response
                outputs = model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Extract response tokens
                response_ids = outputs.sequences[0, prompt_length:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

                # Compute log probabilities for the response
                # scores is a tuple of logits for each generated token
                logprobs = []
                for i, score in enumerate(outputs.scores):
                    probs = F.log_softmax(score[0], dim=-1)
                    token_id = response_ids[i].item()
                    logprobs.append(probs[token_id].item())

                # Compute reward (heuristic: valid patch detection)
                reward = compute_heuristic_reward(response_text)

                group_responses.append({
                    "text": response_text,
                    "token_ids": response_ids.tolist(),
                    "logprobs": logprobs,
                })
                group_logprobs.append(logprobs)
                group_rewards.append(reward)

                print(f"  Sample {sample_idx + 1}: reward={reward:.2f}, tokens={len(response_ids)}")

        # ==================== Compute GRPO Advantages ====================
        mean_reward = sum(group_rewards) / len(group_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_reward = max(std_reward, 1e-8)  # Avoid division by zero

        advantages = [(r - mean_reward) / std_reward for r in group_rewards]

        print(f"  Group: mean_reward={mean_reward:.3f}, std={std_reward:.3f}")

        # ==================== Compute GRPO Loss and Update ====================
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_valid_samples = 0

        for sample_idx, (response, advantage, old_logprobs) in enumerate(
            zip(group_responses, advantages, group_logprobs)
        ):
            if len(response["token_ids"]) == 0:
                continue

            # Prepare full sequence: prompt + response
            full_ids = torch.cat([
                prompt_inputs["input_ids"][0],
                torch.tensor(response["token_ids"], device=device)
            ]).unsqueeze(0)

            # Forward pass through policy model
            policy_outputs = model(full_ids, return_dict=True)
            policy_logits = policy_outputs.logits

            # Forward pass through reference model
            with torch.no_grad():
                ref_outputs = ref_model(full_ids, return_dict=True)
                ref_logits = ref_outputs.logits

            # Compute log probabilities for response tokens
            # Shift by 1 for next-token prediction
            response_start = prompt_length - 1
            response_end = full_ids.shape[1] - 1

            policy_log_probs = F.log_softmax(policy_logits[0, response_start:response_end], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[0, response_start:response_end], dim=-1)

            # Get log probs of actual tokens
            response_token_ids = torch.tensor(response["token_ids"], device=device)
            policy_token_log_probs = policy_log_probs.gather(
                -1, response_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_log_probs = ref_log_probs.gather(
                -1, response_token_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Old log probs from generation
            old_log_probs = torch.tensor(old_logprobs, device=device)

            # Ensure same length
            min_len = min(len(policy_token_log_probs), len(old_log_probs))
            policy_token_log_probs = policy_token_log_probs[:min_len]
            ref_token_log_probs = ref_token_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]

            # ==================== Policy Loss (PPO-style) ====================
            # Importance sampling ratio
            ratio = torch.exp(policy_token_log_probs - old_log_probs)

            # Clipped ratio
            eps_clip = 0.2
            eps_clip_high = 0.28  # DAPO-style asymmetric clipping
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)

            # Policy loss
            advantage_tensor = torch.tensor(advantage, device=device)
            loss1 = -advantage_tensor * ratio
            loss2 = -advantage_tensor * clipped_ratio
            policy_loss = torch.max(loss1, loss2).mean()

            # ==================== KL Loss (Low-variance) ====================
            # Search-R1 low-variance KL: 0.5 * (ratio - 1)^2
            kl_ratio = torch.exp(policy_token_log_probs - ref_token_log_probs)
            kl_loss = 0.5 * ((kl_ratio - 1) ** 2).mean()

            # Total loss
            loss = policy_loss + kl_coef * kl_loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            n_valid_samples += 1

        # Update weights
        if n_valid_samples > 0 and (rollout_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Log metrics
        metrics = {
            "rollout_idx": rollout_idx,
            "instance_id": instance_id,
            "rewards": group_rewards,
            "mean_reward": mean_reward,
            "advantages": advantages,
            "loss": total_loss / max(n_valid_samples, 1),
            "policy_loss": total_policy_loss / max(n_valid_samples, 1),
            "kl_loss": total_kl_loss / max(n_valid_samples, 1),
        }
        all_metrics.append(metrics)
        total_samples += n_samples_per_prompt

        print(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")

        # Save checkpoint
        if (rollout_idx + 1) % save_every == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_{rollout_idx + 1}")
            print(f"\nSaving checkpoint to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save metrics
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)

    # ==================== Final Summary ====================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total rollouts: {len(train_instances)}")
    print(f"Total samples: {total_samples}")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print(f"Avg reward: {sum(m['mean_reward'] for m in all_metrics) / len(all_metrics):.3f}")

    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    print(f"\nSaving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save all metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Push to Hub if requested
    if push_to_hub and hub_model_id:
        print(f"\nPushing model to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    output_volume.commit()

    return {
        "status": "complete",
        "output_dir": output_dir,
        "total_rollouts": len(train_instances),
        "total_samples": total_samples,
        "avg_reward": sum(m["mean_reward"] for m in all_metrics) / len(all_metrics),
        "elapsed_minutes": elapsed / 60,
    }


def compute_heuristic_reward(response: str) -> float:
    """
    Compute heuristic reward based on patch quality.

    Returns:
        +1.0 if response contains a well-formed diff patch
        0.0 if response contains a partial/malformed patch
        -1.0 if no patch detected
    """
    response_lower = response.lower()

    # Check for diff markers
    has_diff_header = ("--- a/" in response or "--- " in response) and \
                      ("+++ b/" in response or "+++ " in response)
    has_hunk_header = "@@" in response
    has_changes = ("-" in response and "+" in response)

    if has_diff_header and has_hunk_header and has_changes:
        # Check for actual file paths
        import re
        file_pattern = r'---\s+[ab]/[\w/\._-]+\.py'
        if re.search(file_pattern, response):
            return 1.0
        return 0.5

    if has_diff_header or has_hunk_header:
        return 0.0

    # Check for code block with diff
    if "```diff" in response_lower:
        return 0.0

    return -1.0


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": model_cache,
        "/outputs": output_volume,
    },
)
def evaluate_model(
    model_path: str = "/outputs/grpo_training/final_model",
    num_samples: int = 30,
    temperature: float = 0.7,
):
    """
    Evaluate trained model on held-out test instances.
    """
    import json
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # Load test instances (use last N Django instances)
    print("\nLoading test instances...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
    django_instances = [x for x in ds if x["repo"] == "django/django"]
    test_instances = django_instances[-num_samples:]
    print(f"Evaluating on {len(test_instances)} instances")

    results = []
    total_reward = 0.0

    for i, instance in enumerate(test_instances):
        print(f"\n[{i+1}/{len(test_instances)}] {instance['instance_id']}")

        prompt = f"""You are an expert software engineer. Fix this bug and provide a git diff patch.

## Repository: {instance["repo"]}
## Problem Statement
{instance["problem_statement"][:2000]}

## Instructions
Provide ONLY a git diff patch that fixes this issue.

Your patch:"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = compute_heuristic_reward(response)

        results.append({
            "instance_id": instance["instance_id"],
            "reward": reward,
            "response_length": len(response),
        })
        total_reward += reward

        print(f"  Reward: {reward:.2f}, Length: {len(response)}")

    # Summary
    avg_reward = total_reward / len(results)
    good_patches = sum(1 for r in results if r["reward"] > 0)
    partial_patches = sum(1 for r in results if r["reward"] == 0)

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total instances: {len(results)}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Good patches (reward > 0): {good_patches} ({good_patches/len(results)*100:.1f}%)")
    print(f"Partial patches (reward = 0): {partial_patches} ({partial_patches/len(results)*100:.1f}%)")

    # Save results
    results_file = "/outputs/grpo_training/eval_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "avg_reward": avg_reward,
            "good_patches": good_patches,
            "partial_patches": partial_patches,
            "total": len(results),
            "results": results,
        }, f, indent=2)

    output_volume.commit()

    return {
        "avg_reward": avg_reward,
        "good_patches": good_patches,
        "partial_patches": partial_patches,
        "total": len(results),
    }


@app.local_entrypoint()
def main(
    action: str = "train",
    num_rollouts: int = 50,
    n_samples: int = 4,
    lr: float = 1e-6,
    kl_coef: float = 0.001,
    temperature: float = 1.0,
    use_lora: bool = True,
    save_every: int = 10,
):
    """
    Main entrypoint for GRPO training.

    Args:
        action: "train", "eval", or "both"
        num_rollouts: Number of training instances
        n_samples: Samples per prompt (group size)
        lr: Learning rate
        kl_coef: KL coefficient
        temperature: Sampling temperature
        use_lora: Use LoRA for efficient training
        save_every: Save checkpoint every N rollouts
    """
    if action in ["train", "both"]:
        print("=" * 70)
        print("Starting GRPO Training on Modal")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Rollouts: {num_rollouts}")
        print(f"  Samples per prompt: {n_samples}")
        print(f"  Learning rate: {lr}")
        print(f"  KL coefficient: {kl_coef}")
        print(f"  Temperature: {temperature}")
        print(f"  Use LoRA: {use_lora}")
        print("")

        result = train_grpo.remote(
            num_rollouts=num_rollouts,
            n_samples_per_prompt=n_samples,
            lr=lr,
            kl_coef=kl_coef,
            temperature=temperature,
            use_lora=use_lora,
            save_every=save_every,
        )
        print(f"\nTraining Result: {result}")

    if action in ["eval", "both"]:
        print("\nStarting Evaluation...")
        eval_result = evaluate_model.remote()
        print(f"\nEvaluation Result: {eval_result}")


if __name__ == "__main__":
    print("Run with: modal run examples/harbor/modal_grpo_trainer.py")
