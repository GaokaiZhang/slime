#!/usr/bin/env python
"""
Harbor GRPO Trainer - Local GPU Version

This trainer runs on a local GPU with Harbor for agent rollouts.
Uses shared code from harbor_core.py.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      Local Machine                           │
    │  ┌──────────────────┐    ┌──────────────────────────────┐   │
    │  │ Harbor Agent     │    │ Training (this script)       │   │
    │  │ (mini-swe-agent) │    │ - Model loading (GPU)        │   │
    │  │                  │    │ - GRPO updates               │   │
    │  └──────────────────┘    │ - Gradient computation       │   │
    │           │              └──────────────────────────────┘   │
    │           │                           │                     │
    │           ▼                           ▼                     │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Docker Containers (local)                │   │
    │  │  - Harbor agent tool execution                        │   │
    │  │  - swebench.harness evaluation                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Test mode (5 instances)
    python examples/harbor/harbor_grpo_local.py --test

    # Full training (201 Django instances)
    python examples/harbor/harbor_grpo_local.py --num-rollouts 201

    # With custom agent
    python examples/harbor/harbor_grpo_local.py --agent qwen-coder --num-rollouts 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared core
from examples.harbor.harbor_core import (
    HarborGRPOConfig,
    run_harbor_agent,
    evaluate_with_swebench,
    train_grpo_step,
    load_model_and_tokenizer,
    load_reference_model,
    load_training_instances,
    create_swebench_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_local_grpo_training(
    config: HarborGRPOConfig,
    num_rollouts: int = 50,
    test_mode: bool = False,
):
    """
    Run Harbor GRPO training on local GPU.

    Uses:
    - Harbor CLI for agent rollouts
    - Local GPU for GRPO training
    - swebench.harness for evaluation

    Args:
        config: Training configuration
        num_rollouts: Number of SWE-bench instances
        test_mode: If True, use only 5 instances
    """
    logger.info("=" * 70)
    logger.info("Harbor GRPO Training - Local GPU")
    logger.info("  - Harbor: Agent rollouts")
    logger.info("  - Local GPU: GRPO weight updates")
    logger.info("  - swebench.harness: Evaluation")
    logger.info("  - SLiME ppo_utils: Search-R1 parameters")
    logger.info("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This trainer requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name()})")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device=device,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Load reference model
    ref_model = load_reference_model(config.model_name, device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )

    # Load training instances
    train_instances = load_training_instances(
        num_instances=num_rollouts,
        test_mode=test_mode,
    )

    # Output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.jobs_dir, exist_ok=True)

    # Training loop
    all_metrics = []
    total_resolved = 0
    total_samples = 0
    start_time = time.time()

    for idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n[{idx + 1}/{len(train_instances)}] {instance_id}")

        # Generate rollouts using Harbor
        responses = []
        patches = []
        rewards = []

        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            # Run Harbor agent
            result = run_harbor_agent(
                instance=instance,
                config=config,
                timeout=1800,
            )

            response = result["response"]
            patch = result["patch"]

            # Evaluate with swebench.harness
            reward = evaluate_with_swebench(
                instance_id=instance_id,
                patch=patch,
                timeout=config.eval_timeout,
            )

            responses.append(response)
            patches.append(patch)
            rewards.append(reward)

            total_samples += 1
            if reward > 0:
                total_resolved += 1

            logger.info(f"    Status: {result['status']}, Reward: {reward}")

        # Train GRPO step
        prompt = create_swebench_prompt(instance)
        metrics = train_grpo_step(
            prompt=prompt,
            responses=responses,
            rewards=rewards,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        metrics["instance_id"] = instance_id
        metrics["idx"] = idx
        all_metrics.append(metrics)

        logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")

        # Save checkpoint
        if (idx + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{idx + 1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"  Saved checkpoint to {checkpoint_path}")

        # Save metrics
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

    # Save final checkpoint
    final_path = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save summary
    summary = {
        "total_instances": len(train_instances),
        "total_samples": total_samples,
        "total_resolved": total_resolved,
        "resolve_rate": resolve_rate,
        "elapsed_minutes": elapsed / 60,
        "config": {
            "model_name": config.model_name,
            "agent": config.agent,
            "lr": config.lr,
            "kl_coef": config.kl_coef,
            "n_samples_per_prompt": config.n_samples_per_prompt,
        },
    }
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Harbor GRPO Trainer - Local GPU")
    parser.add_argument("--num-rollouts", type=int, default=50,
                       help="Number of SWE-bench instances to train on")
    parser.add_argument("--n-samples", type=int, default=4,
                       help="Number of samples per instance (GRPO group size)")
    parser.add_argument("--agent", default="mini-swe-agent-plus",
                       help="Harbor agent (mini-swe-agent-plus, qwen-coder, etc.)")
    parser.add_argument("--model", default="Kwai-Klear/Klear-AgentForge-8B-SFT",
                       help="Model name")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.001,
                       help="KL coefficient")
    parser.add_argument("--output-dir", default="outputs/harbor_grpo_local",
                       help="Output directory")
    parser.add_argument("--jobs-dir", default="jobs",
                       help="Harbor jobs directory")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N instances")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (5 instances)")
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA (use full fine-tuning)")
    args = parser.parse_args()

    config = HarborGRPOConfig(
        model_name=args.model,
        agent=args.agent,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        use_lora=not args.no_lora,
        output_dir=args.output_dir,
        jobs_dir=args.jobs_dir,
        save_every=args.save_every,
    )

    run_local_grpo_training(
        config=config,
        num_rollouts=args.num_rollouts,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
