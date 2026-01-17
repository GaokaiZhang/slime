#!/usr/bin/env python
"""
Agentic GRPO Trainer - Local GPU Version

This trainer runs on a local GPU with vLLM server running locally or remotely.
Uses shared code from agentic_grpo_core.py.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      Local GPU                               │
    │  ┌──────────────────┐    ┌──────────────────────────────┐   │
    │  │ vLLM Server      │    │ Training (this script)       │   │
    │  │ (local/remote)   │    │ - Model loading              │   │
    │  │                  │    │ - GRPO updates               │   │
    │  └──────────────────┘    │ - Gradient computation       │   │
    │           │              └──────────────────────────────┘   │
    │           │                           │                     │
    │           ▼                           ▼                     │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Docker Containers (local)                │   │
    │  │  - Agent tool execution                               │   │
    │  │  - swebench.harness evaluation                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Start vLLM server first (in another terminal)
    python -m vllm.entrypoints.openai.api_server \
        --model Kwai-Klear/Klear-AgentForge-8B-SFT \
        --port 8000 --trust-remote-code

    # Or use Modal vLLM
    modal deploy examples/grpo/modal_vllm.py

    # Run training
    python examples/grpo/agentic_grpo_local.py --test
    python examples/grpo/agentic_grpo_local.py --num-rollouts 50 --n-samples 4

    # With remote vLLM
    python examples/grpo/agentic_grpo_local.py --vllm-url http://remote:8000 --num-rollouts 50
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared core
from examples.grpo.agentic_grpo_core import (
    AgenticGRPOConfig,
    AgentRollout,
    run_agent_rollout,
    evaluate_patch,
    train_grpo_step,
    load_model_and_tokenizer,
    load_reference_model,
    load_training_instances,
    create_training_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_local_grpo_training(
    config: AgenticGRPOConfig,
    num_rollouts: int = 50,
    test_mode: bool = False,
    output_dir: str = "outputs/agentic_grpo_local",
    save_every: int = 10,
):
    """
    Run agentic GRPO training on local GPU.

    Uses shared implementation from agentic_grpo_core.py.
    """
    logger.info("=" * 70)
    logger.info("Agentic GRPO Training - Local GPU")
    logger.info("  - Multi-turn agent interaction with Docker")
    logger.info("  - Tool responses MASKED from loss")
    logger.info("  - SLiME ppo_utils with Search-R1 parameters")
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
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    all_metrics = []
    total_resolved = 0
    total_samples = 0
    start_time = time.time()

    for idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n[{idx + 1}/{len(train_instances)}] {instance_id}")

        # Generate rollouts
        rollouts = []
        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            rollout = run_agent_rollout(
                instance_id=instance_id,
                problem_statement=instance["problem_statement"],
                config=config,
                tokenizer=tokenizer,
            )

            # Evaluate
            rollout.reward = evaluate_patch(
                instance_id,
                rollout.patch,
                timeout=config.eval_timeout,
            )

            rollouts.append(rollout)
            total_samples += 1
            if rollout.reward > 0:
                total_resolved += 1

            logger.info(f"    Turns: {len(rollout.turns)}, Tokens: {rollout.total_model_tokens()}, Reward: {rollout.reward}")

        # Train
        prompt = create_training_prompt(instance)
        metrics = train_grpo_step(
            prompt=prompt,
            rollouts=rollouts,
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
        if (idx + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{idx + 1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"  Saved checkpoint to {checkpoint_path}")

        # Save metrics
        import json
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

    # Save final checkpoint
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save summary
    import json
    summary = {
        "total_instances": len(train_instances),
        "total_samples": total_samples,
        "total_resolved": total_resolved,
        "resolve_rate": resolve_rate,
        "elapsed_minutes": elapsed / 60,
        "config": {
            "model_name": config.model_name,
            "vllm_url": config.vllm_url,
            "lr": config.lr,
            "kl_coef": config.kl_coef,
            "n_samples_per_prompt": config.n_samples_per_prompt,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Agentic GRPO Trainer - Local GPU")
    parser.add_argument("--num-rollouts", type=int, default=50,
                       help="Number of SWE-bench instances to train on")
    parser.add_argument("--n-samples", type=int, default=4,
                       help="Number of samples per instance (GRPO group size)")
    parser.add_argument("--vllm-url", default="http://localhost:8000",
                       help="vLLM server URL")
    parser.add_argument("--model", default="Kwai-Klear/Klear-AgentForge-8B-SFT",
                       help="Model name")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.001,
                       help="KL coefficient")
    parser.add_argument("--output-dir", default="outputs/agentic_grpo_local",
                       help="Output directory")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N instances")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (2 instances)")
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA (use full fine-tuning)")
    parser.add_argument("--max-response-tokens", type=int, default=4096,
                       help="Max response tokens (truncate longer)")
    args = parser.parse_args()

    config = AgenticGRPOConfig(
        model_name=args.model,
        vllm_url=args.vllm_url,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        use_lora=not args.no_lora,
        max_response_tokens=args.max_response_tokens,
    )

    run_local_grpo_training(
        config=config,
        num_rollouts=args.num_rollouts,
        test_mode=args.test,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
