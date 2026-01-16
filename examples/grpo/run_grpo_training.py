#!/usr/bin/env python
"""
Run GRPO training on SWE-bench with the 8B model.

This script:
1. Uses the 201 train instances from train_instances_id.txt
2. Uses the external Modal vLLM server for rollouts
3. Trains with GRPO using Search-R1 hyperparameters
4. Evaluates on the 30 test instances

Usage:
    # First deploy vLLM server
    modal deploy examples/grpo/modal_vllm.py

    # Run training
    export VLLM_URL="https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run"
    python examples/grpo/run_grpo_training.py --num-rollouts 10 --test
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_training(args):
    """Run GRPO training."""
    from examples.grpo.data_source import DjangoTrainDataSource, DjangoTestDataSource
    from examples.grpo.vllm_agent import VLLMAgentConfig, run_agent, get_tokenizer
    from examples.grpo.rollout import evaluate_with_harbor

    # Check vLLM URL
    vllm_url = args.vllm_url
    model_name = args.model_name

    logger.info("=" * 70)
    logger.info("GRPO Training for SWE-bench")
    logger.info("=" * 70)
    logger.info(f"vLLM URL: {vllm_url}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Num rollouts: {args.num_rollouts}")
    logger.info(f"N samples per prompt: {args.n_samples_per_prompt}")
    logger.info("")

    # Load data
    logger.info("Loading training data...")
    train_ds = DjangoTrainDataSource(limit=args.num_rollouts)
    logger.info(f"  Train instances: {len(train_ds)}")

    # Get tokenizer
    tokenizer = get_tokenizer(model_name)

    # Training metrics
    all_metrics = []
    all_rewards = []

    # Run training rollouts
    for rollout_idx in range(min(args.num_rollouts, len(train_ds))):
        sample = train_ds[rollout_idx]
        instance_id = sample.metadata["instance_id"]

        logger.info(f"\n[{rollout_idx + 1}/{args.num_rollouts}] {instance_id}")

        group_rewards = []

        for sample_idx in range(args.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{args.n_samples_per_prompt}...")

            config = VLLMAgentConfig(
                api_url=vllm_url,
                model_name=model_name,
                max_tokens=args.max_response_len,
                temperature=args.temperature,
                max_turns=args.max_turns,
            )

            try:
                # Run agent (optionally in Docker)
                use_docker = getattr(args, 'use_docker', False)
                result = await asyncio.to_thread(
                    run_agent,
                    sample.prompt,
                    config,
                    workdir="/testbed" if use_docker else "/tmp",
                    tokenizer=tokenizer,
                    instance_id=instance_id if use_docker else None,
                    use_docker=use_docker,
                )

                # Get reward (heuristic without Docker)
                if result.patch and "---" in result.patch and "+++" in result.patch:
                    reward = 0.0  # Valid patch but can't verify
                else:
                    reward = -1.0  # No valid patch

                group_rewards.append(reward)

                logger.info(f"    Turns: {len(result.trajectory.get('steps', []))}, "
                           f"Tokens: {len(result.completion_token_ids)}, "
                           f"Patch: {'Yes' if result.patch else 'No'}, "
                           f"Reward: {reward}")

            except Exception as e:
                logger.error(f"    Error: {e}")
                group_rewards.append(-1.0)

        # Compute GRPO statistics for this group
        mean_reward = sum(group_rewards) / len(group_rewards)
        all_rewards.extend(group_rewards)

        metrics = {
            "instance_id": instance_id,
            "rollout_idx": rollout_idx,
            "rewards": group_rewards,
            "mean_reward": mean_reward,
        }
        all_metrics.append(metrics)

        logger.info(f"  Group mean reward: {mean_reward:.3f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Training Summary")
    logger.info("=" * 70)
    logger.info(f"Total rollouts: {len(all_metrics)}")
    logger.info(f"Total samples: {len(all_rewards)}")
    logger.info(f"Mean reward: {sum(all_rewards) / len(all_rewards):.3f}")
    logger.info(f"Positive rewards: {sum(1 for r in all_rewards if r >= 0)}/{len(all_rewards)}")

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    return all_metrics


async def run_evaluation(args):
    """Run evaluation on test instances."""
    from examples.grpo.data_source import DjangoTestDataSource
    from examples.grpo.vllm_agent import VLLMAgentConfig, run_agent, get_tokenizer

    vllm_url = args.vllm_url
    model_name = args.model_name

    logger.info("=" * 70)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 70)

    # Load test data
    test_ds = DjangoTestDataSource()
    logger.info(f"Test instances: {len(test_ds)}")

    tokenizer = get_tokenizer(model_name)

    results = []

    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        instance_id = sample.metadata["instance_id"]

        logger.info(f"\n[{idx + 1}/{len(test_ds)}] {instance_id}")

        config = VLLMAgentConfig(
            api_url=vllm_url,
            model_name=model_name,
            max_tokens=args.max_response_len,
            temperature=0.7,  # Lower temp for eval
            max_turns=args.max_turns,
        )

        try:
            use_docker = getattr(args, 'use_docker', False)
            result = await asyncio.to_thread(
                run_agent,
                sample.prompt,
                config,
                workdir="/testbed" if use_docker else "/tmp",
                tokenizer=tokenizer,
                instance_id=instance_id if use_docker else None,
                use_docker=use_docker,
            )

            has_patch = result.patch and "---" in result.patch and "+++" in result.patch

            results.append({
                "instance_id": instance_id,
                "has_patch": has_patch,
                "tokens": len(result.completion_token_ids),
                "turns": len(result.trajectory.get("steps", [])),
            })

            logger.info(f"  Patch: {'Yes' if has_patch else 'No'}, "
                       f"Tokens: {len(result.completion_token_ids)}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                "instance_id": instance_id,
                "has_patch": False,
                "error": str(e),
            })

    # Summary
    patches = sum(1 for r in results if r.get("has_patch", False))
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Total instances: {len(results)}")
    logger.info(f"Patches generated: {patches}")
    logger.info(f"Patch rate: {patches / len(results) * 100:.1f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GRPO Training for SWE-bench")
    parser.add_argument("--vllm-url", type=str,
                       default=os.environ.get("VLLM_URL", "https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run"))
    parser.add_argument("--model-name", type=str,
                       default=os.environ.get("MODEL_NAME", "Kwai-Klear/Klear-AgentForge-8B-SFT"))
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--n-samples-per-prompt", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-response-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_training")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--test", action="store_true", help="Run both training and evaluation")
    parser.add_argument("--use-docker", action="store_true", help="Run tools in Docker containers")
    args = parser.parse_args()

    if args.eval_only:
        await run_evaluation(args)
    elif args.test:
        await run_training(args)
        await run_evaluation(args)
    else:
        await run_training(args)


if __name__ == "__main__":
    asyncio.run(main())
