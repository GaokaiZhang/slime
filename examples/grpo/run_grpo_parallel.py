#!/usr/bin/env python
"""
Parallel GRPO training on SWE-bench with swebench.harness evaluation.

Improvements over run_grpo_training.py:
1. Parallel sample execution using concurrent.futures
2. Real swebench.harness evaluation for rewards (not just patch detection)
3. Configurable group size (n_samples_per_prompt)

Usage:
    # Deploy vLLM server first
    modal deploy examples/grpo/modal_vllm.py

    # Run parallel training with swebench evaluation
    export VLLM_URL="https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run"
    python examples/grpo/run_grpo_parallel.py \
        --num-rollouts 10 \
        --n-samples-per-prompt 5 \
        --max-workers 4 \
        --use-docker \
        --use-swebench-eval
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_sample(
    sample_idx: int,
    prompt: str,
    instance_id: str,
    config,
    tokenizer,
    use_docker: bool,
    use_swebench_eval: bool,
) -> dict:
    """
    Run a single sample and return results.

    This function is designed to be called in parallel.
    """
    from examples.grpo.vllm_agent import VLLMAgentConfig, run_agent
    from examples.grpo.swebench_utils import get_docker_image

    agent_config = VLLMAgentConfig(
        api_url=config["vllm_url"],
        model_name=config["model_name"],
        max_tokens=config["max_response_len"],
        temperature=config["temperature"],
        max_turns=config["max_turns"],
    )

    try:
        # Run agent
        result = run_agent(
            prompt,
            agent_config,
            workdir="/testbed" if use_docker else "/tmp",
            tokenizer=tokenizer,
            instance_id=instance_id if use_docker else None,
            use_docker=use_docker,
        )

        # Compute reward
        if use_swebench_eval and result.patch:
            # Use swebench.harness for real evaluation
            reward = evaluate_patch_with_swebench(instance_id, result.patch)
        elif result.patch and "---" in result.patch and "+++" in result.patch:
            # Fallback: valid patch format but not verified
            reward = 0.0
        else:
            # No valid patch
            reward = -1.0

        return {
            "sample_idx": sample_idx,
            "instance_id": instance_id,
            "turns": len(result.trajectory.get("steps", [])),
            "tokens": len(result.completion_token_ids),
            "has_patch": bool(result.patch),
            "reward": reward,
            "completion_token_ids": result.completion_token_ids,
            "logprobs": result.logprobs,
            "patch": result.patch,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Sample {sample_idx} failed: {e}")
        return {
            "sample_idx": sample_idx,
            "instance_id": instance_id,
            "turns": 0,
            "tokens": 0,
            "has_patch": False,
            "reward": -1.0,
            "completion_token_ids": [],
            "logprobs": [],
            "patch": None,
            "error": str(e),
        }


def evaluate_patch_with_swebench(instance_id: str, patch: str) -> float:
    """
    Evaluate a patch using swebench.harness.

    Returns:
        +1.0 if patch resolves the issue (all tests pass)
        -1.0 otherwise
    """
    if not patch or not patch.strip():
        return -1.0

    try:
        # Create temporary prediction file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            prediction = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "grpo-agent",
            }
            f.write(json.dumps(prediction) + "\n")
            pred_file = f.name

        # Run swebench evaluation
        run_id = f"grpo_{instance_id}_{int(time.time())}"

        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--split", "test",
            "--predictions_path", pred_file,
            "--max_workers", "1",
            "--timeout", "300",
            "--run_id", run_id,
            "--instance_ids", instance_id,
        ]

        logger.info(f"Running swebench evaluation for {instance_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Check for success by looking at the output
        # swebench creates a report file with resolved_ids
        eval_file = Path(f"grpo-agent.{run_id}.json")
        if not eval_file.exists():
            # Try alternate locations
            for pattern in [f"*.{run_id}.json", f"logs/**/*.{run_id}.json"]:
                matches = list(Path('.').glob(pattern))
                if matches:
                    eval_file = matches[0]
                    break

        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            resolved_ids = eval_data.get("resolved_ids", [])

            # Clean up
            eval_file.unlink()

            if instance_id in resolved_ids:
                logger.info(f"[{instance_id}] RESOLVED!")
                return 1.0
            else:
                logger.info(f"[{instance_id}] Not resolved")
                return -1.0
        else:
            logger.warning(f"[{instance_id}] No evaluation result found")
            return -1.0

    except subprocess.TimeoutExpired:
        logger.error(f"[{instance_id}] Evaluation timeout")
        return -1.0
    except Exception as e:
        logger.error(f"[{instance_id}] Evaluation error: {e}")
        return -1.0
    finally:
        # Clean up temp file
        try:
            os.unlink(pred_file)
        except:
            pass


def run_parallel_samples(
    prompt: str,
    instance_id: str,
    n_samples: int,
    config: dict,
    tokenizer,
    use_docker: bool,
    use_swebench_eval: bool,
    max_workers: int,
) -> list[dict]:
    """
    Run n_samples in parallel and return all results.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_sample,
                idx,
                prompt,
                instance_id,
                config,
                tokenizer,
                use_docker,
                use_swebench_eval,
            ): idx
            for idx in range(n_samples)
        }

        for future in as_completed(futures):
            sample_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(
                    f"  Sample {sample_idx + 1}/{n_samples}: "
                    f"Turns={result['turns']}, Tokens={result['tokens']}, "
                    f"Patch={'Yes' if result['has_patch'] else 'No'}, "
                    f"Reward={result['reward']}"
                )
            except Exception as e:
                logger.error(f"  Sample {sample_idx + 1}/{n_samples} failed: {e}")
                results.append({
                    "sample_idx": sample_idx,
                    "instance_id": instance_id,
                    "reward": -1.0,
                    "error": str(e),
                })

    # Sort by sample_idx for consistent ordering
    results.sort(key=lambda x: x.get("sample_idx", 0))
    return results


async def run_training(args):
    """Run parallel GRPO training."""
    from examples.grpo.data_source import DjangoTrainDataSource, DjangoTestDataSource
    from examples.grpo.vllm_agent import get_tokenizer

    config = {
        "vllm_url": args.vllm_url,
        "model_name": args.model_name,
        "max_response_len": args.max_response_len,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
    }

    logger.info("=" * 70)
    logger.info("Parallel GRPO Training for SWE-bench")
    logger.info("=" * 70)
    logger.info(f"vLLM URL: {args.vllm_url}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Num rollouts: {args.num_rollouts}")
    logger.info(f"N samples per prompt (group size): {args.n_samples_per_prompt}")
    logger.info(f"Max parallel workers: {args.max_workers}")
    logger.info(f"Use Docker: {args.use_docker}")
    logger.info(f"Use swebench eval: {args.use_swebench_eval}")
    logger.info("")

    # Load data
    logger.info("Loading training data...")
    train_ds = DjangoTrainDataSource(limit=args.num_rollouts)
    logger.info(f"  Train instances: {len(train_ds)}")

    # Get tokenizer
    tokenizer = get_tokenizer(args.model_name)

    # Training metrics
    all_metrics = []
    all_rewards = []

    # Run training rollouts
    for rollout_idx in range(min(args.num_rollouts, len(train_ds))):
        sample = train_ds[rollout_idx]
        instance_id = sample.metadata["instance_id"]

        logger.info(f"\n[{rollout_idx + 1}/{args.num_rollouts}] {instance_id}")
        logger.info(f"  Running {args.n_samples_per_prompt} samples in parallel...")

        start_time = time.time()

        # Run samples in parallel
        results = run_parallel_samples(
            prompt=sample.prompt,
            instance_id=instance_id,
            n_samples=args.n_samples_per_prompt,
            config=config,
            tokenizer=tokenizer,
            use_docker=args.use_docker,
            use_swebench_eval=args.use_swebench_eval,
            max_workers=min(args.max_workers, args.n_samples_per_prompt),
        )

        elapsed = time.time() - start_time

        # Compute GRPO statistics
        group_rewards = [r["reward"] for r in results]
        mean_reward = sum(group_rewards) / len(group_rewards)
        all_rewards.extend(group_rewards)

        # Count patches and resolved
        n_patches = sum(1 for r in results if r.get("has_patch", False))
        n_resolved = sum(1 for r in results if r.get("reward", -1) > 0)

        metrics = {
            "instance_id": instance_id,
            "rollout_idx": rollout_idx,
            "rewards": group_rewards,
            "mean_reward": mean_reward,
            "n_patches": n_patches,
            "n_resolved": n_resolved,
            "elapsed_seconds": elapsed,
        }
        all_metrics.append(metrics)

        logger.info(
            f"  Group: mean_reward={mean_reward:.3f}, "
            f"patches={n_patches}/{args.n_samples_per_prompt}, "
            f"resolved={n_resolved}/{args.n_samples_per_prompt}, "
            f"time={elapsed:.1f}s"
        )

        # Save incrementally
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = output_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Training Summary")
    logger.info("=" * 70)
    logger.info(f"Total rollouts: {len(all_metrics)}")
    logger.info(f"Total samples: {len(all_rewards)}")
    logger.info(f"Mean reward: {sum(all_rewards) / len(all_rewards):.3f}")
    logger.info(f"Positive rewards (resolved): {sum(1 for r in all_rewards if r > 0)}/{len(all_rewards)}")
    logger.info(f"Zero rewards (patch only): {sum(1 for r in all_rewards if r == 0)}/{len(all_rewards)}")
    logger.info(f"Negative rewards (no patch): {sum(1 for r in all_rewards if r < 0)}/{len(all_rewards)}")

    return all_metrics


async def run_evaluation(args):
    """Run evaluation on test instances."""
    from examples.grpo.data_source import DjangoTestDataSource
    from examples.grpo.vllm_agent import VLLMAgentConfig, run_agent, get_tokenizer

    config = {
        "vllm_url": args.vllm_url,
        "model_name": args.model_name,
        "max_response_len": args.max_response_len,
        "temperature": 0.7,  # Lower temp for eval
        "max_turns": args.max_turns,
    }

    logger.info("=" * 70)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 70)

    # Load test data
    test_ds = DjangoTestDataSource()
    logger.info(f"Test instances: {len(test_ds)}")

    tokenizer = get_tokenizer(args.model_name)

    results = []

    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        instance_id = sample.metadata["instance_id"]

        logger.info(f"\n[{idx + 1}/{len(test_ds)}] {instance_id}")

        # Run single sample for eval (could also do multiple and take best)
        result = run_single_sample(
            sample_idx=0,
            prompt=sample.prompt,
            instance_id=instance_id,
            config=config,
            tokenizer=tokenizer,
            use_docker=args.use_docker,
            use_swebench_eval=args.use_swebench_eval,
        )

        results.append(result)

        logger.info(
            f"  Patch: {'Yes' if result['has_patch'] else 'No'}, "
            f"Reward: {result['reward']}, "
            f"Tokens: {result['tokens']}"
        )

    # Summary
    patches = sum(1 for r in results if r.get("has_patch", False))
    resolved = sum(1 for r in results if r.get("reward", -1) > 0)

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Total instances: {len(results)}")
    logger.info(f"Patches generated: {patches} ({patches/len(results)*100:.1f}%)")
    logger.info(f"Resolved: {resolved} ({resolved/len(results)*100:.1f}%)")

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
    parser = argparse.ArgumentParser(description="Parallel GRPO Training for SWE-bench")
    parser.add_argument("--vllm-url", type=str,
                       default=os.environ.get("VLLM_URL", "https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run"))
    parser.add_argument("--model-name", type=str,
                       default=os.environ.get("MODEL_NAME", "Kwai-Klear/Klear-AgentForge-8B-SFT"))
    parser.add_argument("--num-rollouts", type=int, default=10,
                       help="Number of instances to train on")
    parser.add_argument("--n-samples-per-prompt", type=int, default=5,
                       help="Group size - number of samples per instance for GRPO")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Max parallel workers for sample execution")
    parser.add_argument("--max-turns", type=int, default=30,
                       help="Max turns per agent run")
    parser.add_argument("--max-response-len", type=int, default=2048,
                       help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_parallel",
                       help="Output directory")
    parser.add_argument("--eval-only", action="store_true",
                       help="Run evaluation only")
    parser.add_argument("--test", action="store_true",
                       help="Run both training and evaluation")
    parser.add_argument("--use-docker", action="store_true",
                       help="Run tools in Docker containers")
    parser.add_argument("--use-swebench-eval", action="store_true",
                       help="Use swebench.harness for real evaluation (slower but accurate)")
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
