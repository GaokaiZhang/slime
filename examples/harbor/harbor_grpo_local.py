#!/usr/bin/env python
"""
Harbor GRPO Trainer - Local GPU

Train any model with any Harbor agent using GRPO (Group Relative Policy Optimization).
Supports both local Docker and Daytona cloud environments.
Supports both SWE-bench and C2Bug data sources.

Usage:
    # SWE-bench data (default)
    python examples/harbor/harbor_grpo_local.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --agent qwen-coder \
        --num-rollouts 10

    # C2Bug data from HuggingFace
    python examples/harbor/harbor_grpo_local.py \
        --data-source c2bug \
        --c2bug-dataset TwelfthStar/c2bug_tasks_django_Jan-22-2026 \
        --env daytona \
        --num-rollouts 10

    # Daytona cloud (no local Docker needed)
    export DAYTONA_API_KEY="your_key"
    export DAYTONA_API_URL="https://app.daytona.io/api"
    python examples/harbor/harbor_grpo_local.py \
        --env daytona \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --num-rollouts 10

Harbor Agents (all work with both docker and daytona):
    qwen-coder, mini-swe-agent, claude-code, openhands, aider, swe-agent, oracle, ...
"""

import argparse
import json
import logging
import os
import subprocess
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


def load_c2bug_instances(
    dataset_id: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    task_root: str = "/tmp/c2bug_harbor_tasks",
    num_instances: int = 50,
    docker_image: str = "swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
    workdir: str = "/testbed",
) -> list[dict]:
    """Load c2bug instances and convert to Harbor format."""
    from examples.harbor.c2bug_adapter import (
        load_c2bug_from_hf,
        C2BugToHarbor,
        C2BugLoader,
    )

    logger.info(f"Loading c2bug data from {dataset_id}...")
    collection = load_c2bug_from_hf(dataset_id)

    run_meta_override = {"docker_image": docker_image, "workdir": workdir}
    task_root_path = Path(task_root)

    converter = C2BugToHarbor(
        collection_source=collection,
        task_root=task_root_path,
        max_timeout_sec=3000.0,
        run_meta_override=run_meta_override,
    )

    logger.info(f"Converting {num_instances} c2bug tasks to Harbor format...")
    converter.generate_many(limit=num_instances, overwrite=True)

    instances = []
    loader = C2BugLoader(collection)
    loader.apply_run_meta(run_meta_override)

    for record in list(loader.iter_records())[:num_instances]:
        task_name = record.task_uid or record.instance_id
        task_dir = task_root_path / task_name
        if task_dir.exists():
            instances.append({
                "instance_id": task_name,
                "task_dir": str(task_dir),
                "problem_statement": record.issue_text,
                "repo": record.repo,
            })

    logger.info(f"Loaded {len(instances)} c2bug instances")
    return instances


def parse_harbor_reward(job_dir: Path) -> float:
    """Parse reward from Harbor job directory (reward.txt)."""
    for reward_file in job_dir.glob("**/reward.txt"):
        try:
            reward_text = reward_file.read_text().strip()
            return 1.0 if reward_text == "1" else -1.0
        except Exception:
            pass
    return -1.0


def run_harbor_c2bug_agent(
    instance: dict,
    agent: str,
    env: str,
    daytona_target: str = None,
    jobs_dir: str = "jobs",
    timeout: int = 1800,
    agent_model: str = None,
    agent_kwargs: dict = None,
) -> dict:
    """Run Harbor agent on a c2bug instance using task_dir.

    Args:
        agent_kwargs: Additional agent kwargs (e.g., {"base_url": "https://...", "api_key": "..."})
    """
    task_dir = instance["task_dir"]
    instance_id = instance["instance_id"]
    job_name = f"c2bug-{instance_id.replace('/', '_').replace('__', '_')[:50]}-{int(time.time())}"

    cmd = [
        "harbor", "run",
        "-p", task_dir,
        "--env", env,
        "--agent", agent,
        "--n-concurrent", "1",
        "--jobs-dir", jobs_dir,
        "--job-name", job_name,
        "--export-traces",
    ]

    if env == "daytona" and daytona_target:
        cmd.extend(["--ek", f"target={daytona_target}"])
    if agent_model:
        cmd.extend(["--model", agent_model])

    # Pass additional agent kwargs (e.g., base_url, api_key for qwen-coder)
    if agent_kwargs:
        for key, value in agent_kwargs.items():
            cmd.extend(["--ak", f"{key}={value}"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        # Parse reward from job directory (Harbor's built-in evaluation)
        job_dir = Path(jobs_dir) / job_name
        reward = parse_harbor_reward(job_dir)

        return {"response": "", "reward": reward, "status": "completed" if reward > 0 else "failed", "job_dir": str(job_dir)}

    except subprocess.TimeoutExpired:
        return {"response": "", "reward": -1.0, "status": "timeout", "job_dir": ""}
    except Exception as e:
        logger.error(f"Harbor error: {e}")
        return {"response": "", "reward": -1.0, "status": "error", "job_dir": ""}


def run_local_grpo_training(
    config: HarborGRPOConfig,
    num_rollouts: int = 50,
    test_mode: bool = False,
    data_source: str = "swebench",
    c2bug_dataset: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    c2bug_docker_image: str = "swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
    daytona_target: str = None,
    skip_training: bool = False,
    eval_method: str = "harbor",
    openai_base_url: str = None,
    openai_api_key: str = "local",
):
    """
    Run Harbor GRPO training on local GPU.

    Uses:
    - Harbor CLI for agent rollouts
    - Local GPU for GRPO training
    - Evaluation: Harbor verifier (default) or swebench.harness

    Args:
        config: Training configuration
        num_rollouts: Number of instances
        test_mode: If True, use only 5 instances
        data_source: "swebench" or "c2bug"
        c2bug_dataset: HuggingFace dataset ID for c2bug
        c2bug_docker_image: Docker image for c2bug tasks
        daytona_target: Daytona target ID
        skip_training: If True, only run rollouts without GRPO training
        eval_method: "harbor" (default, uses Harbor's built-in verifier) or "swebench" (uses swebench.harness)
        openai_base_url: OpenAI-compatible API base URL for agents that need external LLM
        openai_api_key: OpenAI API key (default: local)
    """
    logger.info("=" * 70)
    logger.info("Harbor GRPO Training - Local GPU")
    logger.info(f"  - Data source: {data_source}")
    logger.info(f"  - Agent: {config.agent}")
    logger.info(f"  - Environment: {config.env}")
    logger.info(f"  - Eval method: {eval_method}")
    logger.info("  - Local GPU: GRPO weight updates")
    logger.info("=" * 70)

    # Check GPU
    if not torch.cuda.is_available() and not skip_training:
        logger.error("CUDA not available! This trainer requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        logger.info(f"Device: {device} ({torch.cuda.get_device_name()})")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load training instances based on data source
    if data_source == "c2bug":
        if test_mode:
            num_rollouts = min(5, num_rollouts)
        train_instances = load_c2bug_instances(
            dataset_id=c2bug_dataset,
            num_instances=num_rollouts,
            docker_image=c2bug_docker_image,
        )
    else:
        train_instances = load_training_instances(
            num_instances=num_rollouts,
            test_mode=test_mode,
        )

    logger.info(f"Loaded {len(train_instances)} instances")

    # Load model and tokenizer (skip if only doing rollouts)
    model, tokenizer, ref_model, optimizer = None, None, None, None
    if not skip_training:
        model, tokenizer = load_model_and_tokenizer(
            model_name=config.model_name,
            device=device,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        ref_model = load_reference_model(config.model_name, device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1,
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
        rewards = []

        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            if data_source == "c2bug":
                # Build agent_kwargs for agents that need external LLM (e.g., qwen-coder)
                agent_kwargs = {}
                if openai_base_url:
                    agent_kwargs["base_url"] = openai_base_url
                    agent_kwargs["api_key"] = openai_api_key

                # C2Bug: run Harbor with task_dir
                result = run_harbor_c2bug_agent(
                    instance=instance,
                    agent=config.agent,
                    env=config.env,
                    daytona_target=daytona_target,
                    jobs_dir=config.jobs_dir,
                    agent_model=config.agent_model,
                    agent_kwargs=agent_kwargs if agent_kwargs else None,
                )
                # C2Bug always uses Harbor's built-in verifier (reward.txt)
                reward = result["reward"]
            else:
                # SWE-bench: run Harbor with dataset
                result = run_harbor_agent(
                    instance=instance,
                    config=config,
                    timeout=1800,
                )

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
                        for job_path in Path(config.jobs_dir).glob(f"*{instance_id.replace('/', '_')[:30]}*"):
                            reward = parse_harbor_reward(job_path)
                            if reward > 0:
                                break

            responses.append(result.get("response", ""))
            rewards.append(reward)
            total_samples += 1
            if reward > 0:
                total_resolved += 1

            logger.info(f"    Status: {result['status']}, Reward: {reward}")

        # Train GRPO step (skip if skip_training or no model loaded)
        if skip_training or model is None:
            logger.info(f"  Skipping training (skip_training={skip_training}, rewards={rewards})")
            all_metrics.append({"instance_id": instance_id, "rewards": rewards, "skipped": True})
            continue

        # Create prompt based on data source
        if data_source == "c2bug":
            prompt = f"""Fix this bug in {instance.get('repo', 'the codebase')}:

{instance.get('problem_statement', '')[:4000]}

Analyze and fix the issue."""
        else:
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

        # Save checkpoint (only if training)
        if model is not None and (idx + 1) % config.save_every == 0:
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

    # Save final checkpoint (only if training)
    if model is not None:
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
    parser = argparse.ArgumentParser(
        description="Harbor GRPO Trainer - Train any model with any agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Core arguments
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                       help="HuggingFace model for GRPO training (default: Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--agent", default="qwen-coder",
                       help="Harbor agent: qwen-coder, mini-swe-agent, claude-code, openhands, oracle (default: qwen-coder)")
    parser.add_argument("--agent-model", default=None,
                       help="Model for agent (e.g., openai/gpt-4o, openai/local-model). Defaults to training model.")
    parser.add_argument("--env", default="docker",
                       help="Environment: docker (local) or daytona (cloud) (default: docker)")
    parser.add_argument("--dataset", default="swebench-verified@1.0",
                       help="Harbor dataset (default: swebench-verified@1.0)")
    parser.add_argument("--agent-import-path", default=None,
                       help="Custom agent import path (e.g., my_agents.custom:MyAgent)")

    # Training arguments
    parser.add_argument("--num-rollouts", type=int, default=50,
                       help="Number of instances to train on (default: 50)")
    parser.add_argument("--instances", type=str, default=None,
                       help="Path to file with instance IDs (one per line)")
    parser.add_argument("--n-samples", type=int, default=4,
                       help="GRPO group size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate (default: 1e-6)")
    parser.add_argument("--kl-coef", type=float, default=0.001,
                       help="KL coefficient (default: 0.001)")

    # Output arguments
    parser.add_argument("--output-dir", default="outputs/harbor_grpo",
                       help="Output directory (default: outputs/harbor_grpo)")
    parser.add_argument("--jobs-dir", default="jobs",
                       help="Harbor jobs directory (default: jobs)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N instances (default: 10)")

    # Data source arguments
    parser.add_argument("--data-source", default="swebench", choices=["swebench", "c2bug"],
                       help="Data source: swebench (default) or c2bug")
    parser.add_argument("--c2bug-dataset", default="TwelfthStar/c2bug_tasks_django_Jan-22-2026",
                       help="HuggingFace dataset for c2bug (default: TwelfthStar/c2bug_tasks_django_Jan-22-2026)")
    parser.add_argument("--c2bug-docker-image", default="swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
                       help="Docker image for c2bug tasks")
    parser.add_argument("--daytona-target", default=None,
                       help="Daytona target ID (or set DAYTONA_TARGET env var)")

    # Evaluation method
    parser.add_argument("--eval-method", default="harbor", choices=["harbor", "swebench"],
                       help="Evaluation method: harbor (default, uses Harbor's built-in verifier) or swebench (uses swebench.harness)")

    # OpenAI API settings (for agents like qwen-coder that need an external LLM)
    parser.add_argument("--openai-base-url", default=None,
                       help="OpenAI-compatible API base URL for the agent (e.g., https://your-tunnel.loca.lt/v1)")
    parser.add_argument("--openai-api-key", default="local",
                       help="OpenAI API key (default: local)")

    # Flags
    parser.add_argument("--test", action="store_true",
                       help="Test mode (5 instances)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Only run rollouts, skip GRPO training")
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA (full fine-tuning)")
    args = parser.parse_args()

    # Get daytona target from args or environment
    daytona_target = args.daytona_target or os.environ.get("DAYTONA_TARGET")

    # Set output directory based on data source if using default
    output_dir = args.output_dir
    if output_dir == "outputs/harbor_grpo" and args.data_source == "c2bug":
        output_dir = "outputs/harbor_grpo_c2bug"

    config = HarborGRPOConfig(
        model_name=args.model,
        agent=args.agent,
        agent_model=args.agent_model,
        agent_import_path=getattr(args, 'agent_import_path', None),
        env=args.env,
        dataset=args.dataset,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        use_lora=not args.no_lora,
        output_dir=output_dir,
        jobs_dir=args.jobs_dir,
        save_every=args.save_every,
    )

    run_local_grpo_training(
        config=config,
        num_rollouts=args.num_rollouts,
        test_mode=args.test,
        data_source=args.data_source,
        c2bug_dataset=args.c2bug_dataset,
        c2bug_docker_image=args.c2bug_docker_image,
        daytona_target=daytona_target,
        skip_training=args.skip_training,
        eval_method=args.eval_method,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
    )


if __name__ == "__main__":
    main()
