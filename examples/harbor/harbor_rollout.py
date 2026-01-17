#!/usr/bin/env python
"""
Harbor Rollout Generator for SLiME

Generates trajectories using Harbor CLI, then converts to SLiME Sample format.
This is the bridge between Harbor (evaluation) and SLiME (training).

Key insight:
- Harbor runs the agent (qwen-code) on tasks and collects trajectories
- SLiME consumes these trajectories for GRPO training
- Log probs are NOT needed from Harbor - computed at training time

Usage:
    from harbor_rollout import HarborRolloutGenerator

    generator = HarborRolloutGenerator(
        model="Qwen/Qwen3-Coder-30B-A3B",
        agent="qwen-coder",
    )

    samples = generator.generate(
        dataset="swebench-verified",
        n_tasks=10,
    )
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HarborConfig:
    """Configuration for Harbor rollout generation."""

    # Model settings
    model: str = "Qwen/Qwen3-Coder-30B-A3B"

    # Agent settings
    agent: str = "qwen-coder"
    agent_kwargs: dict[str, Any] = field(default_factory=dict)

    # Environment settings
    environment: str = "docker"
    n_concurrent: int = 4

    # Dataset settings
    dataset: str | None = None
    task_path: str | None = None
    task_names: list[str] | None = None

    # Output settings
    jobs_dir: str = "jobs"
    export_traces: bool = True

    # Harbor binary path (uv tool install location)
    harbor_bin: str = "harbor"

    def __post_init__(self):
        # Find harbor binary if not specified
        if self.harbor_bin == "harbor":
            home = os.path.expanduser("~")
            uv_harbor = os.path.join(home, ".local", "bin", "harbor")
            if os.path.exists(uv_harbor):
                self.harbor_bin = uv_harbor


class HarborRolloutGenerator:
    """
    Generates rollouts using Harbor CLI and converts to SLiME format.

    This class:
    1. Runs Harbor to generate trajectories
    2. Parses the output trajectories
    3. Converts to SLiME Sample format

    No log probs are collected - they are recomputed at training time.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-30B-A3B",
        agent: str = "qwen-coder",
        tokenizer=None,
        config: HarborConfig = None,
    ):
        """
        Args:
            model: Model name for the agent
            agent: Harbor agent name (qwen-coder, openhands, etc.)
            tokenizer: HuggingFace tokenizer (loaded if not provided)
            config: Full configuration (overrides model/agent if provided)
        """
        self.config = config or HarborConfig(model=model, agent=agent)

        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Lazy import converter
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            from trajectory_converter import HarborTrajectoryConverter
            self._converter = HarborTrajectoryConverter(self.tokenizer)
        return self._converter

    def _build_harbor_command(
        self,
        job_name: str,
        dataset: str | None = None,
        task_path: str | None = None,
        task_names: list[str] | None = None,
        n_tasks: int | None = None,
    ) -> list[str]:
        """Build the harbor CLI command."""
        cmd = [
            self.config.harbor_bin,
            "run",
            "--agent", self.config.agent,
            "--model", self.config.model,
            "--env", self.config.environment,
            "--n-concurrent", str(self.config.n_concurrent),
            "--jobs-dir", self.config.jobs_dir,
            "--job-name", job_name,
        ]

        # Dataset or task path
        if dataset or self.config.dataset:
            cmd.extend(["--dataset", dataset or self.config.dataset])
        elif task_path or self.config.task_path:
            cmd.extend(["--path", task_path or self.config.task_path])

        # Task names (filter)
        task_names = task_names or self.config.task_names
        if task_names:
            for name in task_names[:n_tasks] if n_tasks else task_names:
                cmd.extend(["--task-name", name])

        # Agent kwargs
        for key, value in self.config.agent_kwargs.items():
            cmd.extend(["--ak", f"{key}={value}"])

        # Export traces for training
        if self.config.export_traces:
            cmd.append("--export-traces")

        return cmd

    def generate(
        self,
        dataset: str | None = None,
        task_path: str | None = None,
        task_names: list[str] | None = None,
        n_tasks: int | None = None,
        job_name: str | None = None,
    ) -> list["Sample"]:
        """
        Generate trajectories using Harbor.

        Args:
            dataset: Harbor dataset name (e.g., "swebench-verified@1.0")
            task_path: Path to local task directory
            task_names: Specific task names to run
            n_tasks: Maximum number of tasks to run
            job_name: Name for the job (auto-generated if not provided)

        Returns:
            List of SLiME Sample objects
        """
        import time

        job_name = job_name or f"slime-rollout-{int(time.time())}"
        job_dir = Path(self.config.jobs_dir) / job_name

        # Build and run harbor command
        cmd = self._build_harbor_command(
            job_name=job_name,
            dataset=dataset,
            task_path=task_path,
            task_names=task_names,
            n_tasks=n_tasks,
        )

        logger.info(f"Running Harbor: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Harbor failed: {result.stderr}")
                raise RuntimeError(f"Harbor exited with code {result.returncode}")

            logger.info(f"Harbor completed. Output in {job_dir}")

        except subprocess.TimeoutExpired:
            logger.error("Harbor timed out")
            raise

        # Convert trajectories to SLiME format
        samples = self.converter.load_job(job_dir)

        # Load from parquet if available (more complete)
        traces_file = job_dir / "traces.parquet"
        if traces_file.exists():
            parquet_samples = self.converter.load_traces_parquet(traces_file)
            if len(parquet_samples) > len(samples):
                samples = parquet_samples

        logger.info(f"Generated {len(samples)} samples")
        resolved = sum(1 for s in samples if s.reward > 0)
        logger.info(f"Resolved: {resolved}/{len(samples)} ({100*resolved/len(samples):.1f}%)")

        return samples

    def generate_from_existing_job(self, job_dir: str | Path) -> list["Sample"]:
        """
        Load samples from an existing Harbor job directory.

        Useful for resuming training or reprocessing old trajectories.

        Args:
            job_dir: Path to existing Harbor job

        Returns:
            List of SLiME Sample objects
        """
        return self.converter.load_job(job_dir)


class AsyncHarborRolloutGenerator(HarborRolloutGenerator):
    """
    Async version for integration with SLiME's rollout system.

    Can be used as a custom_generate_function in SLiME.
    """

    async def generate_async(
        self,
        dataset: str | None = None,
        task_names: list[str] | None = None,
        n_tasks: int | None = None,
    ) -> list["Sample"]:
        """Async wrapper around generate()."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                dataset=dataset,
                task_names=task_names,
                n_tasks=n_tasks,
            )
        )


def create_slime_data_source(
    generator: HarborRolloutGenerator,
    instances: list[dict],
    n_samples_per_prompt: int = 1,
) -> callable:
    """
    Create a data source function compatible with SLiME's rollout system.

    This allows Harbor rollouts to be used with SLiME's training loop.

    Args:
        generator: HarborRolloutGenerator instance
        instances: List of task instances (with instance_id, problem_statement)
        n_samples_per_prompt: Number of samples per task

    Returns:
        A data source function for SLiME
    """
    from slime.utils.types import Sample

    task_names = [inst["instance_id"] for inst in instances]
    current_idx = 0

    def get_samples(batch_size: int) -> list[list[Sample]]:
        nonlocal current_idx

        # Get task names for this batch
        batch_tasks = task_names[current_idx:current_idx + batch_size]
        current_idx += batch_size

        if not batch_tasks:
            return []

        # Generate samples using Harbor
        samples = generator.generate(task_names=batch_tasks)

        # Group by n_samples_per_prompt
        groups = []
        for i in range(0, len(samples), n_samples_per_prompt):
            group = samples[i:i + n_samples_per_prompt]
            if group:
                groups.append(group)

        return groups

    return get_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Harbor rollouts for SLiME")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B")
    parser.add_argument("--agent", default="qwen-coder")
    parser.add_argument("--dataset", default="swebench-verified@1.0")
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--output", "-o", help="Output JSON file for samples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = HarborConfig(
        model=args.model,
        agent=args.agent,
        n_concurrent=args.n_concurrent,
        jobs_dir=args.jobs_dir,
    )

    generator = HarborRolloutGenerator(config=config)
    samples = generator.generate(dataset=args.dataset, n_tasks=args.n_tasks)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([s.to_dict() for s in samples], f, indent=2)
        print(f"Saved {len(samples)} samples to {args.output}")

    print(f"\nGenerated {len(samples)} samples")
    resolved = sum(1 for s in samples if s.reward > 0)
    print(f"Resolved: {resolved}/{len(samples)} ({100*resolved/len(samples):.1f}%)")
