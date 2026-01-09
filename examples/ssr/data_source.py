"""
SSR Data Source - Provides SWE-bench instances for SSR training.

Loads SWE-bench Verified instances and prepares them for:
- Bug Injector: Original repo state for bug injection
- Bug Solver: Buggy repo state for solving
"""

import json
import logging
import os
import random
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchInstance:
    """SWE-bench instance data."""

    instance_id: str = ""
    repo: str = ""
    base_commit: str = ""
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    hints_text: str = ""
    created_at: str = ""
    version: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    environment_setup_commit: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SWEBenchInstance":
        """Create from dictionary."""
        return cls(
            instance_id=data.get("instance_id", ""),
            repo=data.get("repo", ""),
            base_commit=data.get("base_commit", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            problem_statement=data.get("problem_statement", ""),
            hints_text=data.get("hints_text", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", ""),
            fail_to_pass=data.get("FAIL_TO_PASS", data.get("fail_to_pass", [])),
            pass_to_pass=data.get("PASS_TO_PASS", data.get("pass_to_pass", [])),
            environment_setup_commit=data.get("environment_setup_commit", ""),
        )

    def get_docker_image(self) -> str:
        """
        Get SWE-bench docker image name for this instance.

        SWE-bench naming convention:
            instance_id: django__django-16255
            image: swebench/sweb.eval.x86_64.django_1776_django-16255:latest

        The pattern is: replace '__' with '_1776_' in instance_id.
        """
        # Direct conversion: django__django-16255 -> django_1776_django-16255
        id_docker = self.instance_id.replace("__", "_1776_").lower()
        return f"swebench/sweb.eval.x86_64.{id_docker}:latest"


class SSRDataSource:
    """
    Data source for SSR training.

    Provides SWE-bench instances configured for injector or solver training.
    """

    def __init__(self, args: Namespace):
        self.args = args

        # Load dataset
        self.data_path = getattr(args, "ssr_data_path", None)
        self.instances: list[SWEBenchInstance] = []

        # SSR configuration
        self.agent_type = getattr(args, "ssr_agent_type", "both")  # "injector", "solver", "both"
        self.group_size = getattr(args, "ssr_group_size", 8)
        self.sample_idx = 0

        # Load data
        self._load_data()

        logger.info(f"SSRDataSource initialized with {len(self.instances)} instances")
        logger.info(f"Agent type: {self.agent_type}, Group size: {self.group_size}")

    def _load_data(self) -> None:
        """Load SWE-bench instances from data file."""
        if self.data_path is None:
            # Try default SWE-bench paths
            default_paths = [
                "/home/gaokaizhang/SWE-sft/data/sft/train.jsonl",
                "data/swebench_verified.jsonl",
                "data/swebench_lite.jsonl",
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self.data_path = path
                    break

        if self.data_path is None or not os.path.exists(self.data_path):
            logger.warning(f"No data file found at {self.data_path}")
            return

        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        instance = SWEBenchInstance.from_dict(data)
                        if instance.instance_id:
                            self.instances.append(instance)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")

        logger.info(f"Loaded {len(self.instances)} instances")

    def __len__(self) -> int:
        return len(self.instances)

    @property
    def dataset(self):
        """For compatibility with slime data source interface."""
        return self.instances

    def get_samples(self, batch_size: int) -> list[list[Sample]]:
        """
        Get batch of sample groups for rollout.

        Each group contains n_samples_per_prompt copies of the same instance.

        Returns:
            List of groups, where each group is a list of Sample objects
        """
        n_samples = getattr(self.args, "n_samples_per_prompt", 1)
        groups = []

        for _ in range(batch_size):
            # Get next instance
            if not self.instances:
                break

            instance = self.instances[self.sample_idx % len(self.instances)]
            self.sample_idx += 1

            # Create sample group
            group = []
            for i in range(n_samples):
                sample = self._create_sample(instance, i)
                group.append(sample)

            groups.append(group)

        return groups

    def _create_sample(self, instance: SWEBenchInstance, sample_idx: int) -> Sample:
        """
        Create a Sample from SWE-bench instance.

        IMPORTANT: The gold_patch MUST be applied before bug injection!

        SWE-bench Docker images contain the ORIGINAL BUGGY code. For SSR:
        - Bug Injector: Must apply gold_patch FIRST to get clean code, then inject NEW bugs
        - Bug Solver: Works with buggy code (either original or injected)
        """
        sample = Sample()
        sample.index = self.sample_idx * 100 + sample_idx
        sample.group_index = self.sample_idx

        # Determine agent type for this sample
        if self.agent_type == "both":
            # Alternate between injector and solver
            agent_type = "injector" if sample_idx % 2 == 0 else "solver"
        else:
            agent_type = self.agent_type

        # Set prompt (will be formatted in rollout)
        sample.prompt = ""  # Prompt is constructed in rollout based on agent type

        # Set metadata
        sample.metadata = {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "agent_type": agent_type,
            "repo_path": "/testbed",
            "docker_image": instance.get_docker_image(),

            # CRITICAL: gold_patch fixes the original SWE-bench bug
            # For INJECTOR: Must apply this FIRST to get clean code before injecting new bugs
            # For SOLVER: May be used to understand the expected fix pattern
            "gold_patch": instance.patch,

            # Test patch (adds tests that expose the bug)
            "test_patch": instance.test_patch,

            # Problem statement (original issue description)
            "problem_statement": instance.problem_statement,

            # Test info
            "fail_to_pass": instance.fail_to_pass,
            "pass_to_pass": instance.pass_to_pass,
        }

        # Set label for evaluation
        sample.label = instance.patch  # Ground truth patch

        return sample

    def save(self, rollout_id: int) -> None:
        """Save current state."""
        state = {
            "sample_idx": self.sample_idx,
            "rollout_id": rollout_id,
        }
        state_path = getattr(self.args, "ssr_state_path", "ssr_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)

    def load(self, rollout_id: int | None = None) -> None:
        """Load state from file."""
        state_path = getattr(self.args, "ssr_state_path", "ssr_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                self.sample_idx = state.get("sample_idx", 0)

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Add samples back to buffer (for partial rollout)."""
        # For SSR, we don't reuse partial samples
        pass


class SSRSolverDataSource(SSRDataSource):
    """
    Data source specifically for Bug Solver training.

    Uses bug artifacts generated by injectors as input.
    """

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.agent_type = "solver"

        # Load bug artifacts if available
        self.bug_artifacts: list[dict[str, Any]] = []
        self._load_bug_artifacts()

    def _load_bug_artifacts(self) -> None:
        """Load pre-generated bug artifacts."""
        artifacts_path = getattr(self.args, "ssr_artifacts_path", None)
        if artifacts_path and os.path.exists(artifacts_path):
            with open(artifacts_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.bug_artifacts.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            logger.info(f"Loaded {len(self.bug_artifacts)} bug artifacts")

    def _create_sample(self, instance: SWEBenchInstance, sample_idx: int) -> Sample:
        """Create solver sample with bug artifact."""
        sample = super()._create_sample(instance, sample_idx)
        sample.metadata["agent_type"] = "solver"

        # Find matching bug artifact if available
        instance_id = instance.instance_id
        for artifact in self.bug_artifacts:
            if artifact.get("instance_id") == instance_id:
                sample.metadata["bug_artifact"] = artifact
                break

        return sample


class SSRInjectorDataSource(SSRDataSource):
    """
    Data source specifically for Bug Injector training.

    Provides clean repo states for bug injection.
    """

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.agent_type = "injector"

        # Injector-specific configs
        self.injector_type = getattr(args, "ssr_injector_type", "removal")

    def _create_sample(self, instance: SWEBenchInstance, sample_idx: int) -> Sample:
        """Create injector sample."""
        sample = super()._create_sample(instance, sample_idx)
        sample.metadata["agent_type"] = "injector"
        sample.metadata["injector_type"] = self.injector_type

        return sample


# Factory function for slime
def create_data_source(args: Namespace) -> SSRDataSource:
    """Create appropriate data source based on args."""
    agent_type = getattr(args, "ssr_agent_type", "both")

    if agent_type == "solver":
        return SSRSolverDataSource(args)
    elif agent_type == "injector":
        return SSRInjectorDataSource(args)
    else:
        return SSRDataSource(args)
