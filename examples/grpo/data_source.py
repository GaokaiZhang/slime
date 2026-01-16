"""
DataSource for SWE-bench datasets using Harbor.

Loads SWE-Bench_Verified instances and formats them for SLiME GRPO training.
"""

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# Bug solving prompt template (from bug_solving_prompts.txt)
BUG_SOLVING_PROMPT = """You are an expert software engineer tasked with fixing a bug in a software repository.

## Repository Information
- Repository: {repo}
- Version: {version}
- Base Commit: {base_commit}

## Problem Statement
{problem_statement}

## Instructions
1. Explore the repository to understand its structure
2. Locate the relevant code related to the issue
3. Understand the bug and its root cause
4. Implement a fix that resolves the issue
5. Run the relevant tests to verify your fix

The repository is located at /testbed. Start by exploring the codebase and understanding the issue before making changes.
"""


class SWEBenchVerifiedDataSource(DataSource):
    """
    DataSource for SWE-Bench_Verified dataset.

    Loads instances from HuggingFace or local cache.
    """

    def __init__(
        self,
        args=None,
        split: str = "test",
        subset: list[str] | None = None,
        repos: list[str] | None = None,
        limit: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize the SWE-Bench Verified data source.

        Args:
            args: Training arguments (for compatibility with SLiME)
            split: Dataset split (usually "test" for verified)
            subset: Optional list of specific instance_ids to use
            repos: Optional list of repos to filter (e.g., ["django/django"])
            limit: Maximum number of instances to load
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
        """
        self.args = args
        self.split = split
        self.subset = subset
        self.repos = repos
        self.limit = limit
        self.shuffle_enabled = shuffle
        self.seed = seed
        self._instances = None

        # State tracking
        self.sample_offset = 0
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata = {}

        # N samples per prompt
        self.n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 5) if args else 5

    def _load_instances(self) -> list[dict]:
        """Load SWE-Bench_Verified instances."""
        logger.info("Loading SWE-Bench_Verified dataset...")

        # Load from HuggingFace
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")[self.split]
        instances = list(ds)

        # Filter by specific instance IDs if provided
        if self.subset:
            instances = [i for i in instances if i["instance_id"] in self.subset]
            logger.info(f"Filtered to {len(instances)} instances by subset")

        # Filter by repos if provided
        if self.repos:
            instances = [i for i in instances if i["repo"] in self.repos]
            logger.info(f"Filtered to {len(instances)} instances by repos: {self.repos}")

        # Shuffle if requested
        if self.shuffle_enabled:
            random.seed(self.seed + self.epoch_id)
            random.shuffle(instances)

        # Limit if requested
        if self.limit:
            instances = instances[:self.limit]
            logger.info(f"Limited to {len(instances)} instances")

        logger.info(f"Loaded {len(instances)} SWE-Bench_Verified instances")
        return instances

    @property
    def instances(self) -> list[dict]:
        """Lazy load instances."""
        if self._instances is None:
            self._instances = self._load_instances()
        return self._instances

    def __len__(self) -> int:
        return len(self.instances)

    def _instance_to_sample(self, instance: dict, index: int) -> Sample:
        """Convert a dataset instance to a Sample."""
        # Format the prompt
        prompt = BUG_SOLVING_PROMPT.format(
            repo=instance["repo"],
            version=instance["version"],
            base_commit=instance["base_commit"],
            problem_statement=instance["problem_statement"],
        )

        return Sample(
            index=index,
            prompt=prompt,
            metadata={
                "instance_id": instance["instance_id"],
                "repo": instance["repo"],
                "version": instance["version"],
                "base_commit": instance["base_commit"],
                "patch": instance.get("patch", ""),
                "test_patch": instance.get("test_patch", ""),
            },
        )

    def __getitem__(self, index: int) -> Sample:
        """Get a sample by index."""
        instance = self.instances[index]
        return self._instance_to_sample(instance, index)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples sample groups.

        Each group contains n_samples_per_prompt copies of the same prompt.
        """
        if self.sample_offset + num_samples > len(self.instances):
            # Wrap around to beginning
            prompt_instances = self.instances[self.sample_offset:]
            remaining = num_samples - len(prompt_instances)

            # Increment epoch and reshuffle
            self.epoch_id += 1
            if self.shuffle_enabled:
                random.seed(self.seed + self.epoch_id)
                random.shuffle(self._instances)

            prompt_instances += self.instances[:remaining]
            self.sample_offset = remaining
        else:
            prompt_instances = self.instances[self.sample_offset:self.sample_offset + num_samples]
            self.sample_offset += num_samples

        samples = []
        for instance in prompt_instances:
            group = []
            for _ in range(self.n_samples_per_prompt):
                sample = self._instance_to_sample(instance, self.sample_index)
                sample.group_index = self.sample_group_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)

        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """Add samples to the data source (not supported for read-only source)."""
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        """Save the state of the data source."""
        if self.args is None or getattr(self.args, "save", None) is None:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/swebench_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        """Load the state of the data source."""
        if self.args is None or getattr(self.args, "load", None) is None:
            return

        path = os.path.join(self.args.load, f"rollout/swebench_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"Loading state from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.shuffle_enabled:
            random.seed(self.seed + self.epoch_id)
            random.shuffle(self._instances)

    def get_instance_by_id(self, instance_id: str) -> dict | None:
        """Get a specific instance by ID."""
        for instance in self.instances:
            if instance["instance_id"] == instance_id:
                return instance
        return None


class DjangoTrainDataSource(SWEBenchVerifiedDataSource):
    """
    DataSource for Django training instances from SWE-Bench_Verified.

    Focuses on django/django repository for targeted training.
    Uses the 201 train instances from train_instances_id.txt.
    """

    def __init__(self, args=None, limit: int | None = None, **kwargs):
        # Load train instance IDs from file
        train_ids_file = Path(__file__).parent.parent.parent / "train_instances_id.txt"
        train_ids = None
        if train_ids_file.exists():
            with open(train_ids_file) as f:
                train_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(train_ids)} train instance IDs from {train_ids_file}")

        # Use django/django repo only with train subset
        super().__init__(
            args=args,
            repos=["django/django"],
            subset=train_ids,
            limit=limit,
            **kwargs,
        )


class DjangoTestDataSource(SWEBenchVerifiedDataSource):
    """
    DataSource for Django test instances from SWE-Bench_Verified.

    Uses the 30 test instances from test_instances_id.txt.
    """

    def __init__(self, args=None, limit: int | None = None, **kwargs):
        # Load test instance IDs from file
        test_ids_file = Path(__file__).parent.parent.parent / "test_instances_id.txt"
        test_ids = None
        if test_ids_file.exists():
            with open(test_ids_file) as f:
                test_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(test_ids)} test instance IDs from {test_ids_file}")

        # Use django/django repo only with test subset
        super().__init__(
            args=args,
            repos=["django/django"],
            subset=test_ids,
            limit=limit,
            shuffle=False,  # Don't shuffle test data
            **kwargs,
        )


def create_data_source(
    args=None,
    dataset_type: str = "swebench_verified",
    **kwargs,
) -> DataSource:
    """
    Factory function to create data sources.

    Args:
        args: Training arguments
        dataset_type: Type of dataset ("swebench_verified", "django_train", "django_test")
        **kwargs: Additional arguments passed to data source

    Returns:
        DataSource instance
    """
    if dataset_type == "swebench_verified":
        return SWEBenchVerifiedDataSource(args=args, **kwargs)
    elif dataset_type == "django_train":
        return DjangoTrainDataSource(args=args, **kwargs)
    elif dataset_type == "django_test":
        return DjangoTestDataSource(args=args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == "__main__":
    # Quick test
    ds = DjangoTrainDataSource(limit=5)
    print(f"Loaded {len(ds)} instances")

    # Test get_samples
    samples = ds.get_samples(2)
    print(f"Got {len(samples)} sample groups")
    for i, group in enumerate(samples):
        print(f"Group {i}: {len(group)} samples, instance_id={group[0].metadata['instance_id']}")
