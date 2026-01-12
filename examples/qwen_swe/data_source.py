"""
Data source for loading SWE-bench Django instances.

This module provides data loading utilities for GRPO training on
SWE-bench instances, extending SLiME's DataSource base class.
"""

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchInstance:
    """A single SWE-bench instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str  # Gold patch (the fix)
    test_patch: str  # Test file changes
    hints_text: str
    created_at: str
    version: str
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SWEBenchInstance":
        """Create instance from dictionary."""
        fail_to_pass = data.get("FAIL_TO_PASS", [])
        pass_to_pass = data.get("PASS_TO_PASS", [])

        # Handle string vs list
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass) if fail_to_pass else []
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass) if pass_to_pass else []

        return cls(
            instance_id=data["instance_id"],
            repo=data.get("repo", ""),
            base_commit=data.get("base_commit", ""),
            problem_statement=data.get("problem_statement", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            hints_text=data.get("hints_text", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )

    def get_docker_image(self) -> str:
        """Get the SWE-bench docker image name."""
        id_docker = self.instance_id.replace("__", "_1776_").lower()
        return f"swebench/sweb.eval.x86_64.{id_docker}:latest"


def load_instances_from_ids(
    instance_ids: List[str],
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> List[SWEBenchInstance]:
    """
    Load SWE-bench instances by ID from HuggingFace dataset.

    Args:
        instance_ids: List of instance IDs to load
        dataset_name: HuggingFace dataset name
        split: Dataset split ("test" or "dev")

    Returns:
        List of SWEBenchInstance objects
    """
    from datasets import load_dataset

    logger.info(f"Loading {len(instance_ids)} instances from {dataset_name}")

    ds = load_dataset(dataset_name, split=split)

    # Create lookup by instance_id
    ds_dict = {item["instance_id"]: item for item in ds}

    instances = []
    for iid in instance_ids:
        if iid in ds_dict:
            instances.append(SWEBenchInstance.from_dict(ds_dict[iid]))
        else:
            logger.warning(f"Instance {iid} not found in dataset, skipping")

    logger.info(f"Loaded {len(instances)} instances")
    return instances


def load_instances_from_file(
    file_path: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> List[SWEBenchInstance]:
    """
    Load SWE-bench instances from a file containing instance IDs.

    Args:
        file_path: Path to file with instance IDs (one per line)
        dataset_name: HuggingFace dataset name
        split: Dataset split

    Returns:
        List of SWEBenchInstance objects
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Instance ID file not found: {file_path}")

    instance_ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                instance_ids.append(line)

    return load_instances_from_ids(instance_ids, dataset_name, split)


def load_django_instances(
    split_type: str = "all",  # "all", "train", "test", "part_a", "part_b"
    data_dir: str = "/home/gaokaizhang/SWE-sft/data/raw/splits",
) -> List[SWEBenchInstance]:
    """
    Load Django SWE-bench instances from predefined splits.

    Args:
        split_type: Type of split to load
        data_dir: Directory containing split files

    Returns:
        List of SWEBenchInstance objects
    """
    split_files = {
        "all": "all_231_django.txt",
        "part_a": "part_a_116.txt",
        "part_b": "part_b_115.txt",
    }

    if split_type not in split_files:
        raise ValueError(f"Unknown split type: {split_type}. Choose from {list(split_files.keys())}")

    file_path = os.path.join(data_dir, split_files[split_type])
    return load_instances_from_file(file_path)


class SWEBenchDataSource(DataSource):
    """
    Data source for SLiME GRPO training on SWE-bench instances.

    Extends SLiME's DataSource base class to provide samples for
    the rollout generation process.
    """

    def __init__(self, args):
        """
        Initialize data source from args.

        Args:
            args: Namespace with configuration including:
                - swe_instance_file: Path to file with instance IDs
                - swe_split: Split type (all, part_a, part_b)
                - n_samples_per_prompt: Number of samples per prompt (group size)
                - rollout_shuffle: Whether to shuffle instances
        """
        self.args = args
        self.n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 8)
        self.shuffle = getattr(args, "rollout_shuffle", True)

        # Load instances based on configuration
        if hasattr(args, "swe_instance_file") and args.swe_instance_file:
            self.instances = load_instances_from_file(args.swe_instance_file)
        elif hasattr(args, "swe_split") and args.swe_split:
            self.instances = load_django_instances(args.swe_split)
        else:
            # Default: use train split
            default_file = "/home/gaokaizhang/SWE-sft/data/raw/splits/train_201_django.txt"
            if os.path.exists(default_file):
                self.instances = load_instances_from_file(default_file)
            else:
                self.instances = load_django_instances("all")

        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0

        if self.shuffle:
            import random
            random.shuffle(self.instances)

        logger.info(
            f"Initialized SWEBenchDataSource with {len(self.instances)} instances, "
            f"{self.n_samples_per_prompt} samples per prompt"
        )

    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """
        Get a batch of sample groups for rollout.

        Args:
            num_samples: Number of groups to return

        Returns:
            List of sample groups (each group has n_samples_per_prompt samples)
        """
        from .prompts import format_swebench_prompt

        groups = []
        for _ in range(num_samples):
            if self.sample_offset >= len(self.instances):
                self.epoch_id += 1
                if self.shuffle:
                    import random
                    random.shuffle(self.instances)
                self.sample_offset = 0

            instance = self.instances[self.sample_offset]
            self.sample_offset += 1

            # Create group of samples for this instance
            group = []
            prompt = format_swebench_prompt(instance.problem_statement)

            for i in range(self.n_samples_per_prompt):
                sample = Sample(
                    prompt=prompt,
                    group_index=self.sample_group_index,
                    index=self.sample_index,
                    metadata={
                        "instance_id": instance.instance_id,
                        "repo": instance.repo,
                        "base_commit": instance.base_commit,
                        "gold_patch": instance.patch,
                        "fail_to_pass": instance.fail_to_pass,
                        "pass_to_pass": instance.pass_to_pass,
                    },
                )
                group.append(sample)
                self.sample_index += 1

            self.sample_group_index += 1
            groups.append(group)

        return groups

    def add_samples(self, samples: List[List[Sample]]) -> None:
        """Add partial samples back to the buffer (for abort recovery)."""
        # For now, we don't need to handle partial samples
        pass

    def save(self, rollout_id: int) -> None:
        """Save the state of the data source."""
        import torch
        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
        }
        if hasattr(self.args, "save") and self.args.save:
            path = os.path.join(self.args.save, f"rollout/swebench_data_source_{rollout_id}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)

    def load(self, rollout_id: int = None) -> None:
        """Load the state of the data source."""
        import torch
        if not hasattr(self.args, "load") or not self.args.load:
            return

        path = os.path.join(self.args.load, f"rollout/swebench_data_source_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"Loading data source state from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)

        if self.shuffle:
            import random
            random.seed(self.epoch_id)
            random.shuffle(self.instances)


def create_data_source(args) -> SWEBenchDataSource:
    """
    Create data source from command-line arguments.

    This function is called by SLiME's rollout manager to create the data source.
    Path: examples.qwen_swe.data_source:create_data_source

    Args:
        args: Namespace with configuration

    Returns:
        SWEBenchDataSource instance
    """
    return SWEBenchDataSource(args)
