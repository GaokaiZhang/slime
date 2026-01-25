"""
C2Bug Data Adapter for Harbor GRPO Training

This module provides functions to load and convert c2bug data from HuggingFace
into Harbor task format for training.

Compatible with:
- TwelfthStar/c2bug_tasks_django_Jan-22-2026 dataset
- Original SWE-bench_Verified data loading (unchanged)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Optional, Tuple, Union


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def render_literal(template_text: str, **repls: str) -> str:
    out = template_text
    for key, value in repls.items():
        out = out.replace("{" + key + "}", value)
    return out


@dataclass
class C2BugRecord:
    task_uid: str
    instance_id: str
    repo: str
    commit: str
    issue_variant: str
    issue_text: str
    bug_patch: str
    failed_test_command: str

    @classmethod
    def from_dict(cls, data: dict) -> "C2BugRecord":
        return cls(
            task_uid=data.get("task_uid") or data.get("instance_id") or "",
            instance_id=data.get("instance_id") or data.get("task_uid") or "",
            repo=data.get("repo") or "",
            commit=data.get("commit") or "",
            issue_variant=data.get("issue_variant") or "",
            issue_text=data.get("issue_text") or "",
            bug_patch=data.get("bug_patch") or "",
            failed_test_command=data.get("failed_test_command") or "",
        )


class C2BugLoader:
    """Load c2bug data from JSON file or HuggingFace dataset."""

    def __init__(self, collection: Union[Path, dict, list]) -> None:
        if isinstance(collection, Path):
            payload = json.loads(collection.read_text())
        elif isinstance(collection, dict):
            payload = collection
        elif isinstance(collection, list):
            payload = {"run_meta": {}, "records": collection}
        else:
            raise TypeError("collection must be a Path, dict, or list")

        self._run_meta = payload.get("run_meta") or {}
        self._records = payload.get("records") or []

    @property
    def run_meta(self) -> dict:
        return self._run_meta

    def apply_run_meta(self, overrides: dict) -> None:
        if overrides:
            self._run_meta = {**self._run_meta, **overrides}

    def all_records(self) -> list[dict]:
        return list(self._records)

    def iter_records(
        self,
        *,
        include_all: bool = False,
    ) -> Iterable[C2BugRecord]:
        for record in self._records:
            if not include_all:
                if not (record.get("issue_text") or "").strip():
                    continue
                if not (record.get("bug_patch") or "").strip():
                    continue
                if not (record.get("failed_test_command") or "").strip():
                    continue
            yield C2BugRecord.from_dict(record)


class HarborTaskPaths:
    """Manages paths for a Harbor task directory structure."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = Path(task_dir)
        self.environment_dir = self.task_dir / "environment"
        self.tests_dir = self.task_dir / "tests"
        self.solution_dir = self.task_dir / "solution"

        self.instruction_path = self.task_dir / "instruction.md"
        self.config_path = self.task_dir / "task.toml"
        self.dockerfile_path = self.environment_dir / "Dockerfile"
        self.bug_patch_path = self.environment_dir / "bug.patch"
        self.test_sh_path = self.tests_dir / "test.sh"
        self.config_json_path = self.tests_dir / "config.json"
        self.solution_patch_path = self.solution_dir / "solution.patch"
        self.solve_sh_path = self.solution_dir / "solve.sh"

        self.environment_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)


def reverse_patch(patch_text: str) -> str:
    """
    Reverse a git diff patch (swap + and - lines in diff body).

    For c2bug, the bug_patch introduces the bug.
    Reversing it gives us the fix.

    Note: --- and +++ headers stay as-is, only the diff body
    (+/- lines) and @@ line counts are swapped.
    """
    import re

    lines = patch_text.split("\n")
    reversed_lines = []

    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            # Keep file headers unchanged
            reversed_lines.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            # Swap - to + (deleted becomes added)
            reversed_lines.append("+" + line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            # Swap + to - (added becomes deleted)
            reversed_lines.append("-" + line[1:])
        elif line.startswith("@@"):
            # Swap line numbers in hunk header: @@ -old,count +new,count @@
            # e.g., @@ -1,9 +1,41 @@ -> @@ -1,41 +1,9 @@
            match = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)", line)
            if match:
                old_start, old_count, new_start, new_count, rest = match.groups()
                old_count = old_count or "1"
                new_count = new_count or "1"
                reversed_lines.append(f"@@ -{new_start},{new_count} +{old_start},{old_count} @@{rest}")
            else:
                reversed_lines.append(line)
        else:
            reversed_lines.append(line)

    return "\n".join(reversed_lines)


class C2BugToHarbor:
    """Convert c2bug records to Harbor task directories."""

    def __init__(
        self,
        collection_source: Union[Path, dict, list],
        task_root: Path,
        *,
        template_dir: Optional[Path] = None,
        max_timeout_sec: float = 3000.0,
        include_all: bool = False,
        run_meta_override: Optional[dict] = None,
    ) -> None:
        self.loader = C2BugLoader(collection_source)
        self.task_root = Path(task_root)
        self.task_root.mkdir(parents=True, exist_ok=True)

        # Default template dir is alongside this file
        self.template_dir = Path(template_dir or (Path(__file__).parent / "c2bug_template"))
        self.include_all = include_all
        self.max_timeout = max_timeout_sec

        self.t_instruction = self.template_dir / "instruction.md"
        self.t_config = self.template_dir / "task.toml"
        self.t_test_sh = self.template_dir / "test.sh"
        self.t_dockerfile = self.template_dir / "Dockerfile"

        if run_meta_override:
            self.loader.apply_run_meta(run_meta_override)

    def generate_task(self, record: C2BugRecord, task_name: str, *, overwrite: bool) -> Path:
        task_dir = self.task_root / task_name
        if task_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Target already exists: {task_dir}")
            shutil.rmtree(task_dir)

        paths = HarborTaskPaths(task_dir)
        run_meta = self.loader.run_meta

        instruction_tpl = read_text(self.t_instruction)
        version = run_meta.get("swebench_version") or run_meta.get("version") or "unknown"
        base_commit = run_meta.get("base_commit") or record.commit or "unknown"
        instance_id = record.task_uid or record.instance_id
        instruction = render_literal(
            instruction_tpl,
            problem_statement=dedent(record.issue_text).strip(),
            repo=record.repo,
            version=str(version),
            base_commit=str(base_commit),
            instance_id=instance_id,
        )
        if not instruction.endswith("\n"):
            instruction += "\n"
        paths.instruction_path.write_text(instruction)

        config_tpl = read_text(self.t_config)
        config = render_literal(
            config_tpl,
            max_timeout=str(int(self.max_timeout)),
        )
        paths.config_path.write_text(config)

        dockerfile_tpl = read_text(self.t_dockerfile)
        dockerfile = render_literal(
            dockerfile_tpl,
            docker_image=run_meta.get("docker_image", ""),
            workdir=run_meta.get("workdir", "/testbed"),
        )
        paths.dockerfile_path.write_text(dockerfile)

        bug_patch = record.bug_patch.rstrip()
        if bug_patch and not bug_patch.endswith("\n"):
            bug_patch += "\n"
        paths.bug_patch_path.write_text(bug_patch)

        # Create solution patch (reverse of bug_patch) for oracle agent
        if bug_patch.strip():
            solution_patch = reverse_patch(bug_patch)
            if solution_patch and not solution_patch.endswith("\n"):
                solution_patch += "\n"
            paths.solution_patch_path.write_text(solution_patch)

            # Create solve.sh for oracle agent
            workdir = run_meta.get("workdir", "/testbed")
            solve_sh = f"""#!/bin/bash
set -e
cd {workdir}
git apply /solution/solution.patch
"""
            paths.solve_sh_path.write_text(solve_sh)
            paths.solve_sh_path.chmod(0o755)

        test_sh_tpl = read_text(self.t_test_sh)
        test_sh = render_literal(test_sh_tpl)
        paths.test_sh_path.write_text(test_sh)
        paths.test_sh_path.chmod(0o755)

        config_payload = {
            "task_uid": record.task_uid,
            "instance_id": record.instance_id,
            "repo": record.repo,
            "commit": record.commit,
            "issue_variant": record.issue_variant,
            "base_commit": base_commit,
            "version": version,
            "workdir": run_meta.get("workdir", "/testbed"),
            "bug_patch": record.bug_patch,
            "failed_test_command": record.failed_test_command,
        }
        paths.config_json_path.write_text(json.dumps(config_payload, indent=2) + "\n")
        return paths.task_dir

    def generate_many(
        self,
        *,
        limit: Optional[int] = None,
        overwrite: bool = False,
    ) -> Tuple[list[Path], list[tuple[str, str]]]:
        success: list[Path] = []
        failures: list[tuple[str, str]] = []
        records = list(self.loader.iter_records(include_all=self.include_all))
        if limit is not None:
            records = records[:limit]

        for idx, record in enumerate(records, 1):
            task_name = record.task_uid or record.instance_id or f"c2bug_{idx}"
            try:
                out = self.generate_task(record, task_name, overwrite=overwrite)
                print(f"[{idx}] OK   {task_name} -> {out}")
                success.append(out)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"[{idx}] FAIL {task_name}: {msg}")
                failures.append((task_name, msg))
        return success, failures


def load_c2bug_from_hf(
    dataset_id: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    *,
    split: str = "train",
    config: str | None = None,
    token: str | None = None,
    collection_filename: str = "collection.json",
) -> dict:
    """Load c2bug data from HuggingFace dataset."""
    payload: dict | None = None

    # Try to download collection.json first (has run_meta)
    if collection_filename:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=collection_filename,
                token=token,
            )
            payload = json.loads(Path(downloaded).read_text())
        except Exception as exc:
            print(f"warning: failed to download {collection_filename} from HF: {exc}")

    if payload is not None:
        return payload

    # Fall back to datasets API
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load HF datasets") from exc

    kwargs: dict = {"split": split}
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    dataset = load_dataset(dataset_id, **kwargs)
    return {"run_meta": {}, "records": dataset.to_list()}


def load_c2bug_instances(
    dataset_id: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    task_root: str = "/tmp/c2bug_harbor_tasks",
    *,
    num_instances: int = 50,
    test_mode: bool = False,
    docker_image: str = "sweb.eval.x86_64.django_s_django-13810:latest",
    workdir: str = "/testbed",
    hf_token: str | None = None,
) -> list[dict]:
    """
    Load c2bug instances and convert to Harbor format.

    Returns list of dicts with:
    - instance_id: unique task ID
    - task_dir: path to Harbor task directory
    - problem_statement: issue text for prompt
    - repo: repository name
    """
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    if test_mode:
        num_instances = min(5, num_instances)

    logger.info(f"Loading c2bug data from {dataset_id}...")
    collection = load_c2bug_from_hf(dataset_id, token=hf_token)

    # Apply run_meta overrides
    run_meta_override = {
        "docker_image": docker_image,
        "workdir": workdir,
    }

    task_root_path = Path(task_root)
    converter = C2BugToHarbor(
        collection_source=collection,
        task_root=task_root_path,
        max_timeout_sec=3000.0,
        include_all=False,
        run_meta_override=run_meta_override,
    )

    logger.info(f"Converting {num_instances} c2bug tasks to Harbor format...")
    success_paths, failures = converter.generate_many(
        limit=num_instances,
        overwrite=True,
    )

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
                "commit": record.commit,
                "failed_test_command": record.failed_test_command,
            })

    logger.info(f"Loaded {len(instances)} c2bug instances")
    return instances


def create_c2bug_prompt(instance: dict) -> str:
    """Create prompt for c2bug instance."""
    return f"""You are an expert software engineer. Fix this bug and provide the corrected code.

## Repository: {instance["repo"]}

## Problem Statement
{instance["problem_statement"][:4000]}

## Instructions
Analyze the problem and fix the issue. Make changes to the relevant source files.

Do NOT modify test files. Focus only on fixing the source code.

Begin by exploring the codebase to understand the issue, then make the necessary changes."""
