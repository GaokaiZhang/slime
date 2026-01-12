"""
Reward functions for SWE-bench evaluation.

This module provides reward computation based on swebench.harness evaluation.
Rewards are binary: +1 for resolved instances, -1 for failed.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from SWE-bench evaluation."""

    instance_id: str
    resolved: bool
    patch_applied: bool
    tests_passed: List[str]
    tests_failed: List[str]
    error: Optional[str] = None


def evaluate_patch_in_container(
    instance_id: str,
    patch: str,
    gold_patch: str,
    fail_to_pass: List[str],
    pass_to_pass: List[str],
    timeout: int = 900,  # 15 minutes
) -> EvalResult:
    """
    Evaluate a patch by running tests in a SWE-bench Docker container.

    This is a simplified evaluation that:
    1. Applies the model's patch to the container
    2. Runs the FAIL_TO_PASS tests
    3. Checks if they pass

    Args:
        instance_id: SWE-bench instance ID
        patch: Model-generated patch (git diff format)
        gold_patch: Gold patch (for reference, not applied)
        fail_to_pass: List of tests that should pass after fix
        pass_to_pass: List of tests that should still pass
        timeout: Evaluation timeout in seconds

    Returns:
        EvalResult with resolution status
    """
    if not patch or not patch.strip():
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
            tests_passed=[],
            tests_failed=[],
            error="Empty patch",
        )

    # Get docker image name
    id_docker = instance_id.replace("__", "_1776_").lower()
    image = f"swebench/sweb.eval.x86_64.{id_docker}:latest"
    container_name = f"swe_eval_{instance_id.replace('__', '_')}"

    try:
        # Remove existing container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
        )

        # Start container
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-w", "/testbed",
                image,
                "tail", "-f", "/dev/null"
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )

        # Apply patch
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_file = f.name

        subprocess.run(
            ["docker", "cp", patch_file, f"{container_name}:/tmp/model.patch"],
            capture_output=True,
            check=True,
        )
        os.unlink(patch_file)

        apply_result = subprocess.run(
            ["docker", "exec", container_name, "git", "apply", "/tmp/model.patch"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if apply_result.returncode != 0:
            return EvalResult(
                instance_id=instance_id,
                resolved=False,
                patch_applied=False,
                tests_passed=[],
                tests_failed=fail_to_pass,
                error=f"Patch apply failed: {apply_result.stderr}",
            )

        # Run tests
        tests_passed = []
        tests_failed = []

        for test in fail_to_pass:
            # Run individual test
            test_cmd = f"python -m pytest {test} -x --tb=no -q"
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-c", test_cmd],
                capture_output=True,
                text=True,
                timeout=timeout // max(len(fail_to_pass), 1),
            )

            if result.returncode == 0:
                tests_passed.append(test)
            else:
                tests_failed.append(test)

        resolved = len(tests_passed) == len(fail_to_pass) and len(tests_failed) == 0

        return EvalResult(
            instance_id=instance_id,
            resolved=resolved,
            patch_applied=True,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
        )

    except subprocess.TimeoutExpired:
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
            tests_passed=[],
            tests_failed=fail_to_pass,
            error="Evaluation timeout",
        )
    except Exception as e:
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
            tests_passed=[],
            tests_failed=fail_to_pass,
            error=str(e),
        )
    finally:
        # Cleanup container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
        )


def evaluate_with_swebench_harness(
    instance_id: str,
    patch: str,
    timeout: int = 900,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> EvalResult:
    """
    Evaluate a patch using the official swebench.harness.

    This method uses swebench.harness.run_evaluation for full evaluation,
    matching the approach used in SWE-sft.

    Args:
        instance_id: SWE-bench instance ID
        patch: Model-generated patch
        timeout: Evaluation timeout per instance
        dataset_name: HuggingFace dataset name
        split: Dataset split

    Returns:
        EvalResult with resolution status
    """
    if not patch or not patch.strip():
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
            tests_passed=[],
            tests_failed=[],
            error="Empty patch",
        )

    run_id = f"eval_{instance_id.replace('__', '_')}_{os.getpid()}_{os.urandom(4).hex()}"
    pred_file = None

    try:
        # Write predictions to temp file in the expected format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            pred = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "qwen_swe",
            }
            f.write(json.dumps(pred) + "\n")
            pred_file = f.name

        # Run swebench evaluation (matching SWE-sft approach)
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset_name,
            "--split", split,
            "--instance_ids", instance_id,
            "--predictions_path", pred_file,
            "--max_workers", "1",
            "--timeout", str(timeout),
            "--run_id", run_id,
        ]

        logger.info(f"[{instance_id}] Running swebench harness evaluation...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 600,  # Extra time for setup/teardown
        )

        # Parse results from log directory (matching SWE-sft pattern)
        log_dir = Path(f"logs/run_evaluation/{run_id}")
        if log_dir.exists():
            # Look for per-instance report.json files
            for report_path in log_dir.rglob("report.json"):
                try:
                    with open(report_path) as f:
                        report = json.load(f)

                    # Report format: {instance_id: {resolved: bool, ...}}
                    if isinstance(report, dict) and instance_id in report:
                        entry = report[instance_id]
                        tests_status = entry.get("tests_status", {})
                        fail_to_pass = tests_status.get("FAIL_TO_PASS", {})

                        return EvalResult(
                            instance_id=instance_id,
                            resolved=entry.get("resolved", False),
                            patch_applied=entry.get("patch_successfully_applied", False),
                            tests_passed=fail_to_pass.get("success", []),
                            tests_failed=fail_to_pass.get("failure", []),
                        )
                    elif isinstance(report, dict) and len(report) == 1:
                        # Single entry format
                        inst_id, entry = next(iter(report.items()))
                        tests_status = entry.get("tests_status", {})
                        fail_to_pass = tests_status.get("FAIL_TO_PASS", {})

                        return EvalResult(
                            instance_id=instance_id,
                            resolved=entry.get("resolved", False),
                            patch_applied=entry.get("patch_successfully_applied", False),
                            tests_passed=fail_to_pass.get("success", []),
                            tests_failed=fail_to_pass.get("failure", []),
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"[{instance_id}] Failed to parse report: {e}")
                    continue

        # Check evaluation output for resolution status
        resolved = (
            "1 resolved" in result.stdout or
            "resolved: 1" in result.stdout or
            f'"{instance_id}": {{"resolved": true' in result.stdout
        )

        if result.returncode != 0:
            logger.warning(f"[{instance_id}] Harness returned non-zero: {result.stderr[:500]}")

        return EvalResult(
            instance_id=instance_id,
            resolved=resolved,
            patch_applied="patch_successfully_applied" not in result.stdout or
                         "patch_successfully_applied: true" in result.stdout.lower(),
            tests_passed=[],
            tests_failed=[],
        )

    except subprocess.TimeoutExpired:
        logger.warning(f"[{instance_id}] Evaluation timeout after {timeout}s")
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
            tests_passed=[],
            tests_failed=[],
            error="Evaluation timeout",
        )
    except Exception as e:
        logger.error(f"[{instance_id}] Evaluation error: {e}")
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
            tests_passed=[],
            tests_failed=[],
            error=str(e),
        )
    finally:
        # Cleanup temp file
        if pred_file and os.path.exists(pred_file):
            try:
                os.unlink(pred_file)
            except:
                pass
        # Cleanup log directory to save disk space
        log_dir = Path(f"logs/run_evaluation/{run_id}")
        if log_dir.exists():
            try:
                import shutil
                shutil.rmtree(log_dir)
            except:
                pass


def compute_reward(eval_result: EvalResult) -> float:
    """
    Compute reward from evaluation result.

    Returns:
        +1.0 if resolved
        -1.0 if failed (including no patch)
    """
    return 1.0 if eval_result.resolved else -1.0


async def swebench_reward(
    args,
    samples: Union[Sample, List[Sample]],
    **kwargs,
) -> Union[float, List[float]]:
    """
    Async reward function for SLiME integration.

    This function evaluates model patches using SWE-bench harness
    and returns rewards.

    Args:
        args: Training arguments
        samples: Sample or list of samples with patches to evaluate

    Returns:
        Reward(s) for the sample(s)
    """
    if isinstance(samples, Sample):
        samples = [samples]
        single = True
    else:
        single = False

    rewards = []

    for sample in samples:
        # Extract patch from sample
        patch = sample.metadata.get("patch", "")

        if not patch:
            # Try to extract from response
            patch = sample.response if hasattr(sample, "response") else ""

        instance_id = sample.metadata.get("instance_id", "unknown")
        fail_to_pass = sample.metadata.get("fail_to_pass", [])
        pass_to_pass = sample.metadata.get("pass_to_pass", [])
        gold_patch = sample.metadata.get("gold_patch", "")

        # Choose evaluation method based on args
        use_harness = getattr(args, "use_swebench_harness", False)

        if use_harness:
            eval_result = evaluate_with_swebench_harness(
                instance_id=instance_id,
                patch=patch,
                timeout=getattr(args, "eval_timeout", 900),
            )
        else:
            eval_result = evaluate_patch_in_container(
                instance_id=instance_id,
                patch=patch,
                gold_patch=gold_patch,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                timeout=getattr(args, "eval_timeout", 900),
            )

        reward = compute_reward(eval_result)
        rewards.append(reward)

        # Store evaluation details in metadata
        sample.metadata["eval_result"] = {
            "resolved": eval_result.resolved,
            "patch_applied": eval_result.patch_applied,
            "tests_passed": eval_result.tests_passed,
            "tests_failed": eval_result.tests_failed,
            "error": eval_result.error,
        }

        logger.info(
            f"[{instance_id}] Reward: {reward}, Resolved: {eval_result.resolved}, "
            f"Tests passed: {len(eval_result.tests_passed)}/{len(fail_to_pass)}"
        )

    if single:
        return rewards[0]
    return rewards


# Synchronous wrapper for compatibility
def swebench_reward_sync(
    args,
    sample: Sample,
    **kwargs,
) -> float:
    """Synchronous version of swebench_reward."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(swebench_reward(args, sample, **kwargs))
    finally:
        loop.close()
