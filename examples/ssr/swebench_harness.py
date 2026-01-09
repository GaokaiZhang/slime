"""
SWE-bench Harness Integration for SSR.

Provides integration with swebench.harness for:
- Test specification creation
- Evaluation grading
- Docker image management

Usage:
    from examples.ssr.swebench_harness import (
        evaluate_patch,
        get_test_spec,
        run_tests_in_container,
    )

    # Evaluate a solver's patch
    result = evaluate_patch(
        instance_id="django__django-17087",
        patch="...",
    )
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a patch."""

    instance_id: str
    resolved: bool
    tests_passed: int
    tests_failed: int
    tests_error: int
    test_output: str
    error_message: str | None = None


def get_swebench_instance(instance_id: str, dataset_name: str = "princeton-nlp/SWE-bench_Verified") -> dict | None:
    """
    Load SWE-bench instance data.

    Args:
        instance_id: Instance ID (e.g., "django__django-17087")
        dataset_name: HuggingFace dataset name

    Returns:
        Instance data dict or None if not found
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="test")

        for item in dataset:
            if item["instance_id"] == instance_id:
                return item

        logger.warning(f"Instance {instance_id} not found in {dataset_name}")
        return None

    except Exception as e:
        logger.error(f"Failed to load SWE-bench instance: {e}")
        return None


def get_test_spec_for_instance(instance_id: str) -> dict | None:
    """
    Get test specification for an instance.

    Args:
        instance_id: Instance ID

    Returns:
        Test specification dict
    """
    try:
        from swebench.harness.test_spec.test_spec import TestSpec
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        # Parse instance ID
        parts = instance_id.split("__")
        if len(parts) != 2:
            return None

        repo = parts[0].replace("_", "/")

        # Try to find matching spec
        for repo_key, versions in MAP_REPO_VERSION_TO_SPECS.items():
            if repo.lower() in repo_key.lower():
                # Return first matching version's test spec
                if versions:
                    return {
                        "repo": repo,
                        "instance_id": instance_id,
                        "test_cmd": versions.get("test_cmd", "pytest"),
                    }

        return None

    except ImportError as e:
        logger.warning(f"swebench.harness not fully available: {e}")
        return None


def run_tests_in_container(
    instance_id: str,
    patch: str,
    test_script: str | None = None,
    timeout: int = 300,
) -> EvalResult:
    """
    Run tests in a SWE-bench docker container.

    Args:
        instance_id: SWE-bench instance ID
        patch: Patch to apply and test
        test_script: Optional custom test script
        timeout: Timeout in seconds

    Returns:
        EvalResult with test results
    """
    from .docker_sandbox import DockerSandbox, DockerSandboxConfig

    config = DockerSandboxConfig(timeout=timeout)

    try:
        with DockerSandbox(config) as sandbox:
            # Start container for this instance
            sandbox.start(instance_id=instance_id)

            # Apply patch
            if patch.strip():
                success = sandbox.apply_patch(patch)
                if not success:
                    return EvalResult(
                        instance_id=instance_id,
                        resolved=False,
                        tests_passed=0,
                        tests_failed=0,
                        tests_error=1,
                        test_output="",
                        error_message="Failed to apply patch",
                    )

            # Run tests
            if test_script:
                test_output = sandbox.run_tests(test_script, timeout=timeout - 30)
            else:
                # Default test command
                test_output = sandbox.exec_command(
                    "python -m pytest --tb=short -q 2>&1 | head -100",
                    timeout=timeout - 30,
                )

            # Parse results
            passed, failed, error = parse_test_output(test_output)

            return EvalResult(
                instance_id=instance_id,
                resolved=(failed == 0 and error == 0 and passed > 0),
                tests_passed=passed,
                tests_failed=failed,
                tests_error=error,
                test_output=test_output,
            )

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return EvalResult(
            instance_id=instance_id,
            resolved=False,
            tests_passed=0,
            tests_failed=0,
            tests_error=1,
            test_output="",
            error_message=str(e),
        )


def parse_test_output(output: str) -> tuple[int, int, int]:
    """
    Parse pytest output to extract test counts.

    Args:
        output: Raw pytest output

    Returns:
        Tuple of (passed, failed, error)
    """
    import re

    passed = 0
    failed = 0
    error = 0

    # Look for pytest summary line: "X passed, Y failed, Z error"
    summary_pattern = r"(\d+)\s+(passed|failed|error)"
    for match in re.finditer(summary_pattern, output.lower()):
        count = int(match.group(1))
        status = match.group(2)
        if status == "passed":
            passed = count
        elif status == "failed":
            failed = count
        elif status == "error":
            error = count

    # Fallback: count PASSED/FAILED lines
    if passed == 0 and failed == 0:
        passed = output.count("PASSED")
        failed = output.count("FAILED")
        error = output.count("ERROR")

    return passed, failed, error


def evaluate_patch(
    instance_id: str,
    patch: str,
    run_id: str = "ssr_eval",
) -> EvalResult:
    """
    Evaluate a patch using SWE-bench harness.

    This is the main evaluation function that:
    1. Sets up the test environment
    2. Applies the patch
    3. Runs tests
    4. Returns evaluation result

    Args:
        instance_id: SWE-bench instance ID
        patch: Patch to evaluate
        run_id: Run identifier for logging

    Returns:
        EvalResult with evaluation outcome
    """
    logger.info(f"Evaluating patch for {instance_id}")

    # Try to use swebench.harness if available
    try:
        from swebench.harness.run_evaluation import main as run_swebench_eval
        from swebench.harness.test_spec.test_spec import TestSpec

        # This requires a full SWE-bench setup
        # For now, fall back to our docker-based evaluation
        pass

    except ImportError:
        pass

    # Use our docker-based evaluation
    return run_tests_in_container(instance_id, patch)


def get_instance_docker_image(instance_id: str) -> str:
    """
    Get the docker image name for a SWE-bench instance.

    Args:
        instance_id: Instance ID (e.g., "django__django-17087")

    Returns:
        Docker image name
    """
    # Parse instance ID
    parts = instance_id.split("__")
    if len(parts) != 2:
        return "swebench/sweb.eval.x86_64.django_1776:latest"

    repo = parts[0].lower()
    issue = parts[1].lower()

    # SWE-bench image naming: swebench/sweb.eval.x86_64.{repo}_{version}_{instance}
    # For now, return a generic pattern
    base_image = f"swebench/sweb.eval.x86_64.{repo}"

    # Try to find matching image
    try:
        import docker
        client = docker.from_env()
        images = client.images.list()

        for img in images:
            for tag in img.tags:
                if base_image in tag and issue in tag:
                    return tag

    except:
        pass

    # Default fallback
    return f"swebench/sweb.eval.x86_64.django_1776_django-17087:latest"


def compute_reward_from_eval(eval_result: EvalResult) -> float:
    """
    Compute SSR reward from evaluation result.

    Args:
        eval_result: Evaluation result

    Returns:
        Reward value (+1 for resolved, -1 otherwise)
    """
    if eval_result.resolved:
        return 1.0
    else:
        return -1.0


# SSR-specific evaluation functions

def evaluate_bug_artifact(
    bug_patch: str,
    test_patch: str,
    test_script: str,
    instance_id: str,
) -> dict[str, Any]:
    """
    Evaluate a bug artifact for validity.

    Args:
        bug_patch: Patch that introduces the bug
        test_patch: Patch that weakens tests
        test_script: Test execution script
        instance_id: SWE-bench instance ID

    Returns:
        Dict with validation results
    """
    from .docker_sandbox import DockerSandbox, DockerSandboxConfig

    results = {
        "valid": False,
        "tests_pass_before_bug": False,
        "tests_fail_after_bug": False,
        "tests_pass_after_weakening": False,
        "errors": [],
    }

    config = DockerSandboxConfig(timeout=300)

    try:
        with DockerSandbox(config) as sandbox:
            sandbox.start(instance_id=instance_id)

            # 1. Check tests pass on clean repo
            test_output = sandbox.run_tests(test_script, timeout=90)
            passed, failed, error = parse_test_output(test_output)

            if passed > 0 and failed == 0 and error == 0:
                results["tests_pass_before_bug"] = True
            else:
                results["errors"].append(f"Tests don't pass on clean repo: {passed}p/{failed}f/{error}e")

            # 2. Apply bug patch and verify tests fail
            sandbox.reset()
            sandbox.apply_patch(bug_patch)
            test_output = sandbox.run_tests(test_script, timeout=90)
            passed, failed, error = parse_test_output(test_output)

            if failed > 0 or error > 0:
                results["tests_fail_after_bug"] = True
            else:
                results["errors"].append("Tests still pass after bug injection")

            # 3. Apply test weakening patch and verify some tests pass
            sandbox.apply_patch(test_patch)
            test_output = sandbox.run_tests(test_script, timeout=90)
            passed, failed, error = parse_test_output(test_output)

            if passed > 0:
                results["tests_pass_after_weakening"] = True
            else:
                results["errors"].append("No tests pass after test weakening")

            # Overall validity
            results["valid"] = (
                results["tests_pass_before_bug"] and
                results["tests_fail_after_bug"] and
                results["tests_pass_after_weakening"]
            )

    except Exception as e:
        results["errors"].append(f"Evaluation error: {str(e)}")

    return results


def evaluate_solver_patch(
    solver_patch: str,
    bug_patch: str,
    test_patch: str,
    test_script: str,
    instance_id: str,
) -> dict[str, Any]:
    """
    Evaluate a solver's patch for correctness.

    Args:
        solver_patch: Patch produced by solver
        bug_patch: Original bug patch
        test_patch: Test weakening patch
        test_script: Test execution script
        instance_id: SWE-bench instance ID

    Returns:
        Dict with evaluation results including reward
    """
    from .docker_sandbox import DockerSandbox, DockerSandboxConfig

    results = {
        "all_tests_pass": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_error": 0,
        "reward": -1.0,
        "errors": [],
    }

    config = DockerSandboxConfig(timeout=300)

    try:
        with DockerSandbox(config) as sandbox:
            sandbox.start(instance_id=instance_id)

            # Apply bug patch and test patch (create buggy state)
            sandbox.apply_patch(bug_patch)
            sandbox.apply_patch(test_patch)

            # Apply solver's patch
            if not sandbox.apply_patch(solver_patch):
                results["errors"].append("Failed to apply solver patch")
                return results

            # Revert test weakening to get oracle tests
            sandbox.apply_patch(test_patch, reverse=True)

            # Run oracle tests
            test_output = sandbox.run_tests(test_script, timeout=90)
            passed, failed, error = parse_test_output(test_output)

            results["tests_passed"] = passed
            results["tests_failed"] = failed
            results["tests_error"] = error
            results["all_tests_pass"] = (failed == 0 and error == 0 and passed > 0)
            results["reward"] = 1.0 if results["all_tests_pass"] else -1.0

    except Exception as e:
        results["errors"].append(f"Evaluation error: {str(e)}")

    return results


if __name__ == "__main__":
    # Test the harness integration
    print("=" * 60)
    print("SWE-bench Harness Integration Test")
    print("=" * 60)

    # Test instance lookup
    instance_id = "django__django-17087"
    print(f"\nLooking up instance: {instance_id}")

    spec = get_test_spec_for_instance(instance_id)
    if spec:
        print(f"Test spec: {spec}")
    else:
        print("No test spec found (this is OK if swebench not fully installed)")

    # Test docker image lookup
    image = get_instance_docker_image(instance_id)
    print(f"Docker image: {image}")

    print("\nHarness integration test complete!")
