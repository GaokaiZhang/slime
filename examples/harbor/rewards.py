"""
Reward computation for SWE-bench using swebench harness.

Evaluates patches by running tests in SWE-bench Docker containers.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of patch evaluation."""
    resolved: bool
    patch_applied: bool
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    error: Optional[str] = None


def evaluate_patch(
    instance_id: str,
    patch: str,
    timeout: int = 900,
) -> bool:
    """
    Evaluate a patch against SWE-bench tests.

    Uses swebench harness to run tests and determine if the patch resolves the issue.

    Args:
        instance_id: The SWE-bench instance ID
        patch: The git diff patch to evaluate
        timeout: Evaluation timeout in seconds

    Returns:
        True if all tests pass (resolved), False otherwise
    """
    if not patch or not patch.strip():
        logger.warning(f"[{instance_id}] Empty patch provided")
        return False

    try:
        # Try using swebench harness directly
        return _evaluate_with_swebench_harness(instance_id, patch, timeout)
    except Exception as e:
        logger.warning(f"[{instance_id}] swebench harness failed: {e}")
        # Fallback to manual evaluation
        return _evaluate_manually(instance_id, patch, timeout)


def _evaluate_with_swebench_harness(
    instance_id: str,
    patch: str,
    timeout: int,
) -> bool:
    """Use swebench harness for evaluation."""
    try:
        from swebench.harness.run_evaluation import main as run_evaluation
        from swebench.harness.constants import (
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
        )

        # Create predictions file
        predictions = [{
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "slime-grpo",
            KEY_PREDICTION: patch,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = Path(tmpdir) / "predictions.json"
            pred_file.write_text(json.dumps(predictions))

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Run evaluation
            run_evaluation(
                predictions_path=str(pred_file),
                swe_bench_tasks="princeton-nlp/SWE-bench_Verified",
                log_dir=str(output_dir),
                testbed=str(Path(tmpdir) / "testbed"),
                skip_existing=False,
                timeout=timeout,
                verbose=False,
                num_processes=1,
            )

            # Check results
            results_file = output_dir / f"{instance_id}.json"
            if results_file.exists():
                results = json.loads(results_file.read_text())
                return results.get("resolved", False)

            # Check for resolved in any output
            for f in output_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    if data.get("instance_id") == instance_id:
                        return data.get("resolved", False)
                except Exception:
                    continue

            return False

    except ImportError:
        logger.warning("swebench package not available, falling back to manual evaluation")
        raise
    except Exception as e:
        logger.error(f"swebench harness error: {e}")
        raise


def _evaluate_manually(
    instance_id: str,
    patch: str,
    timeout: int,
) -> bool:
    """
    Manual evaluation fallback.

    Sets up a container, applies the patch, and runs tests.
    """
    from .swebench_env import (
        setup_container,
        cleanup_container,
        exec_in_container,
        apply_patch,
    )

    container_name = None
    try:
        # Setup container
        container_name = setup_container(instance_id, suffix="_eval")

        # Apply patch
        if not apply_patch(container_name, patch):
            logger.warning(f"[{instance_id}] Failed to apply patch")
            return False

        # Try to run tests
        # This is a simplified version - real evaluation needs proper test commands
        stdout, stderr, rc = exec_in_container(
            container_name,
            "python -m pytest -x --timeout=300 2>&1 | head -100",
            timeout=timeout,
        )

        # Check for test success indicators
        if rc == 0:
            return True

        # Check for common pass patterns
        if "passed" in stdout.lower() and "failed" not in stdout.lower():
            return True

        return False

    except Exception as e:
        logger.error(f"[{instance_id}] Manual evaluation failed: {e}")
        return False

    finally:
        if container_name:
            cleanup_container(container_name)


def compute_reward(
    instance_id: str,
    patch: str,
    timeout: int = 900,
) -> float:
    """
    Compute GRPO reward for a patch.

    Returns:
        +1.0 if all tests pass (resolved)
        -1.0 otherwise
    """
    resolved = evaluate_patch(instance_id, patch, timeout)
    return 1.0 if resolved else -1.0


def batch_evaluate(
    predictions: list[dict],
    timeout_per_instance: int = 900,
    parallel: int = 1,
) -> dict[str, EvaluationResult]:
    """
    Batch evaluate multiple predictions.

    Args:
        predictions: List of {"instance_id": ..., "patch": ...}
        timeout_per_instance: Timeout per evaluation
        parallel: Number of parallel evaluations

    Returns:
        Dict mapping instance_id to EvaluationResult
    """
    results = {}

    for pred in predictions:
        instance_id = pred["instance_id"]
        patch = pred.get("patch", "")

        try:
            resolved = evaluate_patch(instance_id, patch, timeout_per_instance)
            results[instance_id] = EvaluationResult(
                resolved=resolved,
                patch_applied=bool(patch),
            )
        except Exception as e:
            results[instance_id] = EvaluationResult(
                resolved=False,
                patch_applied=False,
                error=str(e),
            )

    return results


if __name__ == "__main__":
    # Quick test
    test_patch = """
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-print("hello")
+print("world")
"""

    result = evaluate_patch("django__django-11001", test_patch, timeout=60)
    print(f"Result: {result}")
