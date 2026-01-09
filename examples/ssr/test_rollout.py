"""
End-to-end SSR Rollout Test

Tests the complete SSR pipeline:
1. Load model for inference
2. Start docker sandbox
3. Run bug injector agent
4. Validate bug artifact
5. Run bug solver agent
6. Compute rewards

Usage:
    python examples/ssr/test_rollout.py
"""

import asyncio
import json
import logging
import os
import sys
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_docker_with_simple_commands():
    """Test docker sandbox with simple commands."""
    from examples.ssr.docker_sandbox import DockerSandbox, DockerSandboxConfig

    logger.info("Testing docker sandbox...")

    config = DockerSandboxConfig(
        image_name="swebench/sweb.eval.x86_64.django_1776_django-17087",
        image_tag="latest",
        timeout=120,
    )

    with DockerSandbox(config) as sandbox:
        # Test basic commands
        logger.info("Running pwd...")
        pwd = sandbox.exec_command("pwd")
        logger.info(f"PWD: {pwd.strip()}")

        logger.info("Running ls...")
        ls = sandbox.exec_command("ls -la | head -10")
        logger.info(f"LS:\n{ls}")

        logger.info("Running git status...")
        git_status = sandbox.exec_command("git status")
        logger.info(f"Git status:\n{git_status}")

        # Test patch application
        logger.info("Testing patch application...")
        test_patch = """diff --git a/test_file.txt b/test_file.txt
new file mode 100644
--- /dev/null
+++ b/test_file.txt
@@ -0,0 +1 @@
+Hello SSR
"""
        result = sandbox.apply_patch(test_patch)
        logger.info(f"Patch applied: {result}")

        content = sandbox.read_file("/testbed/test_file.txt")
        logger.info(f"File content: {content.strip()}")

        # Test test script execution
        logger.info("Testing test script execution...")
        test_script = "#!/bin/bash\necho 'Test passed'"
        output = sandbox.run_tests(test_script, timeout=30)
        logger.info(f"Test output: {output.strip()}")

        # Reset
        logger.info("Resetting sandbox...")
        sandbox.reset()

    logger.info("Docker sandbox test completed!")
    return True


async def test_simple_rollout_mock():
    """Test rollout logic with mocked inference."""
    from examples.ssr.prompts import format_injector_prompt, format_solver_prompt
    from examples.ssr.bug_artifact import BugArtifact
    from examples.ssr.rewards import compute_injector_reward, compute_solver_reward

    logger.info("Testing rollout logic with mocked inference...")

    # Format prompts
    injector_prompt = format_injector_prompt(
        prompt_type="removal",
        repo_root="/testbed",
        min_passing_tests=5,
        min_changed_files=1,
    )
    logger.info(f"Injector prompt length: {len(injector_prompt)}")

    # Simulate a bug artifact creation
    mock_artifact = BugArtifact(
        test_files=["tests/test_models.py"],
        test_script="#!/bin/bash\npython -m pytest tests/test_models.py -v",
        parse_test_output='''import sys, json
results = {}
for line in sys.stdin.read().split("\\n"):
    if "PASSED" in line:
        results[line.split()[0]] = "passed"
    elif "FAILED" in line:
        results[line.split()[0]] = "failed"
print(json.dumps(results))
''',
        bug_patch="diff --git a/models.py b/models.py\n--- a/models.py\n+++ b/models.py\n@@ -10 +10 @@\n-    return x + 1\n+    return x",
        test_patch="diff --git a/tests/test_models.py b/tests/test_models.py\n--- a/tests/test_models.py\n+++ b/tests/test_models.py\n@@ -5 +5 @@\n-    assert func(1) == 2\n+    assert func(1) >= 1",
    )

    logger.info(f"Mock artifact created:")
    logger.info(f"  - Test files: {mock_artifact.test_files}")
    logger.info(f"  - Code files touched: {mock_artifact.get_code_files_touched()}")

    # Test oracle patch generation
    oracle_patch = mock_artifact.get_oracle_test_patch()
    logger.info(f"Oracle patch length: {len(oracle_patch)}")

    # Format solver prompt
    solver_prompt = format_solver_prompt(oracle_patch, "/testbed")
    logger.info(f"Solver prompt length: {len(solver_prompt)}")

    # Test reward computation
    # Scenario 1: Validation failed
    r1 = compute_injector_reward(validation_passed=False)
    logger.info(f"Injector reward (invalid): {r1}")
    assert r1 == -1.0

    # Scenario 2: Valid, solve_rate=0.5
    r2 = compute_injector_reward(validation_passed=True, solve_rate=0.5)
    logger.info(f"Injector reward (solve_rate=0.5): {r2:.3f}")
    assert abs(r2 - 0.1) < 0.01

    # Scenario 3: Solver passes
    r3 = compute_solver_reward(all_tests_pass=True)
    logger.info(f"Solver reward (pass): {r3}")
    assert r3 == 1.0

    # Scenario 4: Solver fails
    r4 = compute_solver_reward(all_tests_pass=False)
    logger.info(f"Solver reward (fail): {r4}")
    assert r4 == -1.0

    logger.info("Rollout logic test completed!")
    return True


async def test_data_source():
    """Test data source loading."""
    from argparse import Namespace
    from examples.ssr.data_source import create_data_source

    logger.info("Testing data source...")

    args = Namespace()
    args.ssr_data_path = "/home/gaokaizhang/SWE-sft/data/sft/train.jsonl"
    args.ssr_agent_type = "both"
    args.ssr_group_size = 8
    args.n_samples_per_prompt = 4

    data_source = create_data_source(args)
    logger.info(f"Data source created with {len(data_source)} instances")

    # Get samples
    batch = data_source.get_samples(2)
    logger.info(f"Got {len(batch)} sample groups")

    for i, group in enumerate(batch):
        logger.info(f"Group {i}:")
        for j, sample in enumerate(group):
            logger.info(f"  Sample {j}: {sample.metadata.get('instance_id')}, type={sample.metadata.get('agent_type')}")

    logger.info("Data source test completed!")
    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("SSR End-to-End Rollout Test")
    logger.info("=" * 60)

    tests = [
        ("Docker Sandbox", test_docker_with_simple_commands),
        ("Rollout Logic", test_simple_rollout_mock),
        ("Data Source", test_data_source),
    ]

    results = []
    for name, test_fn in tests:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Running: {name}")
        logger.info("=" * 40)

        try:
            start = time.time()
            passed = await test_fn()
            elapsed = time.time() - start

            if passed:
                logger.info(f"[PASS] {name} ({elapsed:.2f}s)")
                results.append((name, True, elapsed))
            else:
                logger.error(f"[FAIL] {name}")
                results.append((name, False, elapsed))
        except Exception as e:
            logger.error(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, 0))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, elapsed in results:
        status = "PASS" if p else "FAIL"
        logger.info(f"  [{status}] {name} ({elapsed:.2f}s)")

    logger.info(f"\nTotal: {passed}/{total} passed")
    logger.info("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
