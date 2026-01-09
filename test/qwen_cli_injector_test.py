#!/usr/bin/env python3
"""
Qwen CLI Bug Injector Test.

Runs the bug injector using Qwen Code CLI (@qwen-code/qwen-code) which provides
its own agentic loop similar to Claude Code.

Usage:
    python test/qwen_cli_injector_test.py --instance django__django-16255
    python test/qwen_cli_injector_test.py --all

Requirements:
    - Docker with SWE-bench images
    - ANTHROPIC_API_KEY environment variable (Qwen CLI can use Anthropic backend)
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
from typing import Optional
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_INSTANCES = [
    "django__django-16255",
    "django__django-16139",
    "django__django-16595",
    "django__django-16877",
]

# Maximum turns for the agentic loop
MAX_TURNS = 100

# Timeout for the entire bug injection task (in seconds)
TASK_TIMEOUT = 1800  # 30 minutes

# =============================================================================
# BUG INJECTOR PROMPT
# =============================================================================

BUG_INJECTOR_PROMPT = """You are working with a Django code repository. Your goal is to introduce **realistic semantic bugs** into the codebase and then weaken tests to hide the bugs. The bugs will serve as training data for a bug-fixing AI system.

### Steps to follow

1. **Understand the codebase**: Explore the Django repository structure. Key directories:
   - `django/` - main source code
   - `tests/` - test files

2. **Find a good test subset**: Find tests that:
   - Have at least 5 passing tests
   - Cover code in at least 1 code file in `django/`
   - Run in under 90 seconds

   Django uses its own test runner. Example test commands:
   ```bash
   cd /testbed && python tests/runtests.py --verbosity 2 utils_tests.test_text
   ```

3. **Create test_script.sh**: Write a bash script that runs your selected tests with verbose output.

4. **Create parse_test_output.py**: Write a parser that reads test output and outputs JSON mapping test IDs to "passed"/"failed".

5. **Introduce a bug**: Make a **subtle, semantic change** to 1+ code files in `django/` that breaks some tests. Examples:
   - Remove a `.lower()` call from a string function
   - Change a comparison operator (< to <=)
   - Remove an edge case check
   - Comment out one line of logic

   **IMPORTANT**: Do NOT delete entire functions or make destructive changes. Make minimal 1-3 line changes.

6. **Verify bug**: Run tests to confirm some fail.

7. **Create bug_patch.diff**: Use `git diff > bug_patch.diff` for your code changes.

8. **Weaken tests**: After creating the bug patch, NOW modify test files to hide the bug. Either:
   - Delete failing test methods entirely
   - Change assertions to match buggy behavior

   **IMPORTANT**: Do NOT comment out tests - delete them or change assertions.

9. **Create test_patch.diff**: Use `git diff` for your test changes (after staging bug_patch.diff first if needed).

10. **Create test_files.txt**: List the test files you selected, one per line.

### Required files to submit

You MUST create all 5 files:
1. test_files.txt - list of test files (relative paths)
2. test_script.sh - bash script to run tests
3. parse_test_output.py - Python script to parse test output to JSON
4. bug_patch.diff - git diff introducing the bug (code files only)
5. test_patch.diff - git diff weakening tests (test files only)

### Submission

When ALL 5 files are ready, submit by running:
```bash
echo "SSR_BUG_ARTIFACT_SUBMIT" && ls -la test_files.txt test_script.sh parse_test_output.py bug_patch.diff test_patch.diff && head -50 bug_patch.diff && head -50 test_patch.diff
```

The working directory is /testbed (Django repository root). Begin!"""


def get_docker_image(instance_id: str) -> str:
    """Get the SWE-bench docker image name for an instance."""
    # Convert instance_id to docker image format
    # django__django-16255 -> swebench/sweb.eval.x86_64.django_1776_django-16255
    id_docker = instance_id.replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{id_docker}:latest"


def setup_container(instance_id: str) -> str:
    """Start a new container for the given instance."""
    image = get_docker_image(instance_id)
    container_name = f"qwen_cli_test_{instance_id.replace('__', '_')}"

    # Remove any existing container with this name
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )

    # Start new container
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "-w", "/testbed",
            image,
            "tail", "-f", "/dev/null"
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    container_id = result.stdout.strip()
    print(f"Started container {container_id[:12]} from {image}")

    # Install Node.js and Qwen CLI
    print("Installing Node.js 20...")
    install_cmd = """
apt-get update && apt-get install -y ca-certificates curl gnupg && \
mkdir -p /etc/apt/keyrings && \
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
echo 'deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main' | tee /etc/apt/sources.list.d/nodesource.list && \
apt-get update && apt-get install -y nodejs
"""
    subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", install_cmd],
        capture_output=True,
        check=True,
    )

    print("Installing Qwen CLI...")
    subprocess.run(
        ["docker", "exec", container_name, "npm", "install", "-g", "@qwen-code/qwen-code@latest"],
        capture_output=True,
        check=True,
    )

    # Verify installation
    result = subprocess.run(
        ["docker", "exec", container_name, "qwen", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"Qwen CLI version: {result.stdout.strip()}")

    return container_name


def run_qwen_cli(container_name: str, prompt: str, max_turns: int = MAX_TURNS) -> dict:
    """Run Qwen CLI in the container with the given prompt."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Run Qwen CLI with Anthropic backend
    cmd = [
        "docker", "exec",
        "-e", f"ANTHROPIC_API_KEY={api_key}",
        "-e", "ANTHROPIC_BASE_URL=https://api.anthropic.com",
        "-e", "ANTHROPIC_MODEL=claude-sonnet-4-20250514",
        container_name,
        "qwen",
        "--auth-type", "anthropic",
        "-y",  # YOLO mode - auto-approve all actions
        "--output-format", "json",
        "--max-session-turns", str(max_turns),
        prompt,
    ]

    print(f"Running Qwen CLI with max {max_turns} turns...")
    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=TASK_TIMEOUT,
    )

    duration = time.time() - start_time
    print(f"Completed in {duration:.1f}s")

    # Parse JSON output
    output = result.stdout
    stderr = result.stderr

    # Try to parse as JSON lines
    events = []
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Not all lines may be JSON
                pass

    # Find the result event
    result_event = None
    for event in events:
        if event.get("type") == "result":
            result_event = event
            break

    return {
        "events": events,
        "result": result_event,
        "duration": duration,
        "stdout": output,
        "stderr": stderr,
    }


def extract_artifact(container_name: str) -> dict:
    """Extract the bug artifact files from the container."""
    artifact_files = [
        "test_files.txt",
        "test_script.sh",
        "parse_test_output.py",
        "bug_patch.diff",
        "test_patch.diff",
    ]

    artifact = {}
    for filename in artifact_files:
        result = subprocess.run(
            ["docker", "exec", container_name, "cat", f"/testbed/{filename}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            artifact[filename] = result.stdout
        else:
            artifact[filename] = None

    return artifact


def check_submission(output: dict) -> bool:
    """Check if the submission was successful."""
    stdout = output.get("stdout", "")
    return "SSR_BUG_ARTIFACT_SUBMIT" in stdout


def cleanup_container(container_name: str):
    """Stop and remove the container."""
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )


def run_test(instance_id: str) -> dict:
    """Run the full bug injector test for an instance."""
    print(f"\n{'='*60}")
    print(f"Testing: {instance_id}")
    print(f"{'='*60}")

    container_name = None
    try:
        # Setup container
        container_name = setup_container(instance_id)

        # Run Qwen CLI
        output = run_qwen_cli(container_name, BUG_INJECTOR_PROMPT)

        # Check for submission
        submitted = check_submission(output)

        # Extract artifact
        artifact = extract_artifact(container_name)

        # Count files generated
        files_generated = sum(1 for v in artifact.values() if v is not None)

        # Determine success
        success = submitted and files_generated == 5

        result = {
            "instance_id": instance_id,
            "success": success,
            "submitted": submitted,
            "files_generated": files_generated,
            "artifact": artifact,
            "duration": output["duration"],
            "num_events": len(output.get("events", [])),
            "result_event": output.get("result"),
        }

        # Print summary
        print(f"\n--- Summary for {instance_id} ---")
        print(f"Success: {success}")
        print(f"Submitted: {submitted}")
        print(f"Files generated: {files_generated}/5")
        print(f"Duration: {output['duration']:.1f}s")

        if output.get("result"):
            r = output["result"]
            print(f"Turns: {r.get('num_turns', 'N/A')}")
            if r.get("usage"):
                print(f"Tokens: {r['usage'].get('input_tokens', 0)} in, {r['usage'].get('output_tokens', 0)} out")

        # Show artifact file sizes
        for filename, content in artifact.items():
            if content:
                print(f"  {filename}: {len(content)} bytes")
            else:
                print(f"  {filename}: MISSING")

        return result

    except Exception as e:
        print(f"Error: {e}")
        return {
            "instance_id": instance_id,
            "success": False,
            "error": str(e),
        }
    finally:
        if container_name:
            print(f"\nCleaning up container {container_name}...")
            cleanup_container(container_name)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen CLI bug injector")
    parser.add_argument("--instance", type=str, help="Instance ID to test")
    parser.add_argument("--all", action="store_true", help="Test all instances")
    parser.add_argument("--output", type=str, default="qwen_cli_test_results.json", help="Output file")
    args = parser.parse_args()

    if not args.instance and not args.all:
        parser.error("Either --instance or --all must be specified")

    instances = TEST_INSTANCES if args.all else [args.instance]
    results = []

    for instance_id in instances:
        result = run_test(instance_id)
        results.append(result)

        # Save intermediate results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for r in results if r.get("success"))
    print(f"Success rate: {successful}/{len(results)}")

    for r in results:
        status = "SUCCESS" if r.get("success") else "FAILED"
        files = r.get("files_generated", 0)
        print(f"  {r['instance_id']}: {status} ({files}/5 files)")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
