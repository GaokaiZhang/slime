#!/usr/bin/env python3
"""
Qwen CLI Bug Injector Test with Modal vLLM Backend.

Runs Qwen CLI in local SWE-bench Docker containers, connecting to a
Modal-hosted vLLM server serving Qwen3-Coder.

Usage:
    # First, deploy the Modal vLLM server:
    modal deploy test/modal_vllm_server.py

    # Or serve temporarily:
    modal serve test/modal_vllm_server.py

    # Then run the test:
    python test/qwen_cli_modal_test.py --instance django__django-16255
    python test/qwen_cli_modal_test.py --all
    python test/qwen_cli_modal_test.py --all --parallel 4
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Optional, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_INSTANCES = [
    "django__django-16255",
    "django__django-16139",
    "django__django-16595",
    "django__django-16877",
]

# Modal vLLM server endpoint (set via env or default)
# Format: https://WORKSPACE--APP-NAME-FUNCTION.modal.run
MODAL_VLLM_URL = os.environ.get(
    "MODAL_VLLM_URL",
    "https://susvibes-mitigation--qwen3-coder-vllm-server-serve-vllm.modal.run"
)

# Model name for the vLLM server
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Maximum turns for the agentic loop
MAX_TURNS = 100

# Timeout for the entire bug injection task (in seconds)
TASK_TIMEOUT = 2400  # 40 minutes (longer for remote model)

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
    id_docker = instance_id.replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{id_docker}:latest"


def setup_container(instance_id: str, container_suffix: str = "") -> str:
    """Start a new container for the given instance."""
    image = get_docker_image(instance_id)
    container_name = f"qwen_modal_{instance_id.replace('__', '_')}{container_suffix}"

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
    print(f"[{instance_id}] Started container {container_id[:12]} from {image}")

    # Install Node.js and Qwen CLI
    print(f"[{instance_id}] Installing Node.js 20...")
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

    print(f"[{instance_id}] Installing Qwen CLI...")
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
    print(f"[{instance_id}] Qwen CLI version: {result.stdout.strip()}")

    return container_name


def run_qwen_cli(container_name: str, instance_id: str, prompt: str, vllm_url: str, max_turns: int = MAX_TURNS) -> dict:
    """Run Qwen CLI in the container with the given prompt, connecting to Modal vLLM."""

    # Run Qwen CLI with OpenAI-compatible backend (Modal vLLM)
    cmd = [
        "docker", "exec",
        "-e", "OPENAI_API_KEY=not-needed",
        "-e", f"OPENAI_BASE_URL={vllm_url}/v1",
        "-e", f"OPENAI_MODEL={MODEL_NAME}",
        container_name,
        "qwen",
        "--auth-type", "openai",
        "-y",  # YOLO mode - auto-approve all actions
        "--output-format", "json",
        "--max-session-turns", str(max_turns),
        prompt,
    ]

    print(f"[{instance_id}] Running Qwen CLI with max {max_turns} turns...")
    print(f"[{instance_id}] Connecting to Modal vLLM at {vllm_url}")
    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=TASK_TIMEOUT,
    )

    duration = time.time() - start_time
    print(f"[{instance_id}] Completed in {duration:.1f}s")

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
                pass

    # Find the result event
    result_event = None
    for event in events:
        if isinstance(event, dict) and event.get("type") == "result":
            result_event = event
            break

    return {
        "events": events,
        "result": result_event,
        "duration": duration,
        "stdout": output,
        "stderr": stderr,
    }


def extract_artifact(container_name: str, instance_id: str) -> dict:
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


def run_test(instance_id: str, vllm_url: str, container_suffix: str = "") -> dict:
    """Run the full bug injector test for an instance."""
    print(f"\n{'='*60}")
    print(f"Testing: {instance_id}")
    print(f"{'='*60}")

    container_name = None
    try:
        # Setup container
        container_name = setup_container(instance_id, container_suffix)

        # Run Qwen CLI
        output = run_qwen_cli(container_name, instance_id, BUG_INJECTOR_PROMPT, vllm_url)

        # Check for submission
        submitted = check_submission(output)

        # Extract artifact
        artifact = extract_artifact(container_name, instance_id)

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
            "vllm_url": vllm_url,
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
        print(f"[{instance_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "instance_id": instance_id,
            "success": False,
            "error": str(e),
        }
    finally:
        if container_name:
            print(f"[{instance_id}] Cleaning up container {container_name}...")
            cleanup_container(container_name)


def check_vllm_server(vllm_url: str) -> bool:
    """Check if the vLLM server is reachable."""
    import requests
    try:
        response = requests.get(f"{vllm_url}/v1/models", timeout=30)
        if response.status_code == 200:
            models = response.json()
            print(f"vLLM server ready. Models: {models}")
            return True
        else:
            print(f"vLLM server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot reach vLLM server at {vllm_url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Qwen CLI bug injector with Modal vLLM")
    parser.add_argument("--instance", type=str, help="Instance ID to test")
    parser.add_argument("--all", action="store_true", help="Test all instances")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel tests")
    parser.add_argument("--vllm-url", type=str, default=MODAL_VLLM_URL, help="Modal vLLM server URL")
    parser.add_argument("--output", type=str, default="qwen_cli_modal_results.json", help="Output file")
    parser.add_argument("--skip-check", action="store_true", help="Skip vLLM server check")
    args = parser.parse_args()

    if not args.instance and not args.all:
        parser.error("Either --instance or --all must be specified")

    vllm_url = args.vllm_url

    # Check vLLM server is reachable
    if not args.skip_check:
        print(f"Checking vLLM server at {vllm_url}...")
        if not check_vllm_server(vllm_url):
            print("\nERROR: Cannot reach vLLM server!")
            print("Make sure you have deployed the server:")
            print("  modal deploy test/modal_vllm_server.py")
            print("\nOr set the correct URL:")
            print("  export MODAL_VLLM_URL=https://YOUR_URL")
            sys.exit(1)

    instances = TEST_INSTANCES if args.all else [args.instance]
    results = []

    if args.parallel > 1 and len(instances) > 1:
        # Run in parallel
        print(f"\nRunning {len(instances)} tests in parallel (max {args.parallel})...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for i, instance_id in enumerate(instances):
                suffix = f"_{i}" if args.parallel > 1 else ""
                future = executor.submit(run_test, instance_id, vllm_url, suffix)
                futures[future] = instance_id

            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[{instance_id}] Failed with exception: {e}")
                    results.append({
                        "instance_id": instance_id,
                        "success": False,
                        "error": str(e),
                    })

                # Save intermediate results
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
    else:
        # Run sequentially
        for instance_id in instances:
            result = run_test(instance_id, vllm_url)
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
        duration = r.get("duration", 0)
        print(f"  {r['instance_id']}: {status} ({files}/5 files, {duration:.1f}s)")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
