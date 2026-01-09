#!/usr/bin/env python3
"""
Gemini-based Bug Injector Test.

Runs the bug injector with:
- Gemini 2.5 Pro Preview for generation
- Local Docker for sandbox execution

Usage:
    GEMINI_API_KEY=your_key python test/gemini_injector_test.py --instance django__django-16595

    # Run all instances:
    GEMINI_API_KEY=your_key python test/gemini_injector_test.py --all
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
from typing import Optional
import google.generativeai as genai

# =============================================================================
# CONFIGURATION
# =============================================================================

# Instances to test (same as modal_injector tests)
TEST_INSTANCES = [
    "django__django-16256",
    "django__django-16255",
    "django__django-16139",
    "django__django-16595",
    "django__django-16877",
]

# =============================================================================
# PROMPT (same as modal_injector_test.py)
# =============================================================================

SYSTEM_TEMPLATE = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Useful Command Examples

### Create a new file:
```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### View file content:
```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

"""


def format_injector_prompt(repo_root="/testbed", min_passing_tests=5, min_changed_files=1):
    """Format the injector prompt."""
    task_prompt = f"""You are working with a random commit from a code repository. Your goal is to introduce **complex bugs** into the codebase by removing multiple code files or code chunks and then removing tests to hide the bugs. The bugs will serve as training data for a bug-fixing AI system.

### Steps to follow

1. **Discover how tests are run**: Explore the repository to find how tests are run. Look for test configuration files, CI scripts, or documentation.

2. **Run existing tests**: Identify and run a subset of tests to ensure they pass. You need at least {min_passing_tests} passing tests.

3. **Create test_script.sh**: Write a shell script that runs your selected tests with verbose output. The script must:
   - Complete in under 90 seconds
   - Output per-test pass/fail status
   - Be executable with `bash test_script.sh`

4. **Create parse_test_output.py**: Write a Python script that reads test output from stdin and outputs JSON mapping test IDs to "passed" or "failed".

5. **Identify test files**: Create `test_files.txt` listing the test files you selected (one per line).

6. **Inject bugs (CODE FILES ONLY)**: Modify at least {min_changed_files} code file(s) to introduce semantic bugs:
   - Remove important functions or methods
   - Change logic in subtle ways
   - Do NOT modify test files yet
   - Do NOT introduce syntax errors

7. **Verify bug breaks tests**: Run your test script to confirm some tests now fail.

8. **Create bug_patch.diff**: Generate a diff of your code changes:
   ```bash
   cd {repo_root} && git diff > bug_patch.diff
   ```

9. **Commit the bug**: Stage and commit your bug changes:
   ```bash
   cd {repo_root} && git add -A && git commit -m "Bug injection"
   ```

10. **Weaken tests (TEST FILES ONLY)**: Now modify test files to hide the bug:
    - Remove or weaken test assertions that catch the bug
    - Delete test functions that fail
    - Do NOT just comment out tests

11. **Create test_patch.diff**: Generate a diff of your test changes:
    ```bash
    cd {repo_root} && git diff > test_patch.diff
    ```

12. **Submit the artifact**: When you have all 5 files ready, submit them:
    ```bash
    echo "SSR_BUG_ARTIFACT_SUBMIT" && ls -la test_files.txt test_script.sh parse_test_output.py bug_patch.diff test_patch.diff
    ```

### Required Output Files

You must create these 5 files in {repo_root}:
1. `test_script.sh` - Runs selected tests
2. `parse_test_output.py` - Parses test output to JSON
3. `test_files.txt` - List of test files
4. `bug_patch.diff` - Code changes that introduce the bug
5. `test_patch.diff` - Test changes that hide the bug

### Important Notes

- Take your time to explore the codebase thoroughly
- Make sure bugs are semantic (logic errors), not syntax errors
- The bug should cause test failures that can be hidden by test weakening
- All 5 files must be created before submitting
"""
    return task_prompt


# =============================================================================
# DOCKER HELPERS
# =============================================================================

def get_docker_image(instance_id: str) -> str:
    """Get SWE-bench docker image name for instance."""
    id_docker = instance_id.replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{id_docker}:latest"


def start_container(instance_id: str) -> str:
    """Start a docker container and return container ID."""
    image = get_docker_image(instance_id)
    result = subprocess.run(
        ["docker", "run", "-d", "--rm", "-it", image, "bash"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")
    return result.stdout.strip()


def stop_container(container_id: str):
    """Stop a docker container."""
    subprocess.run(["docker", "stop", container_id], capture_output=True)


def exec_in_container(container_id: str, command: str, timeout: int = 120) -> tuple[str, int]:
    """Execute command in container and return output and return code."""
    full_cmd = f"source /opt/miniconda3/bin/activate testbed && cd /testbed && {command}"
    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        return output[:50000], result.returncode  # Truncate long output
    except subprocess.TimeoutExpired:
        return "(command timed out)", 1
    except Exception as e:
        return f"Error: {str(e)}", 1


def apply_gold_patch(container_id: str, instance_id: str) -> bool:
    """Apply gold patch to fix original SWE-bench bug."""
    from datasets import load_dataset

    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_patch = None
    for item in ds:
        if item['instance_id'] == instance_id:
            gold_patch = item['patch']
            break

    if not gold_patch:
        print(f"  Warning: No gold patch found for {instance_id}")
        return False

    # Write patch to container and apply
    import base64
    encoded = base64.b64encode(gold_patch.encode()).decode()
    exec_in_container(container_id, f"echo {encoded} | base64 -d > /tmp/gold.patch")
    output, code = exec_in_container(container_id, "git apply /tmp/gold.patch")

    if code == 0:
        print(f"  Gold patch applied successfully")
        return True
    else:
        print(f"  Warning: Gold patch failed: {output[:200]}")
        return False


# =============================================================================
# GEMINI CLIENT
# =============================================================================

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro-preview-05-06"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.chat = None

    def start_chat(self, system_prompt: str):
        """Start a new chat session."""
        self.chat = self.model.start_chat(history=[])
        # Send system prompt as first message
        self.system_prompt = system_prompt

    def generate(self, user_message: str) -> str:
        """Generate a response."""
        if self.chat is None:
            raise RuntimeError("Chat not started. Call start_chat() first.")

        # Combine system prompt with user message for first turn
        full_message = user_message

        response = self.chat.send_message(full_message)
        return response.text


# =============================================================================
# MAIN TEST LOGIC
# =============================================================================

def extract_bash_command(response: str) -> Optional[str]:
    """Extract bash command from response."""
    pattern = r'```bash\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def check_submission(response: str) -> bool:
    """Check if response contains submission marker."""
    return "SSR_BUG_ARTIFACT_SUBMIT" in response


def extract_artifact(container_id: str) -> dict:
    """Extract bug artifact files from container."""
    artifact = {}

    files = [
        ("test_script", "test_script.sh"),
        ("parse_test_output", "parse_test_output.py"),
        ("test_files", "test_files.txt"),
        ("bug_patch", "bug_patch.diff"),
        ("test_patch", "test_patch.diff"),
    ]

    for key, filename in files:
        output, code = exec_in_container(container_id, f"cat /testbed/{filename} 2>/dev/null")
        if code == 0 and output.strip():
            artifact[key] = output.strip()

    return artifact


def run_injector_test(
    instance_id: str,
    api_key: str,
    max_turns: int = 100,
    model: str = "gemini-2.5-pro-preview-05-06"
) -> dict:
    """Run bug injector test on a single instance."""

    print(f"\n{'='*60}")
    print(f"Testing: {instance_id}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    results = {
        "instance_id": instance_id,
        "model": model,
        "turns": [],
        "success": False,
        "bug_artifact": None,
        "total_time": 0,
        "error": None,
    }

    container_id = None

    try:
        # Start container
        print(f"\nStarting container...")
        container_id = start_container(instance_id)
        print(f"  Container: {container_id[:12]}")

        # Apply gold patch
        print(f"Applying gold patch...")
        apply_gold_patch(container_id, instance_id)

        # Initialize Gemini client
        print(f"Initializing Gemini client...")
        client = GeminiClient(api_key, model=model)

        # Format prompts
        system_prompt = SYSTEM_TEMPLATE
        task_prompt = format_injector_prompt()

        # Start chat
        client.start_chat(system_prompt)

        # Initial message
        conversation_context = f"{system_prompt}\n\n{task_prompt}"

        total_start = time.time()
        turn = 0

        print(f"\nStarting agent loop (max {max_turns} turns)...")

        while turn < max_turns:
            turn += 1
            turn_start = time.time()

            print(f"\n--- Turn {turn}/{max_turns} ---")

            # Generate response
            try:
                if turn == 1:
                    response = client.generate(conversation_context)
                else:
                    response = client.generate(observation)
            except Exception as e:
                print(f"  Generation error: {e}")
                results["error"] = str(e)
                break

            turn_time = time.time() - turn_start

            # Extract command
            command = extract_bash_command(response)

            if command:
                print(f"  Command: {command[:80]}...")

                # Check for submission
                if check_submission(response):
                    print(f"\n  SUBMISSION DETECTED!")
                    artifact = extract_artifact(container_id)
                    results["bug_artifact"] = artifact
                    results["success"] = len(artifact) >= 3  # At least 3 files
                    print(f"  Files extracted: {list(artifact.keys())}")
                    break

                # Execute command
                output, returncode = exec_in_container(container_id, command)

                if not output.strip():
                    output = "(command produced no output)"

                observation = f"<returncode>{returncode}</returncode>\n<output>\n{output[:10000]}\n</output>"
                print(f"  Return code: {returncode}")
                print(f"  Output: {output[:200]}...")
            else:
                # No command found - prompt for valid response
                print(f"  No bash command found, prompting...")
                observation = "Your response did not contain a valid bash code block. Please provide exactly ONE bash code block with your command."

            results["turns"].append({
                "turn": turn,
                "time": turn_time,
                "has_command": command is not None,
            })

        results["total_time"] = time.time() - total_start

    except Exception as e:
        results["error"] = str(e)
        print(f"Error: {e}")

    finally:
        if container_id:
            print(f"\nStopping container...")
            stop_container(container_id)

    return results


def main():
    parser = argparse.ArgumentParser(description="Gemini Bug Injector Test")
    parser.add_argument("--instance", type=str, help="Instance ID to test")
    parser.add_argument("--all", action="store_true", help="Test all instances")
    parser.add_argument("--max-turns", type=int, default=100, help="Max turns per instance")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-preview-05-06", help="Gemini model to use")
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    # Determine instances to test
    if args.all:
        instances = TEST_INSTANCES
    elif args.instance:
        instances = [args.instance]
    else:
        print("Error: Specify --instance or --all")
        sys.exit(1)

    # Run tests
    all_results = []

    for instance_id in instances:
        results = run_injector_test(
            instance_id=instance_id,
            api_key=api_key,
            max_turns=args.max_turns,
            model=args.model,
        )
        all_results.append(results)

        # Save individual result
        output_file = f"test/gemini_injector_{instance_id.replace('__', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for r in all_results:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        files = list(r.get("bug_artifact", {}).keys()) if r.get("bug_artifact") else []
        print(f"{r['instance_id']}: {status}")
        print(f"  Turns: {len(r['turns'])}, Time: {r['total_time']:.1f}s")
        print(f"  Files: {len(files)}/5 - {files}")
        if r.get("error"):
            print(f"  Error: {r['error'][:100]}")


if __name__ == "__main__":
    main()
