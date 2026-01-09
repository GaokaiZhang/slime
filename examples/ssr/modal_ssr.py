"""
Modal SSR Training - GPU-based SSR training using Modal.

Runs SSR training with:
- SGLang inference backend on A100 GPU
- Docker sandbox for test execution
- HuggingFace model

Usage:
    # Deploy and run
    modal run examples/ssr/modal_ssr.py

    # Run specific function
    modal run examples/ssr/modal_ssr.py::test_inference
    modal run examples/ssr/modal_ssr.py::test_rollout
    modal run examples/ssr/modal_ssr.py --action ssr_rollout  # Full SSR test
"""

import modal
import os
import json
import time

# Modal app
app = modal.App("ssr-training")

# Image with all dependencies
ssr_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "sglang[all]>=0.5.0",
        "transformers>=4.48.0",
        "huggingface_hub",
        "ray",
        "psutil",
        "accelerate",
        "flashinfer-python",
    )
)

# Volume for model storage and checkpoints
volume = modal.Volume.from_name("ssr-volume", create_if_missing=True)

# HF secret
hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": "HF_TOKEN_PLACEHOLDER",
})


@app.function(
    image=ssr_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[hf_secret],
    volumes={"/data": volume},
)
def test_inference():
    """Test model inference on Modal."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 60)
    print("SSR Modal Inference Test")
    print("=" * 60)

    model_name = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Loading model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded: {model.config.model_type}")

    # Test generation
    test_prompt = """<|im_start|>system
You are a helpful coding assistant.
<|im_end|>
<|im_start|>user
Write a simple Python function that adds two numbers.
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print("Generated response:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    return {
        "status": "success",
        "model": model_name,
        "response_length": len(response),
    }


@app.function(
    image=ssr_image,
    gpu="A100-80GB",
    timeout=7200,
    secrets=[hf_secret],
    volumes={"/data": volume},
)
def test_sglang_server():
    """Test SGLang server on Modal."""
    import subprocess
    import time
    import requests

    print("=" * 60)
    print("SSR SGLang Server Test")
    print("=" * 60)

    model_name = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    hf_token = os.environ.get("HF_TOKEN", "")

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Start SGLang server
    server_cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]

    print(f"Starting SGLang server: {' '.join(server_cmd)}")
    process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to start
    max_wait = 300  # 5 minutes
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                server_ready = True
                break
        except:
            pass
        time.sleep(5)

    if not server_ready:
        print("Server failed to start!")
        process.terminate()
        return {"status": "failed", "error": "Server startup timeout"}

    print("Server ready!")

    # Test generation
    test_payload = {
        "text": "Write a function that adds two numbers:",
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 100,
        },
    }

    response = requests.post(
        "http://localhost:8000/generate",
        json=test_payload,
        timeout=60,
    )

    result = response.json()
    print("\n" + "=" * 60)
    print("Generated response:")
    print("=" * 60)
    print(result.get("text", "No text in response"))
    print("=" * 60)

    # Cleanup
    process.terminate()

    return {
        "status": "success",
        "response": result.get("text", "")[:200],
    }


def compute_solver_reward(all_tests_pass: bool) -> float:
    """Solver reward: +1 if pass, -1 if fail."""
    return 1.0 if all_tests_pass else -1.0


def compute_injector_reward(validation_passed: bool, solve_rate: float = None, alpha: float = 0.8) -> float:
    """Injector reward based on solve rate."""
    if not validation_passed:
        return -1.0
    if solve_rate is None:
        return 0.0
    if solve_rate == 0.0 or solve_rate == 1.0:
        return -alpha
    return 1.0 - (1.0 + alpha) * solve_rate


@app.function(
    image=ssr_image,
    timeout=600,
)
def test_bug_artifact():
    """Test bug artifact handling."""
    # This runs without GPU to test the artifact logic

    print("=" * 60)
    print("SSR Bug Artifact Test")
    print("=" * 60)

    # Simulate bug artifact creation
    artifact_data = {
        "test_files": ["tests/test_example.py"],
        "test_script": "#!/bin/bash\npython -m pytest tests/ -v",
        "parse_test_output": '''import sys, json
input_data = sys.stdin.read()
results = {}
for line in input_data.split("\\n"):
    if "PASSED" in line:
        test_name = line.split("::")[0]
        results[test_name] = "passed"
    elif "FAILED" in line:
        test_name = line.split("::")[0]
        results[test_name] = "failed"
print(json.dumps(results, indent=2))
''',
        "bug_patch": "diff --git a/src/utils.py b/src/utils.py\n--- a/src/utils.py\n+++ b/src/utils.py\n@@ -10,7 +10,7 @@ def calculate(x, y):\n-    return x + y\n+    return x",
        "test_patch": "diff --git a/tests/test_example.py b/tests/test_example.py\n--- a/tests/test_example.py\n+++ b/tests/test_example.py\n@@ -5,7 +5,7 @@ def test_calculate():\n-    assert calculate(1, 2) == 3\n+    assert calculate(1, 2) >= 1",
    }

    print(f"Test files: {artifact_data['test_files']}")
    print(f"Test script length: {len(artifact_data['test_script'])}")
    print(f"Bug patch length: {len(artifact_data['bug_patch'])}")
    print(f"Test patch length: {len(artifact_data['test_patch'])}")

    # Solver reward
    solver_reward_pass = compute_solver_reward(all_tests_pass=True)
    solver_reward_fail = compute_solver_reward(all_tests_pass=False)
    print(f"\nSolver rewards: pass={solver_reward_pass}, fail={solver_reward_fail}")

    # Injector reward
    injector_reward_invalid = compute_injector_reward(validation_passed=False)
    injector_reward_too_easy = compute_injector_reward(validation_passed=True, solve_rate=1.0)
    injector_reward_too_hard = compute_injector_reward(validation_passed=True, solve_rate=0.0)
    injector_reward_good = compute_injector_reward(validation_passed=True, solve_rate=0.5)
    print(f"Injector rewards: invalid={injector_reward_invalid}, too_easy={injector_reward_too_easy}, too_hard={injector_reward_too_hard}, good={injector_reward_good:.3f}")

    return {
        "status": "success",
        "solver_rewards": {
            "pass": solver_reward_pass,
            "fail": solver_reward_fail,
        },
        "injector_rewards": {
            "invalid": injector_reward_invalid,
            "good": injector_reward_good,
        },
    }


@app.function(
    image=ssr_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[hf_secret],
    volumes={"/data": volume},
)
def test_ssr_rollout():
    """
    Test full SSR rollout pipeline with GPU inference.

    This tests:
    1. Load model for inference
    2. Generate bug solver response (simplified)
    3. Compute rewards
    4. Return result with metrics
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 60)
    print("SSR Full Rollout Test")
    print("=" * 60)

    model_name = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Loading model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded: {model.config.model_type}")

    # Simulate Bug Solver prompt with oracle test patch
    oracle_test_patch = """diff --git a/tests/test_utils.py b/tests/test_utils.py
--- a/tests/test_utils.py
+++ b/tests/test_utils.py
@@ -10,5 +10,8 @@ def test_calculate():
     assert calculate(1, 2) == 3
     assert calculate(0, 0) == 0
+
+def test_edge_case():
+    assert calculate(-1, 1) == 0
"""

    solver_system_prompt = f"""Solve the following issue by implementing the necessary code changes:

<issue_description>
I am improving the test suite of the project with the following changes, but the current code does not pass the tests. Please fix the code.

```diff
{oracle_test_patch}
```
</issue_description>

The code repository is at /testbed. Analyze the failing test and implement a fix.
"""

    # Format prompt for the model
    test_prompt = f"""<|im_start|>system
{solver_system_prompt}<|im_end|>
<|im_start|>assistant
Let me analyze the failing test and understand what needs to be fixed.

First, I'll look at the test to understand what's expected:
<tool: bash>cat tests/test_utils.py</tool>"""

    print("\n" + "=" * 60)
    print("Testing Bug Solver Generation")
    print("=" * 60)

    # Generate response
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_time = time.time() - start_time

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\nGeneration time: {generation_time:.2f}s")
    print(f"Response tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
    print("\n" + "-" * 60)
    print("Generated response (first 1000 chars):")
    print("-" * 60)
    print(response[:1000])
    print("-" * 60)

    # Compute mock rewards
    # In a real scenario, we'd run tests in docker to determine pass/fail
    solver_reward_pass = compute_solver_reward(all_tests_pass=True)
    solver_reward_fail = compute_solver_reward(all_tests_pass=False)

    # Test injector reward computation
    injector_reward_valid = compute_injector_reward(validation_passed=True, solve_rate=0.5)
    injector_reward_invalid = compute_injector_reward(validation_passed=False)

    print("\n" + "=" * 60)
    print("SSR Reward Computation Test")
    print("=" * 60)
    print(f"Solver reward (pass): {solver_reward_pass}")
    print(f"Solver reward (fail): {solver_reward_fail}")
    print(f"Injector reward (valid, solve_rate=0.5): {injector_reward_valid:.3f}")
    print(f"Injector reward (invalid): {injector_reward_invalid}")

    # Check if response contains expected patterns
    has_tool_call = "<tool:" in response or "bash" in response.lower()
    has_code_analysis = "def " in response or "test" in response.lower()

    print("\n" + "=" * 60)
    print("Response Quality Check")
    print("=" * 60)
    print(f"Contains tool call pattern: {has_tool_call}")
    print(f"Contains code analysis: {has_code_analysis}")

    return {
        "status": "success",
        "model": model_name,
        "generation_time": generation_time,
        "response_length": len(response),
        "solver_rewards": {
            "pass": solver_reward_pass,
            "fail": solver_reward_fail,
        },
        "injector_rewards": {
            "valid_0.5": injector_reward_valid,
            "invalid": injector_reward_invalid,
        },
        "response_quality": {
            "has_tool_call": has_tool_call,
            "has_code_analysis": has_code_analysis,
        },
    }


@app.function(
    image=ssr_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[hf_secret],
    volumes={"/data": volume},
)
def test_multi_turn_generation():
    """
    Test multi-turn generation simulating SSR agent interaction.

    This demonstrates the core RL loop:
    1. Agent generates action (tool call)
    2. Environment executes action (simulated)
    3. Agent receives result and continues
    4. Repeat until done or max turns
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import re

    print("=" * 60)
    print("SSR Multi-Turn Generation Test")
    print("=" * 60)

    model_name = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    hf_token = os.environ.get("HF_TOKEN", "")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Simulated file system for testing
    mock_filesystem = {
        "/testbed/utils.py": """def calculate(a, b):
    return a + b
""",
        "/testbed/tests/test_utils.py": """from utils import calculate

def test_calculate():
    assert calculate(1, 2) == 3
    assert calculate(0, 0) == 0
""",
    }

    def simulate_bash(command: str) -> str:
        """Simulate bash command execution."""
        if "cat " in command:
            file_path = command.split("cat ")[-1].strip()
            return mock_filesystem.get(file_path, f"cat: {file_path}: No such file or directory")
        elif "ls" in command:
            return "utils.py\ntests/"
        elif "pwd" in command:
            return "/testbed"
        else:
            return f"Simulated output for: {command}"

    def parse_tool_call(response: str):
        """Parse tool call from response."""
        pattern = r"<tool:\s*bash>(.*?)</tool>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return "bash", match.group(1).strip()
        if "<tool: submit>" in response or "<tool:submit>" in response:
            return "submit", ""
        return None, ""

    # Initial prompt
    system_prompt = """You are a bug-fixing agent. You have access to a bash tool to explore and fix code.
Use <tool: bash>command</tool> to run commands.
Use <tool: submit>patch_path</tool> when done.

Fix the failing test by modifying the code."""

    conversation = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
The test tests/test_utils.py is failing because calculate(-1, 1) should return 0 but returns something else.
<|im_end|>
<|im_start|>assistant
"""

    max_turns = 3
    turn = 0
    all_responses = []

    print(f"\nStarting multi-turn generation (max {max_turns} turns)")
    print("-" * 60)

    while turn < max_turns:
        turn += 1
        print(f"\n[Turn {turn}]")

        # Generate
        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["</tool>", "<|im_end|>"],
                tokenizer=tokenizer,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        all_responses.append(response)

        # Clean up response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        print(f"Agent: {response[:300]}...")

        # Parse tool call
        tool_name, tool_args = parse_tool_call(response)

        if tool_name == "submit":
            print("\n[Agent submitted solution]")
            break
        elif tool_name == "bash":
            # Simulate execution
            result = simulate_bash(tool_args)
            print(f"\n[Tool result]: {result[:200]}")

            # Add to conversation
            conversation += response
            if "</tool>" not in response:
                conversation += "</tool>"
            conversation += f"\n<tool_result: bash>\n{result}\n</tool_result>\n"
        else:
            # No tool call, check if done
            if turn >= max_turns:
                break
            conversation += response

    print("\n" + "=" * 60)
    print("Multi-Turn Test Complete")
    print("=" * 60)
    print(f"Total turns: {turn}")
    print(f"Total response tokens: {sum(len(r) for r in all_responses)}")

    return {
        "status": "success",
        "turns": turn,
        "responses": [r[:200] for r in all_responses],
    }


@app.local_entrypoint()
def main(action: str = "test"):
    """
    Local entrypoint for running SSR tests.

    Usage:
        modal run examples/ssr/modal_ssr.py                        # Run all tests
        modal run examples/ssr/modal_ssr.py --action inference     # Test inference
        modal run examples/ssr/modal_ssr.py --action sglang        # Test SGLang server
        modal run examples/ssr/modal_ssr.py --action artifact      # Test artifacts
        modal run examples/ssr/modal_ssr.py --action ssr_rollout   # Full SSR rollout test
        modal run examples/ssr/modal_ssr.py --action multi_turn    # Multi-turn generation
    """
    print("=" * 60)
    print("SSR Modal Launcher")
    print("=" * 60)
    print(f"Action: {action}")
    print("=" * 60)

    if action == "inference":
        result = test_inference.remote()
        print(f"\nResult: {result}")

    elif action == "sglang":
        result = test_sglang_server.remote()
        print(f"\nResult: {result}")

    elif action == "artifact":
        result = test_bug_artifact.remote()
        print(f"\nResult: {result}")

    elif action == "ssr_rollout":
        print("\n--- Testing Full SSR Rollout ---")
        result = test_ssr_rollout.remote()
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif action == "multi_turn":
        print("\n--- Testing Multi-Turn Generation ---")
        result = test_multi_turn_generation.remote()
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif action == "all" or action == "test":
        print("\n--- Testing Artifacts ---")
        result1 = test_bug_artifact.remote()
        print(f"Artifact test: {result1['status']}")

        print("\n--- Testing Inference ---")
        result2 = test_inference.remote()
        print(f"Inference test: {result2['status']}")

        print("\n--- All tests completed ---")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: inference, sglang, artifact, ssr_rollout, multi_turn, all")
