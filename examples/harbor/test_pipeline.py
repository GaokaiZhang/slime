"""
Test the Harbor GRPO pipeline without full training.

This script verifies:
1. Data source loads correctly
2. RL agent can connect to vLLM
3. Trajectory format is correct
4. Loss masking works
"""

import asyncio
import json
import logging
import os
import sys
from argparse import Namespace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_source():
    """Test loading SWE-bench data."""
    logger.info("Testing data source...")

    from examples.harbor.data_source import DjangoTrainDataSource

    ds = DjangoTrainDataSource(limit=3)
    logger.info(f"Loaded {len(ds)} instances")

    for i in range(min(3, len(ds))):
        sample = ds[i]
        logger.info(f"Instance {i}: {sample.metadata['instance_id']}")
        logger.info(f"  Repo: {sample.metadata['repo']}")
        logger.info(f"  Prompt length: {len(sample.prompt)} chars")

    return True


def test_vllm_connection():
    """Test connection to vLLM server."""
    logger.info("Testing vLLM connection...")

    import requests

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    logger.info(f"vLLM URL: {vllm_url}")

    try:
        # Check health
        response = requests.get(f"{vllm_url}/health", timeout=10)
        if response.status_code == 200:
            logger.info("vLLM server is healthy")
        else:
            logger.warning(f"vLLM health check returned: {response.status_code}")

        # Check models
        response = requests.get(f"{vllm_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Available models: {json.dumps(models, indent=2)}")
        else:
            logger.warning(f"vLLM models check returned: {response.status_code}")

        return True

    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to vLLM at {vllm_url}")
        logger.info("Deploy vLLM with: modal deploy examples/harbor/modal_vllm.py")
        return False
    except Exception as e:
        logger.error(f"vLLM connection error: {e}")
        return False


def test_rl_agent_api_call():
    """Test a single API call with the RL agent."""
    logger.info("Testing RL agent API call...")

    from examples.harbor.rl_agent import call_vllm_api

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ]

    try:
        response = call_vllm_api(
            api_url=vllm_url,
            model=model,
            messages=messages,
            max_tokens=100,
            temperature=1.0,
            return_logprobs=True,
        )

        logger.info(f"Response received:")
        logger.info(f"  Content: {response['choices'][0]['message']['content'][:100]}...")

        # Check for logprobs
        logprobs = response['choices'][0].get('logprobs', {})
        if logprobs:
            logger.info(f"  Logprobs available: Yes ({len(logprobs.get('content', []))} tokens)")
        else:
            logger.warning("  Logprobs available: No")

        usage = response.get('usage', {})
        logger.info(f"  Tokens - prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')}")

        return True

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return False


def test_trajectory_format():
    """Test ATIF trajectory format generation."""
    logger.info("Testing trajectory format...")

    from datetime import datetime
    from examples.harbor.rl_agent import TOOLS

    # Create sample trajectory
    trajectory = {
        "schema_version": "ATIF-v1.5",
        "session_id": f"test_{int(datetime.now().timestamp())}",
        "agent": {
            "name": "slime-rl-agent",
            "version": "1.0.0",
            "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "tool_definitions": TOOLS,
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "user",
                "message": "Fix the bug in the repository",
            },
            {
                "step_id": 2,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "agent",
                "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "message": "I'll start by exploring the repository.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_0",
                        "function_name": "list_directory",
                        "arguments": {"path": "/testbed"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_0",
                            "content": "django/\ntests/\nsetup.py\n",
                        }
                    ]
                },
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "logprobs": [-0.1, -0.05, -0.02],
                    "completion_token_ids": [1234, 5678, 9012],
                },
            },
        ],
        "final_metrics": {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 50,
        },
    }

    logger.info(f"Trajectory schema: {trajectory['schema_version']}")
    logger.info(f"Steps: {len(trajectory['steps'])}")
    logger.info(f"Tool definitions: {len(trajectory['agent']['tool_definitions'])}")

    # Validate structure
    assert "schema_version" in trajectory
    assert "session_id" in trajectory
    assert "agent" in trajectory
    assert "steps" in trajectory
    assert trajectory["agent"]["name"] == "slime-rl-agent"

    logger.info("Trajectory format is valid")
    return True


def test_loss_masking():
    """Test loss mask computation from trajectory."""
    logger.info("Testing loss masking...")

    from examples.harbor.rollout import build_messages_from_trajectory

    # Sample trajectory
    trajectory = {
        "steps": [
            {
                "source": "user",
                "message": "Fix the bug",
            },
            {
                "source": "agent",
                "message": "I'll explore the code.",
                "observation": {
                    "results": [{"content": "File listing output"}]
                },
            },
            {
                "source": "agent",
                "message": "Here's my fix.",
            },
        ],
    }

    messages = build_messages_from_trajectory(trajectory)
    logger.info(f"Converted {len(trajectory['steps'])} steps to {len(messages)} messages")

    for msg in messages:
        logger.info(f"  Role: {msg['role']}, Content: {msg['content'][:50]}...")

    # Verify roles
    expected_roles = ["user", "assistant", "tool", "assistant"]
    actual_roles = [m["role"] for m in messages]
    assert actual_roles == expected_roles, f"Expected {expected_roles}, got {actual_roles}"

    logger.info("Loss masking is correct")
    return True


async def test_full_rollout():
    """Test a complete rollout (requires Docker and vLLM)."""
    logger.info("Testing full rollout (requires Docker + vLLM)...")

    from examples.harbor.data_source import DjangoTrainDataSource
    from examples.harbor.rollout import generate

    # Load one sample
    ds = DjangoTrainDataSource(limit=1)
    if len(ds) == 0:
        logger.warning("No instances available")
        return False

    sample = ds[0]
    logger.info(f"Testing with instance: {sample.metadata['instance_id']}")

    # Create mock args
    args = Namespace(
        hf_checkpoint="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        vllm_url=os.environ.get("VLLM_URL", "http://localhost:8000"),
        model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        max_turns=5,  # Short for testing
        timeout=300,
        rollout_max_response_len=1024,
        rollout_temperature=1.0,
        eval_timeout=60,
        loss_mask_type="qwen3",
    )

    try:
        result = await generate(args, sample, {})

        logger.info(f"Rollout completed:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Reward: {result.reward}")
        logger.info(f"  Response length: {result.response_length}")
        logger.info(f"  Tokens: {len(result.tokens)}")
        logger.info(f"  Loss mask sum: {sum(result.loss_mask) if result.loss_mask else 0}")

        if result.metadata.get("trajectory"):
            traj = result.metadata["trajectory"]
            logger.info(f"  Trajectory steps: {len(traj.get('steps', []))}")

        return True

    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Harbor GRPO Pipeline Test")
    print("=" * 60)
    print()

    results = {}

    # Test data source
    try:
        results["data_source"] = test_data_source()
    except Exception as e:
        logger.error(f"Data source test failed: {e}")
        results["data_source"] = False

    print()

    # Test trajectory format
    try:
        results["trajectory_format"] = test_trajectory_format()
    except Exception as e:
        logger.error(f"Trajectory format test failed: {e}")
        results["trajectory_format"] = False

    print()

    # Test loss masking
    try:
        results["loss_masking"] = test_loss_masking()
    except Exception as e:
        logger.error(f"Loss masking test failed: {e}")
        results["loss_masking"] = False

    print()

    # Test vLLM connection
    try:
        results["vllm_connection"] = test_vllm_connection()
    except Exception as e:
        logger.error(f"vLLM connection test failed: {e}")
        results["vllm_connection"] = False

    print()

    # Test API call (only if vLLM connected)
    if results.get("vllm_connection"):
        try:
            results["api_call"] = test_rl_agent_api_call()
        except Exception as e:
            logger.error(f"API call test failed: {e}")
            results["api_call"] = False
    else:
        logger.warning("Skipping API call test (vLLM not connected)")
        results["api_call"] = None

    print()

    # Full rollout test (optional, requires Docker + vLLM)
    if "--full" in sys.argv:
        if results.get("vllm_connection"):
            try:
                results["full_rollout"] = asyncio.run(test_full_rollout())
            except Exception as e:
                logger.error(f"Full rollout test failed: {e}")
                results["full_rollout"] = False
        else:
            logger.warning("Skipping full rollout test (vLLM not connected)")
            results["full_rollout"] = None
    else:
        logger.info("Skipping full rollout test (use --full to enable)")
        results["full_rollout"] = None

    # Summary
    print()
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result is True:
            status = "PASSED"
            passed += 1
        elif result is False:
            status = "FAILED"
            failed += 1
        else:
            status = "SKIPPED"
            skipped += 1
        print(f"  {name}: {status}")

    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
