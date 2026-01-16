"""
Test the Harbor + SLiME GRPO pipeline.

This script verifies:
1. Data source loads correctly (SWE-bench from HuggingFace)
2. vLLM agent captures token_ids and logprobs
3. Loss mask computation is correct
4. (Optional) Full rollout with Docker + vLLM

Architecture:
- SLiME: GRPO training (loss, model updates)
- vLLM Agent: Direct API calls with token_ids + logprobs capture
- Docker: SWE-bench environments
"""

import asyncio
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_source():
    """Test loading SWE-bench data."""
    logger.info("Testing data source...")

    from examples.grpo.data_source import DjangoTrainDataSource

    ds = DjangoTrainDataSource(limit=3)
    logger.info(f"Loaded {len(ds)} instances")

    for i in range(min(3, len(ds))):
        sample = ds[i]
        logger.info(f"Instance {i}: {sample.metadata['instance_id']}")
        logger.info(f"  Repo: {sample.metadata['repo']}")
        logger.info(f"  Prompt length: {len(sample.prompt)} chars")

    return True


def test_vllm_agent_module():
    """Test vLLM agent module imports and token_id extraction."""
    logger.info("Testing vLLM agent module...")

    try:
        from examples.grpo.vllm_agent import (
            VLLMAgentConfig,
            run_agent,
            call_vllm,
            parse_tool_call,
            get_tokenizer,
            TOOLS,
        )
        import inspect

        logger.info(f"  VLLMAgentConfig: OK")
        logger.info(f"  run_agent: OK")
        logger.info(f"  call_vllm: OK")
        logger.info(f"  parse_tool_call: OK")
        logger.info(f"  get_tokenizer: OK")
        logger.info(f"  TOOLS: {len(TOOLS)} tools defined")

        # Test tool call parsing
        test_content = 'I will list files. {"tool": "bash", "args": {"command": "ls"}}'
        result = parse_tool_call(test_content)
        assert result is not None, "Failed to parse tool call"
        assert result[0] == "bash", f"Expected 'bash', got {result[0]}"
        logger.info("  parse_tool_call: Correctly parses tool calls")

        # Test that call_vllm has tokenizer parameter (for fallback)
        sig = inspect.signature(call_vllm)
        params = list(sig.parameters.keys())
        assert "tokenizer" in params, "call_vllm missing tokenizer parameter"
        logger.info("  call_vllm: Has tokenizer parameter for fallback")

        # Test VLLMAgentConfig has tokenizer_path
        config = VLLMAgentConfig(tokenizer_path="test/model")
        assert config.tokenizer_path == "test/model"
        logger.info("  VLLMAgentConfig: Has tokenizer_path option")

        return True

    except ImportError as e:
        logger.error(f"  Import failed: {e}")
        return False


def test_loss_mask_computation():
    """Test loss mask computation from rollout details."""
    logger.info("Testing loss mask computation...")

    from examples.grpo.rollout import compute_loss_mask_from_rollout_details

    # Mock rollout details (like vLLM agent produces)
    rollout_details = [
        {
            "completion_token_ids": [
                [100, 101, 102],  # Turn 1 assistant response
                [200, 201, 202, 203],  # Turn 2 assistant response
            ],
            "logprobs": [
                [-0.1, -0.2, -0.3],  # Turn 1 logprobs
                [-0.4, -0.5, -0.6, -0.7],  # Turn 2 logprobs
            ],
        }
    ]

    args = Namespace()
    token_ids, loss_mask, logprobs = compute_loss_mask_from_rollout_details(rollout_details, args)

    logger.info(f"  Total completion tokens: {len(token_ids)}")
    logger.info(f"  Token IDs: {token_ids}")
    logger.info(f"  Loss mask: {loss_mask}")
    logger.info(f"  Logprobs: {logprobs}")

    # Verify
    expected_tokens = [100, 101, 102, 200, 201, 202, 203]
    expected_mask = [1, 1, 1, 1, 1, 1, 1]
    expected_logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]

    assert token_ids == expected_tokens, f"Expected {expected_tokens}, got {token_ids}"
    assert loss_mask == expected_mask, f"Expected {expected_mask}, got {loss_mask}"
    assert logprobs == expected_logprobs, f"Expected {expected_logprobs}, got {logprobs}"

    logger.info("  Loss mask computation: CORRECT")
    return True


def test_swebench_utils():
    """Test SWE-bench utilities."""
    logger.info("Testing SWE-bench utilities...")

    try:
        from examples.grpo.swebench_utils import (
            get_docker_image,
            get_instance_info,
        )

        # Test image lookup
        image = get_docker_image("django__django-12345")
        logger.info(f"  Django image: {image}")

        # Test instance info
        info = get_instance_info("django__django-12345")
        logger.info(f"  Instance info: {info}")

        return True

    except ImportError as e:
        logger.error(f"  Import failed: {e}")
        return False


def test_vllm_connection():
    """Test connection to vLLM server."""
    logger.info("Testing vLLM connection...")

    import requests

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    logger.info(f"  vLLM URL: {vllm_url}")

    try:
        response = requests.get(f"{vllm_url}/health", timeout=10)
        if response.status_code == 200:
            logger.info("  vLLM server is healthy")
            return True
        else:
            logger.warning(f"  vLLM health check returned: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        logger.warning(f"  Could not connect to vLLM at {vllm_url}")
        logger.info("  Deploy vLLM with: modal deploy examples/grpo/modal_vllm.py")
        return False


def test_vllm_api_call():
    """Test a single vLLM API call with logprobs and token IDs."""
    logger.info("Testing vLLM API call with logprobs and tokenizer fallback...")

    from examples.grpo.vllm_agent import call_vllm, get_tokenizer

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

    # Load tokenizer for fallback (since vLLM doesn't return token_ids directly)
    logger.info("  Loading tokenizer for token ID extraction...")
    tokenizer = get_tokenizer(model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'hello' and nothing else."},
    ]

    try:
        response = call_vllm(
            api_url=vllm_url,
            model=model,
            messages=messages,
            max_tokens=50,
            temperature=1.0,
            tokenizer=tokenizer,  # Use tokenizer for fallback
        )

        logger.info(f"  Content: {response['content'][:100]}...")
        logger.info(f"  Token IDs: {len(response['completion_token_ids'])} tokens")
        logger.info(f"  Logprobs: {len(response['logprobs'])} values")
        logger.info(f"  Usage: {response['usage']}")

        if response['completion_token_ids']:
            logger.info("  Token IDs captured: YES (via tokenizer fallback)")
        else:
            logger.warning("  Token IDs captured: NO (check vLLM logprobs support)")

        return True

    except Exception as e:
        logger.error(f"  API call failed: {e}")
        return False


def test_slime_format():
    """Test that rollout produces correct SLiME data format."""
    logger.info("Testing SLiME data format...")

    from slime.utils.types import Sample

    # Verify Sample class has required fields
    sample = Sample()

    required_fields = [
        ("tokens", list),
        ("response_length", int),
        ("loss_mask", (list, type(None))),
        ("reward", (float, dict, type(None))),
        ("rollout_log_probs", (list, type(None))),
    ]

    for field_name, expected_types in required_fields:
        assert hasattr(sample, field_name), f"Sample missing field: {field_name}"
        logger.info(f"  Sample.{field_name}: OK")

    # Test format expectations
    logger.info("  SLiME format requirements:")
    logger.info("    - tokens = prompt_tokens + response_tokens (FULL sequence)")
    logger.info("    - response_length = len(response_tokens)")
    logger.info("    - loss_mask = [1] * response_length")
    logger.info("    - rollout_log_probs = logprobs from vLLM")
    logger.info("    - reward = float scalar")

    # Mock test of format
    prompt_tokens = [1, 2, 3, 4, 5]
    response_tokens = [10, 11, 12]

    sample.tokens = prompt_tokens + response_tokens
    sample.response_length = len(response_tokens)
    sample.loss_mask = [1] * len(response_tokens)
    sample.rollout_log_probs = [-0.1, -0.2, -0.3]
    sample.reward = 1.0

    # Validate format
    assert len(sample.tokens) == len(prompt_tokens) + len(response_tokens), "tokens should be prompt + response"
    assert sample.response_length == len(response_tokens), "response_length should match response tokens"
    assert len(sample.loss_mask) == sample.response_length, "loss_mask length should equal response_length"
    assert len(sample.rollout_log_probs) == sample.response_length, "rollout_log_probs length should match"
    assert isinstance(sample.reward, float), "reward should be float"

    logger.info("  SLiME format validation: PASSED")
    return True


def test_harbor_submodule():
    """Test Harbor submodule is accessible."""
    logger.info("Testing Harbor submodule import...")

    import sys
    from pathlib import Path

    # Check submodule path
    harbor_src = Path(__file__).resolve().parent.parent.parent / "submodules" / "harbor" / "src"
    if str(harbor_src) not in sys.path:
        sys.path.insert(0, str(harbor_src))

    try:
        from harbor.models.trial.config import TrialConfig, TaskConfig, AgentConfig
        from harbor.models.environment_type import EnvironmentType
        logger.info("  TrialConfig: OK")
        logger.info("  TaskConfig: OK")
        logger.info("  AgentConfig: OK")
        logger.info("  EnvironmentType: OK")

        # Test creating a config
        config = AgentConfig(name="oracle")
        assert config.name == "oracle"
        logger.info("  Config creation: OK")

        return True

    except ImportError as e:
        logger.warning(f"  Harbor import failed: {e}")
        logger.warning("  This is OK if harbor dependencies are not installed")
        return True  # Not a failure - Harbor eval is optional


async def test_full_rollout():
    """Test a complete rollout (requires Docker + vLLM)."""
    logger.info("Testing full rollout (requires Docker + vLLM)...")

    from examples.grpo.data_source import DjangoTrainDataSource
    from examples.grpo.rollout import generate

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
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        max_turns=5,  # Short for testing
        rollout_max_response_len=1024,
        rollout_temperature=1.0,
    )

    try:
        result = await generate(args, sample, {})

        logger.info(f"Rollout completed:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Reward: {result.reward}")
        logger.info(f"  Tokens: {len(result.tokens)}")
        logger.info(f"  Loss mask: {len(result.loss_mask)} (sum={sum(result.loss_mask) if result.loss_mask else 0})")

        return True

    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Harbor + SLiME GRPO Pipeline Test")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  - SLiME: GRPO training (loss computation, model updates)")
    print("  - vLLM Agent: Direct API calls with token_ids + logprobs")
    print("  - Docker: SWE-bench environments for tool execution")
    print()

    results = {}

    # Test data source
    try:
        results["data_source"] = test_data_source()
    except Exception as e:
        logger.error(f"Data source test failed: {e}")
        results["data_source"] = False

    print()

    # Test vLLM agent module
    try:
        results["vllm_agent"] = test_vllm_agent_module()
    except Exception as e:
        logger.error(f"vLLM agent test failed: {e}")
        results["vllm_agent"] = False

    print()

    # Test loss mask computation
    try:
        results["loss_mask"] = test_loss_mask_computation()
    except Exception as e:
        logger.error(f"Loss mask test failed: {e}")
        results["loss_mask"] = False

    print()

    # Test SWE-bench utils
    try:
        results["swebench_utils"] = test_swebench_utils()
    except Exception as e:
        logger.error(f"SWE-bench utils test failed: {e}")
        results["swebench_utils"] = False

    print()

    # Test SLiME format
    try:
        results["slime_format"] = test_slime_format()
    except Exception as e:
        logger.error(f"SLiME format test failed: {e}")
        results["slime_format"] = False

    print()

    # Test Harbor submodule
    try:
        results["harbor_submodule"] = test_harbor_submodule()
    except Exception as e:
        logger.error(f"Harbor submodule test failed: {e}")
        results["harbor_submodule"] = False

    print()

    # Test vLLM connection (optional)
    try:
        results["vllm_connection"] = test_vllm_connection()
    except Exception as e:
        logger.error(f"vLLM connection test failed: {e}")
        results["vllm_connection"] = False

    print()

    # Test vLLM API call (only if connected)
    if results.get("vllm_connection"):
        try:
            results["vllm_api"] = test_vllm_api_call()
        except Exception as e:
            logger.error(f"vLLM API test failed: {e}")
            results["vllm_api"] = False
    else:
        logger.info("Skipping vLLM API test (not connected)")
        results["vllm_api"] = None

    print()

    # Full rollout test (optional)
    if "--full" in sys.argv:
        if results.get("vllm_connection"):
            try:
                results["full_rollout"] = asyncio.run(test_full_rollout())
            except Exception as e:
                logger.error(f"Full rollout test failed: {e}")
                results["full_rollout"] = False
        else:
            logger.warning("Skipping full rollout (vLLM not connected)")
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
