#!/usr/bin/env python
"""
Test GRPO rollout with Harbor evaluation.

This script tests the full pipeline:
1. Load SWE-bench instance
2. Run vLLM agent to generate solution
3. Evaluate with Harbor (if tasks available) or heuristic
4. Verify SLiME data format

Run with:
    export VLLM_URL="https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run"
    python examples/grpo/test_grpo_rollout.py
"""

import asyncio
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add Harbor submodule
_HARBOR_SRC = _PROJECT_ROOT / "submodules" / "harbor" / "src"
if str(_HARBOR_SRC) not in sys.path:
    sys.path.insert(0, str(_HARBOR_SRC))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_rollout():
    """Test a single GRPO rollout with evaluation."""
    from examples.grpo.data_source import DjangoTrainDataSource
    from examples.grpo.rollout import generate
    from slime.utils.types import Sample

    print("=" * 70)
    print("GRPO Rollout Test with Harbor Evaluation")
    print("=" * 70)
    print()

    # Check vLLM URL
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    print(f"vLLM URL: {vllm_url}")
    print(f"Model: {model_name}")
    print()

    # Load data
    print("1. Loading data source...")
    ds = DjangoTrainDataSource(limit=1)
    if len(ds) == 0:
        print("ERROR: No instances available")
        return False

    sample = ds[0]
    instance_id = sample.metadata.get("instance_id", "unknown")
    print(f"   Instance: {instance_id}")
    print(f"   Prompt length: {len(sample.prompt)} chars")
    print()

    # Create args
    args = Namespace(
        hf_checkpoint=model_name,
        vllm_url=vllm_url,
        model_name=model_name,
        max_turns=5,  # Short for testing
        rollout_max_response_len=2048,
        rollout_temperature=1.0,
        use_harbor_eval=True,  # Try Harbor evaluation
    )

    # Run rollout
    print("2. Running agent rollout...")
    print(f"   Max turns: {args.max_turns}")
    print()

    try:
        result = await generate(args, sample, {})
    except Exception as e:
        print(f"ERROR: Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check results
    print("3. Rollout Results:")
    print(f"   Status: {result.status}")
    print(f"   Reward: {result.reward}")
    print()

    # Verify SLiME format
    print("4. SLiME Format Verification:")
    errors = []

    # Check tokens
    if not result.tokens:
        errors.append("tokens is empty")
    else:
        print(f"   tokens: {len(result.tokens)} tokens")

    # Check response_length
    if result.response_length <= 0:
        errors.append("response_length is 0")
    else:
        print(f"   response_length: {result.response_length}")

    # Check loss_mask
    if not result.loss_mask:
        errors.append("loss_mask is empty")
    elif len(result.loss_mask) != result.response_length:
        errors.append(f"loss_mask length ({len(result.loss_mask)}) != response_length ({result.response_length})")
    else:
        print(f"   loss_mask: {len(result.loss_mask)} values (all 1s)")

    # Check rollout_log_probs
    if not result.rollout_log_probs:
        errors.append("rollout_log_probs is empty")
    elif len(result.rollout_log_probs) != result.response_length:
        errors.append(f"rollout_log_probs length ({len(result.rollout_log_probs)}) != response_length ({result.response_length})")
    else:
        print(f"   rollout_log_probs: {len(result.rollout_log_probs)} values")

    # Check reward
    if result.reward is None:
        errors.append("reward is None")
    elif not isinstance(result.reward, (int, float)):
        errors.append(f"reward is not a number: {type(result.reward)}")
    else:
        print(f"   reward: {result.reward}")

    # Check that tokens = prompt + response
    prompt_length = result.metadata.get("prompt_length", 0)
    if prompt_length > 0:
        total_expected = prompt_length + result.response_length
        if len(result.tokens) != total_expected:
            errors.append(f"tokens length ({len(result.tokens)}) != prompt ({prompt_length}) + response ({result.response_length})")
        else:
            print(f"   tokens = prompt ({prompt_length}) + response ({result.response_length}): CORRECT")

    print()

    if errors:
        print("5. ERRORS FOUND:")
        for e in errors:
            print(f"   - {e}")
        return False
    else:
        print("5. All format checks PASSED!")

    # Show sample data
    print()
    print("6. Sample Data:")
    print(f"   First 10 tokens: {result.tokens[:10]}")
    print(f"   First 10 logprobs: {[round(lp, 4) for lp in result.rollout_log_probs[:10]]}")

    # Check trajectory
    trajectory = result.metadata.get("trajectory", {})
    steps = trajectory.get("steps", [])
    print(f"   Trajectory steps: {len(steps)}")

    # Check patch
    patch = result.metadata.get("patch", "")
    if patch:
        print(f"   Patch: {len(patch)} chars")
        print(f"   Resolved: {result.metadata.get('resolved', False)}")
    else:
        print("   Patch: None (agent did not submit)")

    print()
    print("=" * 70)
    print("GRPO Rollout Test: SUCCESS")
    print("=" * 70)
    return True


async def test_multiple_rollouts(n_samples: int = 3):
    """Test multiple rollouts (group) for GRPO."""
    from examples.grpo.data_source import DjangoTrainDataSource
    from examples.grpo.rollout import generate_group
    from slime.utils.types import Sample

    print()
    print("=" * 70)
    print(f"GRPO Group Rollout Test (n_samples={n_samples})")
    print("=" * 70)
    print()

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

    # Load data
    print("1. Loading data source...")
    ds = DjangoTrainDataSource(limit=1)
    if len(ds) == 0:
        print("ERROR: No instances available")
        return False

    base_sample = ds[0]
    instance_id = base_sample.metadata.get("instance_id", "unknown")
    print(f"   Instance: {instance_id}")
    print()

    # Create group of samples (same prompt, multiple attempts)
    print(f"2. Creating group of {n_samples} samples...")
    group = []
    for i in range(n_samples):
        sample = Sample(
            prompt=base_sample.prompt,
            group_index=0,
            index=i,
            metadata=base_sample.metadata.copy(),
        )
        group.append(sample)
    print()

    # Create args
    args = Namespace(
        hf_checkpoint=model_name,
        vllm_url=vllm_url,
        model_name=model_name,
        max_turns=3,  # Short for testing
        rollout_max_response_len=1024,
        rollout_temperature=1.0,
        use_harbor_eval=False,  # Disable Harbor eval for speed
    )

    # Run group rollout
    print("3. Running group rollout...")
    try:
        results = await generate_group(args, group, {})
    except Exception as e:
        print(f"ERROR: Group rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze results
    print()
    print("4. Group Results:")
    rewards = []
    for i, result in enumerate(results):
        rewards.append(result.reward)
        print(f"   Sample {i}: status={result.status}, reward={result.reward}, tokens={len(result.tokens)}")

    print()
    print("5. GRPO Statistics:")
    print(f"   Rewards: {rewards}")
    print(f"   Mean reward: {sum(rewards) / len(rewards):.3f}")
    print(f"   Reward variance: {sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards):.3f}")

    print()
    print("=" * 70)
    print("GRPO Group Rollout Test: SUCCESS")
    print("=" * 70)
    return True


async def main():
    """Run all tests."""
    # Test single rollout
    success1 = await test_single_rollout()

    if success1:
        # Test group rollout
        success2 = await test_multiple_rollouts(n_samples=2)
    else:
        success2 = False

    print()
    print("=" * 70)
    print("Final Results:")
    print(f"  Single rollout: {'PASSED' if success1 else 'FAILED'}")
    print(f"  Group rollout: {'PASSED' if success2 else 'FAILED'}")
    print("=" * 70)

    return success1 and success2


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
