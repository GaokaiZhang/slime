#!/usr/bin/env python3
"""
Test the new SLiME-integrated GRPO setup.

This script tests:
1. Data source loading
2. Rollout function with loss masking
3. Reward computation via swebench.harness

Usage:
    # First deploy Modal inference server:
    modal deploy examples/qwen_swe/modal_inference.py

    # Then run test:
    python examples/qwen_swe/test_slime_integration.py --modal-url https://YOUR_URL
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_data_source():
    """Test that data source loads correctly."""
    from argparse import Namespace
    from examples.qwen_swe.data_source import create_data_source

    logger.info("Testing data source...")

    args = Namespace(
        swe_instance_file="os.path.join(os.path.dirname(__file__), "data", "train_201_django.txt")",
        swe_split=None,
        n_samples_per_prompt=2,
        rollout_shuffle=False,
    )

    data_source = create_data_source(args)

    # Get a batch
    groups = data_source.get_samples(num_samples=1)

    assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
    group = groups[0]
    assert len(group) == 2, f"Expected 2 samples per group, got {len(group)}"

    sample = group[0]
    logger.info(f"Instance ID: {sample.metadata.get('instance_id')}")
    logger.info(f"Prompt length: {len(sample.prompt)} chars")
    logger.info(f"Group index: {sample.group_index}")
    logger.info(f"Sample index: {sample.index}")

    logger.info("Data source test PASSED")
    return data_source, group


def test_loss_mask_generator():
    """Test that loss mask generator works correctly."""
    from argparse import Namespace
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer

    logger.info("Testing loss mask generator...")

    # Load tokenizer
    model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    tokenizer = load_tokenizer(model_name, trust_remote_code=True)

    # Create mask generator
    mask_gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    # Test messages
    messages = [
        {"role": "user", "content": "Fix the bug in the code."},
        {"role": "assistant", "content": "I'll analyze the code and fix it."},
        {"role": "tool", "content": "File content: def foo(): pass"},
        {"role": "assistant", "content": "I found the issue. Here's the fix."},
    ]

    token_ids, loss_mask = mask_gen.get_loss_mask(messages)

    logger.info(f"Token IDs length: {len(token_ids)}")
    logger.info(f"Loss mask length: {len(loss_mask)}")
    logger.info(f"Tokens with loss_mask=1: {sum(loss_mask)}")
    logger.info(f"Tokens with loss_mask=0: {len(loss_mask) - sum(loss_mask)}")

    # Verify that we have both 0s and 1s
    assert sum(loss_mask) > 0, "Expected some tokens with loss_mask=1"
    assert sum(loss_mask) < len(loss_mask), "Expected some tokens with loss_mask=0"

    logger.info("Loss mask generator test PASSED")
    return mask_gen


async def test_rollout(modal_url: str, sample):
    """Test the rollout function with Modal inference."""
    from argparse import Namespace
    from examples.qwen_swe.rollout import generate

    logger.info("Testing rollout function...")
    logger.info(f"Modal URL: {modal_url}")

    # Create args
    args = Namespace(
        hf_checkpoint="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        vllm_url=modal_url,
        qwen_model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        qwen_max_turns=10,  # Reduced for testing
        qwen_timeout=600,   # 10 minutes for testing
        eval_timeout=300,   # 5 minutes
        loss_mask_type="qwen3",
    )

    # Set environment
    os.environ["VLLM_URL"] = modal_url

    # Run rollout
    sampling_params = {"temperature": 0.7, "top_p": 0.95}

    logger.info(f"Running rollout for instance: {sample.metadata.get('instance_id')}")

    result = await generate(args, sample, sampling_params)

    logger.info(f"Rollout completed!")
    logger.info(f"Status: {result.status}")
    logger.info(f"Reward: {result.reward}")
    logger.info(f"Response length: {result.response_length}")
    logger.info(f"Tokens: {len(result.tokens) if result.tokens else 0}")
    logger.info(f"Loss mask: {len(result.loss_mask) if result.loss_mask else 0}")

    if result.loss_mask:
        mask_ones = sum(result.loss_mask)
        logger.info(f"Loss mask 1s: {mask_ones}, 0s: {len(result.loss_mask) - mask_ones}")

    if result.metadata.get("patch"):
        logger.info(f"Patch generated: {len(result.metadata['patch'])} chars")
        logger.info(f"Resolved: {result.metadata.get('resolved', False)}")
    else:
        logger.info("No patch generated")

    logger.info("Rollout test completed")
    return result


def test_qwen_agent_config(modal_url: str):
    """Test qwen_agent configuration."""
    from examples.qwen_swe.qwen_agent import QwenAgentConfig

    logger.info("Testing QwenAgentConfig...")

    config = QwenAgentConfig(
        model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        api_base_url=modal_url,
        max_turns=10,
        timeout=600,
    )

    logger.info(f"Config created:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  API base: {config.api_base_url}")
    logger.info(f"  Max turns: {config.max_turns}")
    logger.info(f"  Timeout: {config.timeout}")

    logger.info("QwenAgentConfig test PASSED")
    return config


async def main():
    parser = argparse.ArgumentParser(description="Test SLiME-integrated GRPO setup")
    parser.add_argument("--modal-url", type=str, required=True, help="Modal vLLM server URL")
    parser.add_argument("--skip-rollout", action="store_true", help="Skip rollout test (requires Docker)")
    parser.add_argument("--instance-id", type=str, default=None, help="Specific instance ID to test")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Testing SLiME-integrated GRPO Setup")
    logger.info("=" * 60)

    # Test 1: Data source
    data_source, group = test_data_source()

    # Test 2: Loss mask generator
    test_loss_mask_generator()

    # Test 3: QwenAgentConfig
    test_qwen_agent_config(args.modal_url)

    # Test 4: Full rollout (if not skipped)
    if not args.skip_rollout:
        sample = group[0]
        if args.instance_id:
            # Find specific instance
            from examples.qwen_swe.data_source import load_instances_from_file
            instances = load_instances_from_file("os.path.join(os.path.dirname(__file__), "data", "train_201_django.txt")")
            for inst in instances:
                if inst.instance_id == args.instance_id:
                    from examples.qwen_swe.prompts import format_swebench_prompt
                    from slime.utils.types import Sample
                    sample = Sample(
                        prompt=format_swebench_prompt(inst.problem_statement),
                        group_index=0,
                        index=0,
                        metadata={
                            "instance_id": inst.instance_id,
                            "repo": inst.repo,
                            "base_commit": inst.base_commit,
                        },
                    )
                    break

        result = await test_rollout(args.modal_url, sample)

        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Instance: {sample.metadata.get('instance_id')}")
        logger.info(f"Status: {result.status}")
        logger.info(f"Reward: {result.reward}")
        logger.info(f"Resolved: {result.metadata.get('resolved', False)}")
        logger.info(f"Loss mask computed: {len(result.loss_mask) > 0 if result.loss_mask else False}")
    else:
        logger.info("Skipping rollout test (--skip-rollout)")

    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
