#!/usr/bin/env python3
"""
Test a single rollout with detailed logging.

Usage:
    python examples/qwen_swe/test_single_rollout.py \
        --modal-url https://YOUR_URL \
        --max-turns 50
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Test single rollout")
    parser.add_argument("--modal-url", type=str, required=True, help="Modal vLLM URL")
    parser.add_argument("--max-turns", type=int, default=50, help="Max agent turns")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    parser.add_argument("--instance-id", type=str, default=None, help="Specific instance ID")

    args = parser.parse_args()

    # Set environment
    os.environ["VLLM_URL"] = args.modal_url

    from argparse import Namespace
    from examples.qwen_swe.data_source import create_data_source, load_instances_from_file
    from examples.qwen_swe.prompts import format_swebench_prompt
    from examples.qwen_swe.rollout import generate
    from slime.utils.types import Sample

    # Load data
    logger.info("Loading SWE-bench instances...")
    data_source = create_data_source(Namespace(
        swe_instance_file="os.path.join(os.path.dirname(__file__), "data", "train_201_django.txt")",
        swe_split=None,
        n_samples_per_prompt=1,
        rollout_shuffle=False,
    ))

    # Get sample
    if args.instance_id:
        instances = load_instances_from_file("os.path.join(os.path.dirname(__file__), "data", "train_201_django.txt")")
        instance = next((i for i in instances if i.instance_id == args.instance_id), None)
        if not instance:
            logger.error(f"Instance {args.instance_id} not found")
            return
        sample = Sample(
            prompt=format_swebench_prompt(instance.problem_statement),
            group_index=0,
            index=0,
            metadata={
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
            },
        )
    else:
        groups = data_source.get_samples(1)
        sample = groups[0][0]

    instance_id = sample.metadata.get("instance_id")
    logger.info(f"Testing instance: {instance_id}")
    logger.info(f"Max turns: {args.max_turns}")
    logger.info(f"Timeout: {args.timeout}s")

    # Create args for rollout
    rollout_args = Namespace(
        hf_checkpoint="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        vllm_url=args.modal_url,
        qwen_model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        qwen_max_turns=args.max_turns,
        qwen_timeout=args.timeout,
        eval_timeout=600,
        loss_mask_type="qwen3",
    )

    # Run rollout
    logger.info("=" * 60)
    logger.info("Starting rollout...")
    logger.info("=" * 60)

    result = await generate(rollout_args, sample, {})

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Instance: {instance_id}")
    logger.info(f"Status: {result.status}")
    logger.info(f"Reward: {result.reward}")
    logger.info(f"Response length: {result.response_length}")
    logger.info(f"Tokens: {len(result.tokens) if result.tokens else 0}")

    if result.loss_mask:
        mask_ones = sum(result.loss_mask)
        logger.info(f"Loss mask - 1s: {mask_ones}, 0s: {len(result.loss_mask) - mask_ones}")
    else:
        logger.info("Loss mask: Not computed")

    if result.metadata.get("patch"):
        logger.info(f"Patch length: {len(result.metadata['patch'])} chars")
        logger.info(f"Resolved: {result.metadata.get('resolved', False)}")
        logger.info(f"Patch applied: {result.metadata.get('patch_applied', False)}")

        # Print first 500 chars of patch
        patch = result.metadata['patch']
        if len(patch) > 500:
            logger.info(f"Patch preview:\n{patch[:500]}...")
        else:
            logger.info(f"Patch:\n{patch}")
    else:
        logger.info("No patch generated")

    if result.response:
        logger.info(f"Response preview: {result.response[:500]}..." if len(result.response) > 500 else f"Response: {result.response}")

    logger.info("=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
