"""
Rollout function for SLiME GRPO training using Harbor-style agent.

This module implements the rollout interface expected by SLiME, using
the RL agent that records ATIF trajectories with token_ids and logprobs.
"""

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .rl_agent import RLAgentConfig, run_rl_agent
from .swebench_env import setup_container, cleanup_container
from .rewards import evaluate_patch

logger = logging.getLogger(__name__)

# Global state
TOKENIZER = None
MASK_GENERATOR = None


def get_tokenizer(args):
    """Get or create tokenizer singleton."""
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return TOKENIZER


def get_mask_generator(args):
    """Get or create mask generator singleton."""
    global MASK_GENERATOR
    if MASK_GENERATOR is None:
        tokenizer = get_tokenizer(args)
        tokenizer_type = getattr(args, "loss_mask_type", "qwen3")
        MASK_GENERATOR = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=tokenizer_type)
    return MASK_GENERATOR


def build_messages_from_trajectory(trajectory: dict) -> list[dict]:
    """
    Convert ATIF trajectory to chat messages format for loss masking.

    Maps trajectory steps to messages with proper roles:
    - User steps -> user messages (mask=0)
    - Agent steps -> assistant messages (mask=1 for trainable)
    - Tool observations -> tool messages (mask=0)
    """
    messages = []

    for step in trajectory.get("steps", []):
        source = step.get("source", "")
        message = step.get("message", "")

        if not message:
            continue

        if source == "user":
            messages.append({"role": "user", "content": message})
        elif source == "agent":
            messages.append({"role": "assistant", "content": message})

            # If there's an observation with tool results, add as tool message
            observation = step.get("observation", {})
            if observation and "results" in observation:
                tool_content = "\n\n".join([
                    r.get("content", "")
                    for r in observation["results"]
                    if r.get("content")
                ])
                if tool_content:
                    messages.append({"role": "tool", "content": tool_content})

    return messages


def compute_loss_mask_for_trajectory(
    args,
    prompt: str,
    trajectory: dict,
) -> tuple[list[int], list[int], int]:
    """
    Compute loss mask from ATIF trajectory.

    Uses MultiTurnLossMaskGenerator to properly mask:
    - Assistant messages: loss_mask = 1 (train on these)
    - Tool responses: loss_mask = 0 (don't train)
    - User messages: loss_mask = 0 (don't train)

    Returns:
        (token_ids, loss_mask, response_length)
    """
    mask_generator = get_mask_generator(args)

    # Convert trajectory to messages
    messages = build_messages_from_trajectory(trajectory)

    if not messages:
        # Empty trajectory - return empty
        tokenizer = get_tokenizer(args)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return prompt_ids, [], 0

    # Generate loss mask using message format
    token_ids, loss_mask = mask_generator.get_loss_mask(messages)

    # Response length is the portion with loss_mask = 1
    response_length = sum(loss_mask)

    return token_ids, loss_mask, response_length


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Generate SWE-bench solution using RL agent with ATIF trajectory.

    This is the main rollout function called by SLiME.
    Path: examples.harbor.rollout:generate

    Args:
        args: Training arguments
        sample: Sample with prompt and metadata
        sampling_params: Sampling parameters

    Returns:
        Sample with tokens, response, loss_mask, and reward
    """
    instance_id = sample.metadata.get("instance_id", "unknown")
    container_name = None

    try:
        # Get vLLM URL from args or environment
        vllm_url = getattr(args, "vllm_url", None) or os.environ.get("VLLM_URL", "http://localhost:8000")

        # Setup agent config
        config = RLAgentConfig(
            model_name=getattr(args, "model_name", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            api_base_url=vllm_url,
            max_turns=getattr(args, "max_turns", 50),
            timeout=getattr(args, "timeout", 1800),
            max_tokens=getattr(args, "rollout_max_response_len", 4096),
            temperature=getattr(args, "rollout_temperature", 1.0),  # GRPO uses temp=1.0
            return_logprobs=True,  # Enable for RL training
        )

        # Setup container
        container_name = await asyncio.to_thread(
            setup_container,
            instance_id,
            suffix=f"_{sample.index}",
        )

        # Run RL agent
        result = await asyncio.to_thread(
            run_rl_agent,
            container_name,
            instance_id,
            sample.prompt,
            config,
        )

        # Compute loss mask from trajectory
        token_ids, loss_mask, response_length = compute_loss_mask_for_trajectory(
            args, sample.prompt, result.trajectory
        )

        sample.tokens = token_ids
        sample.loss_mask = loss_mask
        sample.response_length = response_length

        # Store trajectory in metadata for analysis
        sample.metadata["trajectory"] = result.trajectory
        sample.metadata["final_metrics"] = result.trajectory.get("final_metrics", {})

        # Set response text (assistant messages only)
        response_parts = []
        for step in result.trajectory.get("steps", []):
            if step.get("source") == "agent":
                response_parts.append(step.get("message", ""))
        sample.response = "\n".join(response_parts)

        # Evaluate patch for reward
        if result.patch:
            resolved = await asyncio.to_thread(
                evaluate_patch,
                instance_id,
                result.patch,
                timeout=getattr(args, "eval_timeout", 900),
            )
            sample.reward = 1.0 if resolved else -1.0
            sample.metadata["patch"] = result.patch
            sample.metadata["resolved"] = resolved
            logger.info(f"[{instance_id}] Resolved: {resolved}")
        else:
            sample.reward = -1.0
            sample.metadata["patch"] = ""
            sample.metadata["resolved"] = False
            logger.info(f"[{instance_id}] No patch generated")

        # Set status
        if result.exit_status == "timeout":
            sample.status = Sample.Status.TRUNCATED
        elif result.exit_status == "failed":
            sample.status = Sample.Status.FAILED
        else:
            sample.status = Sample.Status.COMPLETED

    except Exception as e:
        logger.error(f"[{instance_id}] Rollout failed: {e}")
        sample.reward = -1.0
        sample.status = Sample.Status.FAILED
        sample.metadata["error"] = str(e)

        # Set minimal response
        tokenizer = get_tokenizer(args)
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
        sample.tokens = prompt_ids
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []

    finally:
        if container_name:
            await asyncio.to_thread(cleanup_container, container_name)

    return sample


async def generate_group(
    args: Namespace,
    group: list[Sample],
    sampling_params: dict[str, Any],
) -> list[Sample]:
    """
    Generate solutions for a group of samples (same instance, multiple attempts).

    This enables parallel execution within a group for GRPO.
    """
    tasks = [
        asyncio.create_task(generate(args, sample, sampling_params))
        for sample in group
    ]
    return await asyncio.gather(*tasks)


# Reward function for SLiME rm_hub (if needed)
async def reward_func(args, sample, **kwargs) -> float:
    """
    Compute reward for a sample.

    Returns the pre-computed reward from rollout.
    """
    return sample.reward if sample.reward is not None else -1.0
