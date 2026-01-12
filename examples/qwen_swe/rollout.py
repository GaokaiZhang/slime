"""
SWE-bench rollout function for SLiME.

Implements agent interaction with Docker containers using qwen-code CLI.
Properly sets loss_mask for tool responses (0) vs model responses (1).
"""

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .qwen_agent import (
    QwenAgentConfig,
    setup_container,
    run_qwen_agent,
    cleanup_container,
)
from .rewards import evaluate_with_swebench_harness

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


def build_messages_from_events(events: list[dict]) -> list[dict]:
    """
    Convert qwen-code events to chat messages format for loss masking.

    Handles two event formats:
    1. New format: {"type": "assistant|user", "message": {...}, ...}
    2. Old format: {"type": "message", "message": {"role": "user|model", "parts": [...]}}

    Returns list of messages with role (user/assistant/tool) and content.
    """
    messages = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type", "")

        # Skip system events
        if event_type == "system":
            continue

        # Handle new format: type is "assistant" or "user"
        if event_type in ("assistant", "user"):
            message = event.get("message", {})
            content = ""
            has_tool_response = False
            has_tool_call = False

            if message:
                role = message.get("role", event_type)

                # New format: message.content can be a string or list
                msg_content = message.get("content", "")
                if isinstance(msg_content, str):
                    content = msg_content
                elif isinstance(msg_content, list):
                    # Content is list of content blocks
                    content_parts = []
                    for block in msg_content:
                        if isinstance(block, str):
                            content_parts.append(block)
                        elif isinstance(block, dict):
                            if block.get("type") == "text":
                                content_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                has_tool_call = True
                                content_parts.append(f"<tool_call>{block.get('name', '')}</tool_call>")
                            elif block.get("type") == "tool_result":
                                has_tool_response = True
                                result = block.get("content", "")
                                if isinstance(result, str) and len(result) > 2000:
                                    result = result[:2000] + "..."
                                content_parts.append(result)
                    content = "\n".join(content_parts)

                # Old format: message.parts
                if not content:
                    parts = message.get("parts", [])
                    content_parts = []
                    for part in parts:
                        if isinstance(part, str):
                            content_parts.append(part)
                        elif isinstance(part, dict):
                            if "text" in part:
                                content_parts.append(part["text"])
                            elif "functionCall" in part:
                                has_tool_call = True
                                fc = part["functionCall"]
                                content_parts.append(f"<tool_call>{fc.get('name', '')}</tool_call>")
                            elif "functionResponse" in part:
                                has_tool_response = True
                                fr = part["functionResponse"]
                                resp = fr.get("response", {})
                                result = resp.get("result", str(resp)) if isinstance(resp, dict) else str(resp)
                                if len(result) > 2000:
                                    result = result[:2000] + "..."
                                content_parts.append(result)
                    content = "\n".join(content_parts)

            # Fallback: try direct content field on event
            if not content and "content" in event:
                content = event["content"]

            if not content:
                continue

            # Determine role for loss masking
            if event_type == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif event_type == "user":
                if has_tool_response:
                    messages.append({"role": "tool", "content": content})
                else:
                    messages.append({"role": "user", "content": content})

        # Handle old format: type is "message"
        elif event_type == "message":
            message = event.get("message", {})
            if not message:
                continue

            role = message.get("role", "")
            parts = message.get("parts", [])

            content_parts = []
            has_tool_response = False

            for part in parts:
                if isinstance(part, str):
                    content_parts.append(part)
                elif isinstance(part, dict):
                    if "text" in part:
                        content_parts.append(part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        content_parts.append(f"<tool_call>{fc.get('name', '')}</tool_call>")
                    elif "functionResponse" in part:
                        has_tool_response = True
                        fr = part["functionResponse"]
                        resp = fr.get("response", {})
                        result = resp.get("result", str(resp)) if isinstance(resp, dict) else str(resp)
                        if len(result) > 2000:
                            result = result[:2000] + "..."
                        content_parts.append(result)
                    elif "thought" in part:
                        content_parts.append(part["thought"])

            content = "\n".join(content_parts)
            if not content:
                continue

            if role == "model":
                messages.append({"role": "assistant", "content": content})
            elif role == "user":
                if has_tool_response:
                    messages.append({"role": "tool", "content": content})
                else:
                    messages.append({"role": "user", "content": content})

    return messages


def compute_loss_mask_for_response(
    args,
    prompt: str,
    messages: list[dict],
) -> tuple[list[int], list[int], int]:
    """
    Compute loss mask for the response portion.
    
    Uses MultiTurnLossMaskGenerator to properly mask:
    - Assistant messages: loss_mask = 1 (train on these)
    - Tool responses: loss_mask = 0 (don't train)
    - User messages: loss_mask = 0 (don't train)
    
    Returns:
        (token_ids, loss_mask, response_length)
    """
    tokenizer = get_tokenizer(args)
    mask_generator = get_mask_generator(args)
    
    # Build full message list including system/user prompt
    full_messages = [
        {"role": "user", "content": prompt},
    ] + messages
    
    # Generate loss mask
    token_ids, loss_mask = mask_generator.get_loss_mask(full_messages)
    
    # Response length is the portion with loss_mask = 1
    response_length = sum(loss_mask)
    
    return token_ids, loss_mask, response_length


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Generate SWE-bench solution using qwen-code CLI in Docker.
    
    This is the main rollout function called by SLiME.
    Path: examples.qwen_swe.rollout:generate
    
    Args:
        args: Training arguments
        sample: Sample with prompt and metadata
        sampling_params: Sampling parameters (not used - qwen-code handles this)
        
    Returns:
        Sample with tokens, response, loss_mask, and reward
    """
    instance_id = sample.metadata.get("instance_id", "unknown")
    container_name = None
    
    try:
        # Get vLLM URL from args or environment
        vllm_url = getattr(args, "vllm_url", None) or os.environ.get("VLLM_URL", "http://localhost:8000")
        
        # Setup agent config
        config = QwenAgentConfig(
            model_name=getattr(args, "qwen_model_name", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            api_base_url=vllm_url,
            max_turns=getattr(args, "qwen_max_turns", 50),
            timeout=getattr(args, "qwen_timeout", 1800),
        )
        
        # Setup container
        container_name = await asyncio.to_thread(
            setup_container,
            instance_id,
            suffix=f"_{sample.index}",
        )
        
        # Run agent
        result = await asyncio.to_thread(
            run_qwen_agent,
            container_name,
            instance_id,
            sample.prompt,
            config,
        )
        
        # Build messages from raw events for loss masking
        # Use raw_events (not converted messages) to preserve tool response info
        logger.info(f"[{instance_id}] Raw events count: {len(result.raw_events) if result.raw_events else 0}")
        if result.raw_events:
            # Debug: show first few event types
            event_types = [e.get("type", "unknown") for e in result.raw_events[:10]]
            logger.info(f"[{instance_id}] Event types (first 10): {event_types}")

        messages = build_messages_from_events(result.raw_events) if result.raw_events else []
        logger.info(f"[{instance_id}] Parsed messages: {len(messages)}")

        if messages:
            roles = [m["role"] for m in messages[:10]]
            logger.info(f"[{instance_id}] Message roles (first 10): {roles}")

        # Compute loss mask
        if messages:
            token_ids, loss_mask, response_length = compute_loss_mask_for_response(
                args, sample.prompt, messages
            )
            sample.tokens = token_ids
            sample.loss_mask = loss_mask[-response_length:] if response_length > 0 else []
            sample.response_length = response_length
        else:
            # Fallback: use stdout as response
            tokenizer = get_tokenizer(args)
            response_text = result.stdout if result.stdout else result.patch or ""
            prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)
            sample.tokens = prompt_ids + response_ids
            sample.response_length = len(response_ids)
            sample.loss_mask = [1] * len(response_ids)  # Train on all if no structure
        
        # Set response text
        if messages:
            response_parts = []
            for msg in messages:
                if msg["role"] == "assistant":
                    response_parts.append(msg["content"])
            sample.response = "\n".join(response_parts)
        else:
            sample.response = result.stdout if result.stdout else result.patch or ""
        
        # Evaluate patch for reward
        if result.patch:
            eval_result = await asyncio.to_thread(
                evaluate_with_swebench_harness,
                instance_id,
                result.patch,
                timeout=getattr(args, "eval_timeout", 900),
            )
            sample.reward = 1.0 if eval_result.resolved else -1.0
            sample.metadata["patch"] = result.patch
            sample.metadata["resolved"] = eval_result.resolved
            sample.metadata["patch_applied"] = eval_result.patch_applied
            logger.info(f"[{instance_id}] Resolved: {eval_result.resolved}")
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
    
    This enables parallel execution within a group.
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
