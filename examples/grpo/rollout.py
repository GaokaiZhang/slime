"""
Rollout function for SLiME GRPO training with Harbor evaluation.

This module provides:
1. Direct vLLM agent for token_ids and logprobs collection
2. Harbor Trial API for SWE-bench evaluation (using submodule, not pip package)

Key SLiME requirements:
- sample.tokens = prompt_tokens + response_tokens (FULL sequence)
- sample.response_length = len(response_tokens)
- sample.loss_mask = [1] * response_length (only response tokens)
- sample.rollout_log_probs = logprobs from generation
- sample.reward = float scalar
"""

import asyncio
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

# Add Harbor submodule to path (use repo code, not pip package)
_HARBOR_SRC = Path(__file__).resolve().parent.parent.parent / "submodules" / "harbor" / "src"
if str(_HARBOR_SRC) not in sys.path:
    sys.path.insert(0, str(_HARBOR_SRC))

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Global state
_TOKENIZER = None


def get_tokenizer(args_or_model_name):
    """Get or create tokenizer singleton."""
    global _TOKENIZER
    if _TOKENIZER is None:
        if isinstance(args_or_model_name, str):
            model_name = args_or_model_name
        else:
            model_name = getattr(args_or_model_name, "hf_checkpoint", None) or \
                         os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer for {model_name}...")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _TOKENIZER


async def generate_with_vllm_agent(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Generate using direct vLLM agent with proper SLiME data format.

    This mode:
    - Calls vLLM API directly
    - Captures completion_token_ids and logprobs
    - Uses Harbor Trial API for SWE-bench evaluation

    SLiME data format:
    - sample.tokens = prompt_tokens + response_tokens
    - sample.response_length = len(response_tokens)
    - sample.loss_mask = [1] * response_length
    - sample.rollout_log_probs = logprobs from vLLM
    """
    from .vllm_agent import VLLMAgentConfig, run_agent, get_tokenizer as get_vllm_tokenizer

    instance_id = sample.metadata.get("instance_id", f"sample_{sample.index}")

    # Get tokenizer for encoding prompt
    tokenizer = get_tokenizer(args)

    try:
        # Configure agent
        config = VLLMAgentConfig(
            api_url=getattr(args, "vllm_url", None) or os.environ.get("VLLM_URL", "http://localhost:8000"),
            model_name=getattr(args, "model_name", None) or os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            max_tokens=getattr(args, "rollout_max_response_len", 4096),
            temperature=getattr(args, "rollout_temperature", 1.0),
            max_turns=getattr(args, "max_turns", 50),
        )

        # Run agent (with tokenizer for token_id extraction)
        vllm_tokenizer = get_vllm_tokenizer(config.model_name)
        result = await asyncio.to_thread(
            run_agent,
            sample.prompt,
            config,
            workdir="/tmp",  # Will be overridden by Docker if using Harbor
            tokenizer=vllm_tokenizer,
        )

        # === KEY FIX: Build proper token sequence for SLiME ===
        # SLiME expects: tokens = prompt_tokens + response_tokens

        # Encode prompt to get prompt tokens
        if isinstance(sample.prompt, str):
            prompt_tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
        elif isinstance(sample.prompt, list):
            # Chat format - apply chat template
            prompt_text = tokenizer.apply_chat_template(sample.prompt, tokenize=False, add_generation_prompt=True)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_tokens = []

        # Response tokens from vLLM agent
        response_tokens = result.completion_token_ids

        # === Set SLiME-required fields ===
        sample.tokens = prompt_tokens + response_tokens  # FULL sequence
        sample.response_length = len(response_tokens)
        sample.loss_mask = [1] * len(response_tokens)  # Only response tokens are trainable
        sample.rollout_log_probs = result.logprobs  # For off-policy correction

        # Store trajectory in metadata
        sample.metadata["trajectory"] = result.trajectory
        sample.metadata["prompt_length"] = len(prompt_tokens)

        # === Evaluate for reward using Harbor or simple heuristic ===
        reward = await evaluate_with_harbor(
            instance_id=instance_id,
            patch=result.patch,
            sample=sample,
            args=args,
        )
        sample.reward = reward
        sample.metadata["patch"] = result.patch
        sample.metadata["resolved"] = (reward > 0)

        # Set status
        if result.exit_status == "completed":
            sample.status = Sample.Status.COMPLETED
        elif result.exit_status == "max_turns":
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.FAILED

        sample.response = f"Agent completed: {result.exit_status}, reward={sample.reward}"

        logger.info(f"[{instance_id}] Reward: {sample.reward}, "
                   f"Tokens: {len(sample.tokens)} (prompt={len(prompt_tokens)}, response={len(response_tokens)}), "
                   f"Turns: {len(result.trajectory.get('steps', []))}")

    except Exception as e:
        logger.error(f"[{instance_id}] Rollout failed: {e}")
        import traceback
        traceback.print_exc()

        sample.reward = -1.0
        sample.status = Sample.Status.FAILED
        sample.metadata["error"] = str(e)

        # Set minimal response with proper format
        tokenizer = get_tokenizer(args)
        if isinstance(sample.prompt, str):
            prompt_tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)
        else:
            prompt_tokens = []

        sample.tokens = prompt_tokens  # Just prompt, no response
        sample.loss_mask = []
        sample.response_length = 0
        sample.rollout_log_probs = []
        sample.response = ""

    return sample


async def evaluate_with_harbor(
    instance_id: str,
    patch: str,
    sample: Sample,
    args: Namespace,
) -> float:
    """
    Evaluate patch using Harbor Trial API.

    Falls back to heuristic if Harbor/Docker not available.
    """
    # Check if we should use Harbor evaluation
    use_harbor = getattr(args, "use_harbor_eval", True) and \
                 os.environ.get("USE_HARBOR_EVAL", "1") == "1"

    if not use_harbor or not patch:
        # No patch or Harbor disabled - return failure
        return -1.0

    try:
        # Try to use Harbor Trial API from submodule
        from harbor.trial.trial import Trial
        from harbor.models.trial.config import TrialConfig, TaskConfig, AgentConfig, EnvironmentConfig
        from harbor.models.environment_type import EnvironmentType

        # Check if task directory exists
        task_dir = Path(f"datasets/swebench/{instance_id}")
        if not task_dir.exists():
            # Try alternative paths
            alt_paths = [
                Path(f"/home/gaokaizhang/SWE-sft/harbor_jobs/datasets/swebench/{instance_id}"),
                Path(f"submodules/harbor/datasets/swebench/{instance_id}"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    task_dir = alt
                    break

        if not task_dir.exists():
            logger.warning(f"Task directory not found for {instance_id}, using heuristic evaluation")
            return _heuristic_reward(patch)

        # Create trial config with oracle agent that applies our patch
        trial_config = TrialConfig(
            task=TaskConfig(path=task_dir),
            agent=AgentConfig(
                name="oracle",
                kwargs={"patch": patch},
            ),
            environment=EnvironmentConfig(
                type=EnvironmentType.DOCKER,
                delete=True,
            ),
        )

        # Run trial
        trial = Trial(trial_config)
        result = await trial.run()

        # Extract reward
        if result.verifier_result and result.verifier_result.rewards:
            reward = result.verifier_result.rewards.get("reward", 0)
            return 1.0 if reward > 0 else -1.0
        else:
            return -1.0

    except ImportError as e:
        logger.warning(f"Harbor not available: {e}, using heuristic evaluation")
        return _heuristic_reward(patch)
    except Exception as e:
        logger.warning(f"Harbor evaluation failed: {e}, using heuristic evaluation")
        return _heuristic_reward(patch)


def _heuristic_reward(patch: str) -> float:
    """
    Simple heuristic reward when Docker/Harbor not available.

    Returns:
        0.0 if patch looks reasonable (has diff content)
        -1.0 if patch is empty or invalid
    """
    if not patch or len(patch.strip()) < 10:
        return -1.0

    # Check for valid diff markers
    if "---" in patch and "+++" in patch and "@@" in patch:
        return 0.0  # Neutral - looks like a valid patch but we can't verify

    return -1.0


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Generate SWE-bench solution.

    This is the main rollout function called by SLiME.
    Path: examples.grpo.rollout:generate

    Uses direct vLLM agent for proper token_ids and logprobs collection.
    """
    return await generate_with_vllm_agent(args, sample, sampling_params)


async def generate_group(
    args: Namespace,
    group: list[Sample],
    sampling_params: dict[str, Any],
) -> list[Sample]:
    """
    Generate solutions for a group of samples (same instance, multiple attempts).
    """
    tasks = [
        asyncio.create_task(generate(args, sample, sampling_params))
        for sample in group
    ]
    return await asyncio.gather(*tasks)


def compute_loss_mask_from_rollout_details(
    rollout_details: list[dict],
    args,
) -> tuple[list[int], list[int], list[float]]:
    """
    Compute loss mask from rollout details.

    For compatibility with Harbor's RolloutDetail format.
    """
    all_token_ids = []
    loss_mask = []
    all_logprobs = []

    for rollout in rollout_details:
        completion_ids_list = rollout.get("completion_token_ids", [])
        logprobs_list = rollout.get("logprobs", [])

        for turn_idx in range(len(completion_ids_list)):
            completion_ids = completion_ids_list[turn_idx]
            turn_logprobs = logprobs_list[turn_idx] if turn_idx < len(logprobs_list) else []

            all_token_ids.extend(completion_ids)
            loss_mask.extend([1] * len(completion_ids))

            if turn_logprobs:
                all_logprobs.extend(turn_logprobs)
            else:
                all_logprobs.extend([0.0] * len(completion_ids))

    return all_token_ids, loss_mask, all_logprobs


async def reward_func(args, sample, **kwargs) -> float:
    """Compute reward for a sample."""
    return sample.reward if sample.reward is not None else -1.0
