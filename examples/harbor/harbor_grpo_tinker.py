#!/usr/bin/env python
"""
Harbor GRPO Trainer - Tinker GPU Version

This trainer uses Tinker (Thinking Machines Lab) for GPU inference and training:
- Harbor CLI for agent rollouts (locally with Docker)
- Tinker for LLM inference (OpenAI-compatible API)
- Tinker for GRPO training (via weighted SFT approximation)

Uses shared code from harbor_core.py.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Tinker Cloud (GPU)                        │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Inference (OpenAI-compatible API)                     │   │
    │  │ - Chat/Completions endpoints                          │   │
    │  │ - Base URL: https://tinker...thinkingmachines.dev     │   │
    │  ├──────────────────────────────────────────────────────┤   │
    │  │ Training (Native SDK)                                 │   │
    │  │ - LoRA fine-tuning                                    │   │
    │  │ - forward_backward + optim_step                       │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                              ▲
                              │ responses + rewards
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                    Local Machine                             │
    │  ┌──────────────────┐    ┌──────────────────────────────┐   │
    │  │ Harbor Agent     │    │ Reward Computation            │   │
    │  │ (qwen-coder)     │    │ (test execution)              │   │
    │  └──────────────────┘    └──────────────────────────────┘   │
    │           │                           │                     │
    │           ▼                           ▼                     │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Docker Containers                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Tinker API Reference:
    - Docs: https://tinker-docs.thinkingmachines.ai/
    - OpenAI endpoint: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
    - Supported models: Qwen3-8B, Qwen3-30B-A3B, Llama-3.1-8B-Instruct, etc.

Usage:
    # Set Tinker API key
    export TINKER_API_KEY="tml-..."

    # Test mode (5 instances)
    python examples/harbor/harbor_grpo_tinker.py --test

    # Full training
    python examples/harbor/harbor_grpo_tinker.py --num-rollouts 50

    # With specific Tinker model
    python examples/harbor/harbor_grpo_tinker.py --tinker-model "Qwen/Qwen3-8B"
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Tinker Configuration
# ==============================================================================

# Tinker OpenAI-compatible API endpoint
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Default Tinker model (must be in Tinker's supported model list)
# Available: Qwen3-8B, Qwen3-30B-A3B, Llama-3.1-8B-Instruct, etc.
DEFAULT_TINKER_MODEL = "Qwen/Qwen3-8B"


@dataclass
class TinkerGRPOConfig:
    """Configuration for Tinker GRPO training."""

    # Tinker model (from Tinker's supported list)
    tinker_model: str = DEFAULT_TINKER_MODEL

    # Harbor agent and environment
    agent: str = "qwen-coder"
    env: str = "docker"
    dataset: str = "swebench-verified@1.0"

    # GRPO training (Search-R1 parameters)
    n_samples_per_prompt: int = 4
    lr: float = 1e-6
    kl_coef: float = 0.001

    # LoRA configuration for Tinker
    lora_rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True

    # Evaluation
    eval_timeout: int = 300

    # Output
    output_dir: str = "outputs/harbor_grpo_tinker"
    jobs_dir: str = "jobs"
    save_every: int = 10


# ==============================================================================
# Tinker Client Wrapper
# ==============================================================================

class TinkerClient:
    """
    Client for Tinker API (Thinking Machines Lab).

    Provides both OpenAI-compatible inference and native training capabilities.
    Supports true GRPO/PPO training via Tinker's native PPO loss function.
    """

    def __init__(self, api_key: str = None, base_model: str = DEFAULT_TINKER_MODEL):
        """
        Initialize Tinker client.

        Args:
            api_key: Tinker API key (or set TINKER_API_KEY env var)
            base_model: Base model for training/inference
        """
        self.api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tinker API key required. Set TINKER_API_KEY env var or pass api_key parameter.\n"
                "Get your key at: https://tinker-console.thinkingmachines.ai"
            )

        self.base_model = base_model
        self.service_client = None
        self.training_client = None
        self.sampling_client = None
        self.tokenizer = None
        self._checkpoint_path = None

        # Try to import tinker SDK
        try:
            import tinker
            self._tinker_sdk_available = True
            logger.info("Tinker SDK available")
        except ImportError:
            self._tinker_sdk_available = False
            logger.warning("Tinker SDK not installed. Install with: pip install tinker")
            logger.info("Will use OpenAI-compatible API for inference only")

    def get_openai_client(self):
        """Get OpenAI client configured for Tinker's OpenAI-compatible API."""
        try:
            from openai import OpenAI
            return OpenAI(
                base_url=TINKER_BASE_URL,
                api_key=self.api_key,
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def init_training(self, lora_rank: int = 32, train_mlp: bool = True, train_attn: bool = True):
        """
        Initialize Tinker training client with LoRA.

        Args:
            lora_rank: LoRA rank
            train_mlp: Whether to train MLP layers
            train_attn: Whether to train attention layers
        """
        if not self._tinker_sdk_available:
            raise RuntimeError("Tinker SDK required for training. Install with: pip install tinker")

        import tinker

        logger.info(f"Initializing Tinker training client with model: {self.base_model}")

        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.base_model,
            rank=lora_rank,
            train_mlp=train_mlp,
            train_attn=train_attn,
        )

        # Get tokenizer for encoding/decoding
        self.tokenizer = self.training_client.get_tokenizer()

        logger.info(f"Tinker training client initialized (LoRA rank={lora_rank})")
        return self.training_client

    def init_sampling(self):
        """Initialize Tinker sampling client."""
        if not self._tinker_sdk_available:
            logger.info("Tinker SDK not available, using OpenAI-compatible API for sampling")
            return None

        import tinker

        if self.training_client is not None:
            # Get sampling client from training client (shares weights)
            logger.info("Creating sampling client from training client")
            self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
                name=f"grpo-checkpoint-{int(time.time())}"
            )
        else:
            # Create standalone sampling client
            logger.info(f"Creating standalone sampling client for {self.base_model}")
            if self.service_client is None:
                import tinker
                self.service_client = tinker.ServiceClient()
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=self.base_model
            )

        # Get tokenizer if not already set
        if self.tokenizer is None:
            self.tokenizer = self.sampling_client.get_tokenizer()

        return self.sampling_client

    async def sample_with_logprobs_async(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate text using Tinker's sampling API, returning both text and logprobs.

        This is required for PPO/GRPO training - we need the logprobs from sampling.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with keys: 'text', 'tokens', 'logprobs'
        """
        if not self._tinker_sdk_available:
            raise RuntimeError("Tinker SDK required for sampling with logprobs")

        import tinker
        import tinker.types as types

        # Ensure sampling client exists
        if self.sampling_client is None:
            self.init_sampling()

        # Encode prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt)
        encoded_chunk = tinker.EncodedTextChunk(tokens=prompt_tokens)
        model_input = types.ModelInput(chunks=[encoded_chunk])

        # Sample with Tinker
        result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

        # Extract results
        sequence = result.sequences[0]
        output_tokens = sequence.tokens
        output_logprobs = sequence.logprobs
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

        return {
            'text': output_text,
            'tokens': output_tokens,
            'logprobs': output_logprobs,
            'prompt_tokens': prompt_tokens,
        }

    async def sample_async(self, prompt: str, max_tokens: int = 2048, temperature: float = 1.0) -> str:
        """
        Generate text using Tinker's sampling API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        result = await self.sample_with_logprobs_async(prompt, max_tokens, temperature)
        return result['text']

    async def train_grpo_step_async(
        self,
        rollouts: List[Dict[str, Any]],
        learning_rate: float = 1e-6,
        eps_clip_low: float = 0.8,      # SLiME: 1 - eps_clip (0.2) = 0.8
        eps_clip_high: float = 1.28,    # SLiME: 1 + eps_clip_high (0.28) = 1.28
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step using Tinker's native PPO loss.

        Tinker's PPO loss requires:
        - target_tokens: Token IDs from sampling
        - logprobs: Logprobs recorded during sampling (reference policy)
        - advantages: GRPO advantages computed from rewards

        Args:
            rollouts: List of rollout dicts with keys:
                - 'prompt_tokens': List[int] - Prompt token IDs
                - 'tokens': List[int] - Response token IDs
                - 'logprobs': List[float] - Logprobs from sampling
                - 'reward': float - Reward value
            learning_rate: Learning rate for this step
            eps_clip_low: PPO low clipping threshold (default 0.8)
            eps_clip_high: PPO high clipping threshold (default 1.2)

        Returns:
            Training metrics dict
        """
        if not self._tinker_sdk_available or self.training_client is None:
            raise RuntimeError("Training client not initialized. Call init_training() first.")

        import tinker
        import tinker.types as types
        import numpy as np

        if not rollouts:
            return {"loss": 0.0, "n_samples": 0, "mean_reward": 0.0}

        # Compute group-relative advantages (GRPO)
        rewards = [r['reward'] for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)
        if len(rewards) > 1:
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_reward = max(variance ** 0.5, 1e-8)
            advantages = [(r - mean_reward) / std_reward for r in rewards]
        else:
            std_reward = 0.0
            advantages = [0.0 for _ in rewards]

        logger.info(f"  GRPO advantages: mean_reward={mean_reward:.3f}, std={std_reward:.3f}")

        # Create Datum objects for PPO training
        # Note: For PPO, model_input should contain ONLY the response tokens,
        # and all loss_fn_inputs arrays must match this length.
        data = []
        for rollout, advantage in zip(rollouts, advantages):
            response_tokens = list(rollout['tokens'])
            response_logprobs = list(rollout['logprobs'])

            # Create model input with response tokens only (not full sequence)
            encoded_chunk = tinker.EncodedTextChunk(tokens=response_tokens)
            model_input = types.ModelInput(chunks=[encoded_chunk])

            # Create TensorData for loss_fn_inputs
            # PPO requires: target_tokens, logprobs, advantages (all same length as model_input)
            response_tokens_arr = np.array(response_tokens, dtype=np.int64)
            response_logprobs_arr = np.array(response_logprobs, dtype=np.float32)
            advantages_arr = np.full(len(response_tokens), advantage, dtype=np.float32)

            datum = tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_numpy(response_tokens_arr),
                    "logprobs": tinker.TensorData.from_numpy(response_logprobs_arr),
                    "advantages": tinker.TensorData.from_numpy(advantages_arr),
                }
            )
            data.append(datum)

        # Forward-backward with PPO loss
        fwd_bwd_future = self.training_client.forward_backward(
            data=data,
            loss_fn="ppo",
            loss_fn_config={
                "clip_low_threshold": eps_clip_low,
                "clip_high_threshold": eps_clip_high,
            }
        )
        fwd_bwd_result = fwd_bwd_future.result()

        # Optimizer step
        optim_future = self.training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        )
        optim_result = optim_future.result()

        # Extract loss from result
        loss = 0.0
        if hasattr(fwd_bwd_result, 'loss'):
            loss = float(fwd_bwd_result.loss)
        elif hasattr(fwd_bwd_result, 'items') and fwd_bwd_result.items:
            # Sum up losses from items
            for item in fwd_bwd_result.items:
                if hasattr(item, 'loss'):
                    loss += float(item.loss)
            loss = loss / len(fwd_bwd_result.items) if fwd_bwd_result.items else 0.0

        return {
            "loss": loss,
            "n_samples": len(rollouts),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }

    async def train_step_async(
        self,
        examples: List[Dict[str, Any]],
        learning_rate: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Perform one SFT training step using Tinker's cross_entropy loss.

        This is a fallback for when we don't have logprobs from sampling.

        Args:
            examples: List of training examples with 'prompt' and 'completion' keys
            learning_rate: Learning rate for this step

        Returns:
            Training metrics dict
        """
        if not self._tinker_sdk_available or self.training_client is None:
            raise RuntimeError("Training client not initialized. Call init_training() first.")

        import tinker
        import tinker.types as types
        import numpy as np

        if not examples:
            return {"loss": 0.0, "n_samples": 0}

        # Create Datum objects for SFT training
        data = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            completion = ex.get("completion", "")

            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt)
            completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
            full_tokens = prompt_tokens + completion_tokens

            # Create model input
            encoded_chunk = tinker.EncodedTextChunk(tokens=full_tokens)
            model_input = types.ModelInput(chunks=[encoded_chunk])

            # For cross_entropy, we need target_tokens
            target_tokens_arr = np.array(completion_tokens, dtype=np.int64)

            datum = tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_numpy(target_tokens_arr),
                }
            )
            data.append(datum)

        # Forward-backward with cross_entropy loss
        fwd_bwd_future = self.training_client.forward_backward(
            data=data,
            loss_fn="cross_entropy",
        )
        fwd_bwd_result = fwd_bwd_future.result()

        # Optimizer step
        optim_future = self.training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        )
        optim_result = optim_future.result()

        # Extract loss
        loss = 0.0
        if hasattr(fwd_bwd_result, 'loss'):
            loss = float(fwd_bwd_result.loss)

        return {
            "loss": loss,
            "n_samples": len(examples),
        }

    def save_checkpoint(self, name: str = None) -> str:
        """
        Save training checkpoint to Tinker.

        Args:
            name: Checkpoint name

        Returns:
            Checkpoint path (Tinker URI)
        """
        if not self._tinker_sdk_available or self.training_client is None:
            raise RuntimeError("Training client not initialized")

        name = name or f"grpo-checkpoint-{int(time.time())}"

        # Save weights and get sampling client (which also saves the checkpoint)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=name
        )

        # The checkpoint path would be a tinker:// URI
        self._checkpoint_path = f"tinker://{name}"
        logger.info(f"Saved checkpoint: {self._checkpoint_path}")

        return self._checkpoint_path


# ==============================================================================
# Harbor Rollout Functions (Tinker Inference)
# ==============================================================================

def run_harbor_agent_with_tinker(
    instance: dict,
    config: TinkerGRPOConfig,
    tinker_api_key: str,
    timeout: int = 1800,
    openai_base_url: str = None,
    openai_api_key: str = None,
) -> dict:
    """
    Run Harbor agent on a single instance.

    For agents that need LLM inference (qwen-coder, etc.), provide openai_base_url
    and openai_api_key to specify the inference endpoint.

    Args:
        instance: Instance dict with instance_id, task_dir (for c2bug) or problem_statement
        config: Training configuration
        tinker_api_key: Tinker API key (for training)
        timeout: Timeout in seconds
        openai_base_url: OpenAI-compatible API URL for agent inference
        openai_api_key: API key for agent inference

    Returns:
        {"response": str, "reward": float, "status": str, "job_dir": str}
    """
    instance_id = instance["instance_id"]
    job_name = f"tinker-{instance_id.replace('/', '_').replace('__', '_')[:50]}-{int(time.time())}"

    # Build Harbor command
    cmd = [
        "harbor", "run",
        "--env", config.env,
        "--agent", config.agent,
        "--n-concurrent", "1",
        "--jobs-dir", config.jobs_dir,
        "--job-name", job_name,
        "--export-traces",
    ]

    # Handle different data sources
    if "task_dir" in instance:
        # C2Bug: use task_dir
        cmd.extend(["-p", instance["task_dir"]])
    else:
        # SWE-bench: use dataset and task-name
        cmd.extend([
            "--dataset", config.dataset,
            "--task-name", instance_id,
        ])

    # Configure agent inference endpoint (if not oracle agent)
    if config.agent != "oracle" and config.agent != "nop":
        # Use provided OpenAI-compatible API endpoint for inference
        if openai_base_url:
            cmd.extend([
                "--ak", f"base_url={openai_base_url}",
                "--ak", f"api_key={openai_api_key or 'dummy'}",
                "--ak", f"model={config.tinker_model}",
            ])
        else:
            logger.warning(
                f"Agent {config.agent} requires an OpenAI-compatible API. "
                "Use --proxy-url to specify a Tinker proxy server, or use --agent oracle."
            )

    logger.info(f"  Running Harbor with Tinker: {config.agent} on {instance_id}")
    logger.debug(f"  Command: {' '.join(cmd)}")

    try:
        # Set environment for subprocess
        env = os.environ.copy()
        env["TINKER_API_KEY"] = tinker_api_key

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        job_dir = Path(config.jobs_dir) / job_name

        if result.returncode != 0:
            logger.warning(f"  Harbor failed: {result.stderr[:500] if result.stderr else 'No error output'}")

        # Parse reward from job directory (Harbor's built-in evaluation)
        reward = parse_harbor_reward(job_dir)

        # Parse response from trajectory
        response = parse_harbor_response(job_dir)

        return {
            "response": response,
            "reward": reward,
            "status": "completed" if reward > 0 else "failed",
            "job_dir": str(job_dir),
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"  Harbor timeout for {instance_id}")
        return {"response": "", "reward": -1.0, "status": "timeout", "job_dir": ""}
    except Exception as e:
        logger.error(f"  Harbor error: {e}")
        return {"response": "", "reward": -1.0, "status": "error", "job_dir": ""}


def parse_harbor_reward(job_dir: Path) -> float:
    """Parse reward from Harbor job directory (reward.txt)."""
    if not job_dir.exists():
        return -1.0

    for reward_file in job_dir.glob("**/reward.txt"):
        try:
            reward_text = reward_file.read_text().strip()
            return 1.0 if reward_text == "1" else -1.0
        except Exception:
            pass
    return -1.0


def parse_harbor_response(job_dir: Path) -> str:
    """Parse response from Harbor job directory."""
    if not job_dir.exists():
        return ""

    response = ""
    for traj_file in job_dir.glob("**/trajectory*.json"):
        try:
            with open(traj_file) as f:
                data = json.load(f)

            messages = data.get("messages", [])
            if messages:
                for msg in messages:
                    if msg.get("role") == "assistant":
                        response += msg.get("content", "") + "\n"
            elif "trajectory" in data:
                for step in data["trajectory"]:
                    if "action" in step:
                        response += step["action"] + "\n"
            break
        except Exception as e:
            logger.debug(f"Error parsing trajectory: {e}")

    return response.strip()


# ==============================================================================
# Data Loading
# ==============================================================================

def load_training_instances(
    num_instances: int = 50,
    test_mode: bool = False,
    data_source: str = "swebench",
    c2bug_dataset: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    c2bug_docker_image: str = "swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
) -> list:
    """Load training instances from SWE-bench or C2Bug."""
    if test_mode:
        num_instances = min(5, num_instances)

    if data_source == "c2bug":
        from examples.harbor.c2bug_adapter import (
            load_c2bug_from_hf,
            C2BugToHarbor,
            C2BugLoader,
        )

        logger.info(f"Loading c2bug data from {c2bug_dataset}...")
        collection = load_c2bug_from_hf(c2bug_dataset)

        run_meta_override = {"docker_image": c2bug_docker_image, "workdir": "/testbed"}
        task_root = Path("/tmp/c2bug_harbor_tinker")

        converter = C2BugToHarbor(
            collection_source=collection,
            task_root=task_root,
            max_timeout_sec=3000.0,
            run_meta_override=run_meta_override,
        )
        converter.generate_many(limit=num_instances, overwrite=True)

        instances = []
        loader = C2BugLoader(collection)
        loader.apply_run_meta(run_meta_override)

        for record in list(loader.iter_records())[:num_instances]:
            task_name = record.task_uid or record.instance_id
            task_dir = task_root / task_name
            if task_dir.exists():
                instances.append({
                    "instance_id": task_name,
                    "task_dir": str(task_dir),
                    "problem_statement": record.issue_text,
                    "repo": record.repo,
                })

        logger.info(f"Loaded {len(instances)} c2bug instances")
        return instances

    else:
        # SWE-bench
        from datasets import load_dataset

        train_file = Path(__file__).parent / "train_instances_id.txt"
        if not train_file.exists():
            train_file = Path("/home/gaokaizhang/slime/examples/harbor/train_instances_id.txt")

        if train_file.exists():
            with open(train_file) as f:
                instance_ids = [line.strip() for line in f if line.strip()]
            instance_ids = instance_ids[:num_instances]

            dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            id_to_instance = {item["instance_id"]: item for item in dataset}

            instances = []
            for iid in instance_ids:
                if iid in id_to_instance:
                    item = id_to_instance[iid]
                    instances.append({
                        "instance_id": iid,
                        "problem_statement": item["problem_statement"],
                        "repo": item["repo"],
                    })

            logger.info(f"Loaded {len(instances)} SWE-bench instances")
            return instances

        # Fallback: load from HuggingFace
        logger.info("Loading from HuggingFace...")
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        django_instances = [
            {
                "instance_id": item["instance_id"],
                "problem_statement": item["problem_statement"],
                "repo": item["repo"],
            }
            for item in dataset
            if "django" in item["instance_id"].lower()
        ]

        logger.info(f"Found {len(django_instances)} Django instances")
        return django_instances[:num_instances]


# ==============================================================================
# GRPO Training (True PPO with Tinker)
# ==============================================================================

def create_swebench_prompt(instance: dict) -> str:
    """Create a SWE-bench style prompt for code generation."""
    problem = instance.get("problem_statement", "")
    repo = instance.get("repo", "django/django")

    prompt = f"""You are an expert software engineer. Fix the following issue in the {repo} repository.

## Issue Description
{problem[:4000]}

## Instructions
1. Analyze the issue carefully
2. Identify the root cause
3. Provide a fix in the form of a unified diff patch

Output your fix as a unified diff patch that can be applied with `git apply`:

```diff"""
    return prompt


def parse_patch_from_response(response: str) -> str:
    """Extract a diff patch from the model's response."""
    import re

    # Try to find diff block
    diff_match = re.search(r'```diff\s*(.*?)\s*```', response, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    # Try to find any diff-like content
    lines = response.split('\n')
    diff_lines = []
    in_diff = False

    for line in lines:
        if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
            if line.startswith('@@') or line.startswith('+') or line.startswith('-') or line.startswith(' '):
                continue
            elif line.strip() == '' and diff_lines:
                continue

    if diff_lines:
        return '\n'.join(diff_lines)

    return response  # Return raw response as fallback


async def evaluate_patch_with_harbor(
    instance: dict,
    patch: str,
    config: TinkerGRPOConfig,
    timeout: int = 300,
) -> float:
    """
    Evaluate a patch using Harbor's Docker environment.

    Args:
        instance: Instance dict with instance_id, etc.
        patch: The diff patch to evaluate
        config: Configuration
        timeout: Timeout in seconds

    Returns:
        Reward: 1.0 if tests pass, -1.0 otherwise
    """
    import tempfile

    instance_id = instance["instance_id"]
    job_name = f"eval-{instance_id.replace('/', '_').replace('__', '_')[:40]}-{int(time.time())}"

    # Write patch to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        # Use oracle agent with the patch
        cmd = [
            "harbor", "run",
            "--env", config.env,
            "--agent", "oracle",  # Oracle applies the gold patch, but we can override
            "--n-concurrent", "1",
            "--jobs-dir", config.jobs_dir,
            "--job-name", job_name,
            "--dataset", config.dataset,
            "--task-name", instance_id,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        job_dir = Path(config.jobs_dir) / job_name
        reward = parse_harbor_reward(job_dir)
        return reward

    except subprocess.TimeoutExpired:
        return -1.0
    except Exception as e:
        logger.warning(f"Evaluation error: {e}")
        return -1.0
    finally:
        os.unlink(patch_file)


async def run_grpo_with_tinker(
    config: TinkerGRPOConfig,
    num_rollouts: int = 50,
    test_mode: bool = False,
    data_source: str = "swebench",
    c2bug_dataset: str = "TwelfthStar/c2bug_tasks_django_Jan-22-2026",
    c2bug_docker_image: str = "swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
    skip_training: bool = False,
    proxy_url: str = "http://localhost:8000/v1",
):
    """
    Run GRPO training using Tinker for GPU via proxy server.

    Architecture:
        Harbor Agent (qwen-coder, etc.)
            │ OpenAI API calls
            ▼
        Tinker Proxy (localhost:8000)
            │ Tinker SDK
            ▼
        Tinker Cloud (GPU)
            ├── sample_async() → Generate + store logprobs
            └── forward_backward(loss_fn="ppo") → Train

    Requirements:
        1. Start the proxy server first:
           python examples/harbor/tinker_proxy.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507"

        2. Then run this script with --proxy-url
    """
    # Get Tinker API key (not required when using proxy)
    tinker_api_key = os.environ.get("TINKER_API_KEY", "proxy-mode")
    if tinker_api_key == "proxy-mode":
        logger.info("No TINKER_API_KEY set - using proxy mode (proxy handles authentication)")

    logger.info("=" * 70)
    logger.info("Harbor GRPO Training - Tinker GPU (via Proxy)")
    logger.info("=" * 70)
    logger.info(f"  Tinker model: {config.tinker_model}")
    logger.info(f"  Proxy URL: {proxy_url}")
    logger.info(f"  Data source: {data_source}")
    logger.info(f"  Agent: {config.agent}")
    logger.info(f"  Environment: {config.env}")
    logger.info(f"  n_samples_per_prompt: {config.n_samples_per_prompt}")
    logger.info("=" * 70)

    # Verify proxy is running
    try:
        import requests
        health = requests.get(f"{proxy_url.rstrip('/v1')}/health", timeout=5)
        if health.status_code == 200:
            logger.info(f"Proxy server healthy: {health.json()}")
        else:
            logger.warning(f"Proxy health check failed: {health.status_code}")
    except Exception as e:
        logger.warning(f"Could not connect to proxy at {proxy_url}: {e}")
        logger.info("Make sure to start the proxy first:")
        logger.info(f"  python examples/harbor/tinker_proxy.py --model {config.tinker_model}")

    # Load training instances
    train_instances = load_training_instances(
        num_instances=num_rollouts,
        test_mode=test_mode,
        data_source=data_source,
        c2bug_dataset=c2bug_dataset,
        c2bug_docker_image=c2bug_docker_image,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.jobs_dir, exist_ok=True)

    # Training loop
    all_metrics = []
    total_resolved = 0
    total_samples = 0
    completion_ids = []  # Track completion IDs from proxy for PPO training
    completion_rewards = []  # Track rewards for each completion
    start_time = time.time()

    for idx, instance in enumerate(train_instances):
        instance_id = instance["instance_id"]
        logger.info(f"\n[{idx + 1}/{len(train_instances)}] {instance_id}")

        # Collect rollouts for this instance
        instance_rewards = []

        for sample_idx in range(config.n_samples_per_prompt):
            logger.info(f"  Sample {sample_idx + 1}/{config.n_samples_per_prompt}")

            # Get current completion IDs before Harbor run (for tracking new ones)
            try:
                import requests
                pre_run_ids = set(requests.get(
                    f"{proxy_url.rstrip('/v1')}/v1/logprobs", timeout=5
                ).json().get("ids", []))
            except Exception:
                pre_run_ids = set()

            # Run Harbor agent (uses proxy for LLM inference)
            result = run_harbor_agent_with_tinker(
                instance=instance,
                config=config,
                tinker_api_key=tinker_api_key,
                timeout=1800,
                openai_base_url=proxy_url,
            )

            reward = result["reward"]
            instance_rewards.append(reward)

            # Get new completion IDs after Harbor run (timestamp-based tracking)
            try:
                post_run_ids = set(requests.get(
                    f"{proxy_url.rstrip('/v1')}/v1/logprobs", timeout=5
                ).json().get("ids", []))
                new_ids = post_run_ids - pre_run_ids
                if new_ids:
                    logger.info(f"    Captured {len(new_ids)} completions from this run")
                    for cid in new_ids:
                        completion_ids.append(cid)
                        completion_rewards.append(reward)
            except Exception as e:
                logger.warning(f"    Failed to capture completion IDs: {e}")

            total_samples += 1
            if reward > 0:
                total_resolved += 1

            logger.info(f"    Status: {result['status']}, Reward: {reward}")

        # Compute GRPO statistics for this instance
        mean_reward = sum(instance_rewards) / len(instance_rewards)
        logger.info(f"  Mean reward: {mean_reward:.3f} (positive: {sum(1 for r in instance_rewards if r > 0)}/{len(instance_rewards)})")

        # Train via proxy when we have enough samples
        if not skip_training and len(completion_ids) >= config.n_samples_per_prompt:
            logger.info(f"  Training via proxy with {len(completion_ids)} samples...")

            try:
                import requests
                train_response = requests.post(
                    f"{proxy_url.rstrip('/v1')}/v1/train/ppo",
                    json={
                        "completion_ids": completion_ids,
                        "rewards": completion_rewards,
                        "learning_rate": config.lr,
                    },
                    timeout=120,
                )

                if train_response.status_code == 200:
                    train_result = train_response.json()
                    logger.info(f"  PPO training: mean_reward={train_result.get('mean_reward', 'N/A'):.3f}")
                    train_result["instance_id"] = instance_id
                    train_result["idx"] = idx
                    all_metrics.append(train_result)
                else:
                    logger.warning(f"  Training failed: {train_response.text}")
                    all_metrics.append({
                        "instance_id": instance_id,
                        "idx": idx,
                        "error": train_response.text,
                    })

            except Exception as e:
                logger.error(f"  Training error: {e}")
                all_metrics.append({
                    "instance_id": instance_id,
                    "idx": idx,
                    "error": str(e),
                })

            # Clear buffers after training
            completion_ids = []
            completion_rewards = []
        else:
            # Not enough samples yet, or skip_training is True
            all_metrics.append({
                "instance_id": instance_id,
                "idx": idx,
                "rewards": instance_rewards,
                "mean_reward": mean_reward,
                "skipped": skip_training,
            })

        # Save metrics
        with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    resolve_rate = total_resolved / total_samples if total_samples > 0 else 0

    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total instances: {len(train_instances)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total resolved: {total_resolved} ({resolve_rate * 100:.1f}%)")
    logger.info(f"Time: {elapsed / 60:.1f} minutes")
    logger.info(f"Note: Checkpoints are managed by the proxy server")

    # Save summary
    summary = {
        "total_instances": len(train_instances),
        "total_samples": total_samples,
        "total_resolved": total_resolved,
        "resolve_rate": resolve_rate,
        "elapsed_minutes": elapsed / 60,
        "config": {
            "tinker_model": config.tinker_model,
            "agent": config.agent,
            "lr": config.lr,
            "n_samples_per_prompt": config.n_samples_per_prompt,
        },
    }
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harbor GRPO Trainer with Tinker GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test mode (5 instances)
    python harbor_grpo_tinker.py --test

    # Full training
    python harbor_grpo_tinker.py --num-rollouts 50

    # With specific Tinker model
    python harbor_grpo_tinker.py --tinker-model "Qwen/Qwen3-8B"

    # Rollouts only (no training)
    python harbor_grpo_tinker.py --skip-training --test

Tinker Setup:
    1. Get API key: https://tinker-console.thinkingmachines.ai
    2. Set environment: export TINKER_API_KEY="tml-..."
    3. Install SDK (optional): pip install tinker

Supported Tinker Models:
    - Qwen/Qwen3-8B (default)
    - Qwen/Qwen3-30B-A3B
    - meta-llama/Llama-3.1-8B-Instruct
    - meta-llama/Llama-3.1-70B
    """,
    )

    # Tinker arguments
    parser.add_argument("--tinker-model", default=DEFAULT_TINKER_MODEL,
                       help=f"Tinker model (default: {DEFAULT_TINKER_MODEL})")
    parser.add_argument("--lora-rank", type=int, default=32,
                       help="LoRA rank for training (default: 32)")

    # Harbor arguments
    parser.add_argument("--agent", default="qwen-coder",
                       help="Harbor agent (default: qwen-coder)")
    parser.add_argument("--env", default="docker",
                       help="Environment: docker or daytona (default: docker)")
    parser.add_argument("--dataset", default="swebench-verified@1.0",
                       help="Harbor dataset (default: swebench-verified@1.0)")

    # Training arguments
    parser.add_argument("--num-rollouts", type=int, default=50,
                       help="Number of instances to train on (default: 50)")
    parser.add_argument("--n-samples", type=int, default=4,
                       help="GRPO group size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate (default: 1e-6)")

    # Data source arguments
    parser.add_argument("--data-source", default="swebench", choices=["swebench", "c2bug"],
                       help="Data source (default: swebench)")
    parser.add_argument("--c2bug-dataset", default="TwelfthStar/c2bug_tasks_django_Jan-22-2026",
                       help="HuggingFace dataset for c2bug")
    parser.add_argument("--c2bug-docker-image", default="swebench/sweb.eval.x86_64.django_1776_django-13810:latest",
                       help="Docker image for c2bug tasks")

    # Output arguments
    parser.add_argument("--output-dir", default="outputs/harbor_grpo_tinker",
                       help="Output directory (default: outputs/harbor_grpo_tinker)")
    parser.add_argument("--jobs-dir", default="jobs",
                       help="Harbor jobs directory (default: jobs)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N instances (default: 10)")

    # Flags
    parser.add_argument("--test", action="store_true",
                       help="Test mode (5 instances)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Only run rollouts, skip training")
    parser.add_argument("--proxy-url", default="http://172.17.0.1:8000/v1",
                       help="URL of Tinker proxy server (default: http://172.17.0.1:8000/v1 for Docker)")

    args = parser.parse_args()

    # Update output dir for c2bug
    output_dir = args.output_dir
    if output_dir == "outputs/harbor_grpo_tinker" and args.data_source == "c2bug":
        output_dir = "outputs/harbor_grpo_tinker_c2bug"

    config = TinkerGRPOConfig(
        tinker_model=args.tinker_model,
        agent=args.agent,
        env=args.env,
        dataset=args.dataset,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        lora_rank=args.lora_rank,
        output_dir=output_dir,
        jobs_dir=args.jobs_dir,
        save_every=args.save_every,
    )

    # Run async training loop
    asyncio.run(run_grpo_with_tinker(
        config=config,
        num_rollouts=args.num_rollouts,
        test_mode=args.test,
        data_source=args.data_source,
        c2bug_dataset=args.c2bug_dataset,
        c2bug_docker_image=args.c2bug_docker_image,
        skip_training=args.skip_training,
        proxy_url=args.proxy_url,
    ))


if __name__ == "__main__":
    main()
