#!/usr/bin/env python
"""
GRPO-compatible agent using mini-swe-agent-plus.

This module provides:
1. GRPOModel: A vLLM-based model that captures token_ids and logprobs for GRPO
2. DockerEnvironment: Environment wrapper for executing in swebench Docker containers
3. run_agent_for_grpo(): Run mini-swe-agent-plus and return GRPO rollout data

The agent runs multi-turn interactions with tools, and we capture all model
outputs (token_ids, logprobs) for GRPO training.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# Add mini-swe-agent-plus to path
MINI_SWE_AGENT_PATH = Path(__file__).parent.parent.parent / "submodules" / "mini-swe-agent-plus" / "src"
sys.path.insert(0, str(MINI_SWE_AGENT_PATH))

logger = logging.getLogger(__name__)


class DockerEnvironment:
    """
    Environment that executes commands inside a Docker container.

    Compatible with mini-swe-agent-plus's Environment interface.
    """

    def __init__(self, container_name: str, working_dir: str = "/testbed", timeout: int = 120):
        self.container_name = container_name
        self.working_dir = working_dir
        self.timeout = timeout

    def execute(self, command: str) -> dict:
        """Execute a bash command in the Docker container."""
        cmd = [
            "docker", "exec",
            "-w", self.working_dir,
            self.container_name,
            "bash", "-c", command
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            return {"output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired as e:
            output = e.stdout.decode() if e.stdout else ""
            raise TimeoutError(output)
        except Exception as e:
            return {"output": str(e), "returncode": -1}

    def get_template_vars(self) -> dict:
        """Return template variables for the environment."""
        return {
            "working_dir": self.working_dir,
            "container_name": self.container_name,
            "cwd": self.working_dir,  # Required by swebench config templates
        }


@dataclass
class GRPOModelConfig:
    """Configuration for GRPO-compatible model."""
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    vllm_url: str = "http://localhost:8000"
    temperature: float = 1.0
    max_tokens: int = 2048
    echo_prompt: bool = False  # Whether to echo prompt in logprobs


@dataclass
class TurnData:
    """Data captured from a single model turn."""
    prompt_tokens: list[int] = field(default_factory=list)
    response_tokens: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    response_text: str = ""


class GRPOModel:
    """
    A model that captures token_ids and logprobs for GRPO training.

    Compatible with mini-swe-agent-plus's Model interface.
    Uses vLLM's completions API with logprobs enabled.
    """

    def __init__(self, config: GRPOModelConfig = None, **kwargs):
        self.config = config or GRPOModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        # Accumulated data across all turns
        self.turns: list[TurnData] = []
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        return self._tokenizer

    def _call_vllm_completions(self, prompt: str) -> dict:
        """
        Call vLLM completions API with logprobs enabled.

        Returns:
            Dict with 'text', 'token_ids', 'logprobs'
        """
        url = f"{self.config.vllm_url}/v1/completions"

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "logprobs": 1,  # Return top-1 logprob
            "echo": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise

        choice = result["choices"][0]
        text = choice["text"]

        # Extract token_ids and logprobs from response
        logprobs_data = choice.get("logprobs", {})
        token_ids = logprobs_data.get("tokens", [])
        token_logprobs = logprobs_data.get("token_logprobs", [])

        # vLLM returns token strings, we need to convert to IDs
        tokenizer = self._get_tokenizer()

        # If vLLM returned token strings, encode them
        if token_ids and isinstance(token_ids[0], str):
            # Re-encode the generated text to get token IDs
            response_token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            response_token_ids = token_ids

        # Align logprobs with token_ids
        if len(token_logprobs) != len(response_token_ids):
            # Fallback: re-tokenize and use available logprobs
            response_token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_logprobs) > len(response_token_ids):
                token_logprobs = token_logprobs[:len(response_token_ids)]
            elif len(token_logprobs) < len(response_token_ids):
                # Pad with zeros (shouldn't happen, but be safe)
                token_logprobs = token_logprobs + [0.0] * (len(response_token_ids) - len(token_logprobs))

        return {
            "text": text,
            "token_ids": response_token_ids,
            "logprobs": token_logprobs,
        }

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert chat messages to a single prompt string."""
        tokenizer = self._get_tokenizer()

        # Try to use chat template if available
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple concatenation
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}\n")
                elif role == "user":
                    parts.append(f"User: {content}\n")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}\n")
            parts.append("Assistant: ")
            prompt = "".join(parts)

        return prompt

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """
        Query the model and capture GRPO data.

        This is compatible with mini-swe-agent-plus's Model interface.
        """
        self.n_calls += 1

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Tokenize prompt (for recording)
        tokenizer = self._get_tokenizer()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        # Call vLLM
        result = self._call_vllm_completions(prompt)

        # Record turn data
        turn = TurnData(
            prompt_tokens=prompt_tokens,
            response_tokens=result["token_ids"],
            response_logprobs=result["logprobs"],
            response_text=result["text"],
        )
        self.turns.append(turn)

        # Return in mini-swe-agent-plus format
        return {
            "content": result["text"],
            "extra": {
                "token_ids": result["token_ids"],
                "logprobs": result["logprobs"],
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables (mini-swe-agent-plus interface)."""
        return {
            "model_name": self.config.model_name,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }

    def get_rollout_data(self) -> dict:
        """
        Get accumulated rollout data for GRPO training.

        Returns:
            Dict with:
            - prompt_tokens: list[int] - First turn's prompt tokens
            - response_tokens: list[int] - All response tokens concatenated
            - logprobs: list[float] - All logprobs concatenated
            - n_turns: int - Number of turns
        """
        if not self.turns:
            return {
                "prompt_tokens": [],
                "response_tokens": [],
                "logprobs": [],
                "n_turns": 0,
            }

        # First turn's prompt is the initial prompt
        prompt_tokens = self.turns[0].prompt_tokens

        # Concatenate all response tokens and logprobs
        all_response_tokens = []
        all_logprobs = []

        for turn in self.turns:
            all_response_tokens.extend(turn.response_tokens)
            all_logprobs.extend(turn.response_logprobs)

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": all_response_tokens,
            "logprobs": all_logprobs,
            "n_turns": len(self.turns),
        }

    def reset(self):
        """Reset accumulated data for a new episode."""
        self.turns = []
        self.n_calls = 0
        self.cost = 0.0


def run_agent_for_grpo(
    instance_id: str,
    problem_statement: str,
    container_name: str,
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    vllm_url: str = "http://localhost:8000",
    temperature: float = 1.0,
    max_tokens: int = 2048,
    step_limit: int = 30,
    cost_limit: float = 3.0,
    config_path: str = None,
    working_dir: str = "/testbed",
) -> dict:
    """
    Run mini-swe-agent-plus on a SWE-bench instance and return GRPO data.

    Args:
        instance_id: SWE-bench instance ID
        problem_statement: The problem/issue description
        container_name: Docker container name (swebench container)
        model_name: HuggingFace model name
        vllm_url: vLLM server URL
        temperature: Sampling temperature
        max_tokens: Max tokens per turn
        step_limit: Max agent steps
        cost_limit: Cost limit (not used for vLLM)
        config_path: Path to agent config YAML (optional)
        working_dir: Working directory in container (default: /testbed)

    Returns:
        Dict with:
        - prompt_tokens: list[int]
        - response_tokens: list[int]
        - logprobs: list[float]
        - patch: str - Generated patch
        - exit_status: str - How agent exited
        - n_turns: int
    """
    from minisweagent.agents.default import DefaultAgent, AgentConfig

    # Load config if provided (templates from config file, explicit params override)
    agent_kwargs = {}

    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        if "agent" in config_data:
            agent_kwargs.update(config_data["agent"])

    # Explicit parameters override config file
    agent_kwargs["step_limit"] = step_limit
    agent_kwargs["cost_limit"] = cost_limit

    # Create GRPO model
    model = GRPOModel(
        model_name=model_name,
        vllm_url=vllm_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Create Docker environment (executes commands in swebench container)
    env = DockerEnvironment(
        container_name=container_name,
        working_dir=working_dir,
        timeout=120,
    )

    # Create agent
    agent = DefaultAgent(model=model, env=env, **agent_kwargs)

    # Run agent
    try:
        exit_status, exit_message = agent.run(task=problem_statement)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        exit_status = "Error"
        exit_message = str(e)

    # Get patch from git diff (inside container)
    try:
        result = subprocess.run(
            ["docker", "exec", "-w", working_dir, container_name, "git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        patch = result.stdout
    except Exception as e:
        logger.error(f"Failed to get patch: {e}")
        patch = ""

    # Get rollout data
    rollout_data = model.get_rollout_data()

    return {
        "instance_id": instance_id,
        "prompt_tokens": rollout_data["prompt_tokens"],
        "response_tokens": rollout_data["response_tokens"],
        "logprobs": rollout_data["logprobs"],
        "patch": patch,
        "exit_status": exit_status,
        "exit_message": exit_message,
        "n_turns": rollout_data["n_turns"],
    }


# For testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Kwai-Klear/Klear-AgentForge-8B-SFT")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--container-name", required=True, help="Docker container name")
    args = parser.parse_args()

    # Simple test
    result = run_agent_for_grpo(
        instance_id="test-001",
        problem_statement="Fix the bug in the code",
        container_name=args.container_name,
        model_name=args.model_name,
        vllm_url=args.vllm_url,
    )

    print(f"Exit status: {result['exit_status']}")
    print(f"N turns: {result['n_turns']}")
    print(f"Response tokens: {len(result['response_tokens'])}")
    print(f"Patch length: {len(result['patch'])}")
