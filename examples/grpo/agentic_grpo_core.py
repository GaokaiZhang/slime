#!/usr/bin/env python
"""
Agentic GRPO Core - Shared Implementation for Modal and Local GPU

This module contains the shared logic for agentic GRPO training:
1. Agent rollout with multi-turn Docker interaction
2. Tool response masking
3. GRPO training step using SLiME's ppo_utils

Both `agentic_grpo_trainer.py` (local) and `agentic_grpo_modal.py` (Modal)
import from this module to ensure identical behavior.

Search-R1 GRPO Parameters:
- lr: 1e-6
- kl_coef: 0.001
- kl_loss_type: low_var_kl
- eps_clip: 0.2, eps_clip_high: 0.28 (DAPO asymmetric)
- temperature: 1.0
- n_samples_per_prompt: 4
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
import torch
import torch.nn.functional as F

# Import SLiME's GRPO utilities
from slime.utils.ppo_utils import (
    compute_approx_kl,
    compute_policy_loss,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration with Search-R1 Parameters
# ==============================================================================

@dataclass
class AgenticGRPOConfig:
    """Configuration with Search-R1 GRPO parameters."""

    # Model
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    vllm_url: str = "http://localhost:8000"

    # Agent
    max_turns: int = 30
    tool_timeout: int = 120
    max_response_tokens: int = 4096  # Truncate to prevent OOM

    # Search-R1 GRPO Parameters
    lr: float = 1e-6
    kl_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    n_samples_per_prompt: int = 4
    temperature: float = 1.0
    eps_clip: float = 0.2  # Lower bound
    eps_clip_high: float = 0.28  # Upper bound (DAPO asymmetric)
    gamma: float = 1.0  # No discounting
    max_new_tokens: int = 2048

    # LoRA
    use_lora: bool = True
    lora_r: int = 16

    # Evaluation
    eval_timeout: int = 300

    # Memory optimization
    gradient_checkpointing: bool = True


# ==============================================================================
# Agent Turn Data Structure
# ==============================================================================

@dataclass
class TurnData:
    """Data from a single model turn (for GRPO training)."""
    token_ids: list = field(default_factory=list)  # Model output token IDs
    logprobs: list = field(default_factory=list)   # Log probs for each token
    text: str = ""                                  # Decoded text


@dataclass
class AgentRollout:
    """Complete agent rollout with properly separated data."""
    instance_id: str = ""
    turns: list = field(default_factory=list)  # List of TurnData
    env_observations: list = field(default_factory=list)  # Environment responses (masked)
    patch: str = ""
    reward: float = -1.0
    exit_status: str = ""

    def get_model_token_ids(self) -> list:
        """Get all model-generated token IDs (for training)."""
        all_ids = []
        for turn in self.turns:
            all_ids.extend(turn.token_ids)
        return all_ids

    def get_model_logprobs(self) -> list:
        """Get all model-generated log probs (for training)."""
        all_logprobs = []
        for turn in self.turns:
            all_logprobs.extend(turn.logprobs)
        return all_logprobs

    def total_model_tokens(self) -> int:
        """Total number of model-generated tokens."""
        return sum(len(turn.token_ids) for turn in self.turns)


# ==============================================================================
# vLLM API with Log Probs
# ==============================================================================

def call_vllm_with_logprobs(
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: int = 600,
) -> dict:
    """
    Call vLLM chat completions API with logprobs enabled.

    Returns:
        {
            "content": str,
            "token_ids": list[int],  # May be empty if not supported
            "logprobs": list[float],
        }
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": 1,
    }

    response = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    choice = data["choices"][0]
    content = choice["message"]["content"]

    # Extract logprobs
    token_ids = []
    logprobs = []

    if "logprobs" in choice and choice["logprobs"]:
        logprobs_content = choice["logprobs"].get("content", [])
        for item in logprobs_content:
            logprobs.append(item.get("logprob", 0.0))

    return {
        "content": content,
        "token_ids": token_ids,
        "logprobs": logprobs,
    }


# ==============================================================================
# Docker Environment for Tool Execution
# ==============================================================================

class DockerToolEnvironment:
    """Execute tools in a swebench Docker container."""

    def __init__(self, container_id: str, workdir: str = "/testbed"):
        self.container_id = container_id
        self.workdir = workdir

    def execute_bash(self, command: str, timeout: int = 120) -> str:
        """Execute bash command in container."""
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.container_id, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return output[:10000]
        except subprocess.TimeoutExpired:
            return "[Timeout]"
        except Exception as e:
            return f"[Error: {e}]"

    def read_file(self, path: str) -> str:
        """Read file from container."""
        if not path.startswith("/"):
            path = f"{self.workdir}/{path}"
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_id, "cat", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout[:50000]
        except Exception as e:
            return f"[Error reading file: {e}]"

    def write_file(self, path: str, content: str) -> str:
        """Write file in container."""
        if not path.startswith("/"):
            path = f"{self.workdir}/{path}"
        try:
            cmd = f"cat > {path} << 'EOFMARKER'\n{content}\nEOFMARKER"
            result = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.container_id, "bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return f"Written to {path}"
            return f"[Error: {result.stderr}]"
        except Exception as e:
            return f"[Error writing file: {e}]"

    def get_git_diff(self) -> str:
        """Get git diff of changes."""
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.container_id, "git", "diff"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout
        except Exception as e:
            return ""


def start_swebench_container(instance_id: str) -> str | None:
    """Start a swebench Docker container for an instance."""
    try:
        from examples.grpo.swebench_utils import get_docker_image
        image = get_docker_image(instance_id)
    except ImportError:
        # Fallback
        parts = instance_id.split("__")
        image = f"swebench/sweb.eval.x86_64.{parts[0]}_{instance_id.replace('__', '_')}"

    if not image:
        logger.error(f"No Docker image found for {instance_id}")
        return None

    try:
        result = subprocess.run(
            ["docker", "run", "-d", "--workdir", "/testbed", image, "sleep", "3600"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            container_id = result.stdout.strip()[:12]
            logger.info(f"Started container {container_id} for {instance_id}")
            return container_id
    except Exception as e:
        logger.error(f"Failed to start container: {e}")

    return None


def stop_container(container_id: str):
    """Stop and remove container."""
    if container_id:
        try:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)
        except:
            pass


# ==============================================================================
# Tool Parsing
# ==============================================================================

def parse_tool_call(content: str) -> tuple[str, dict] | None:
    """Parse tool call from model output."""
    try:
        for match in re.finditer(r'\{[^{}]*"tool"[^{}]*\}', content):
            try:
                data = json.loads(match.group())
                if "tool" in data and "args" in data:
                    return data["tool"], data["args"]
            except:
                pass

        json_match = re.search(r'\{.*?"tool"\s*:\s*"([^"]+)".*?"args"\s*:\s*(\{[^}]*\}).*?\}', content, re.DOTALL)
        if json_match:
            tool = json_match.group(1)
            args = json.loads(json_match.group(2))
            return tool, args
    except:
        pass

    return None


# ==============================================================================
# Agent System Prompt
# ==============================================================================

SYSTEM_PROMPT = """You are an expert software engineer fixing bugs in a codebase.

Available tools:
- {"tool": "bash", "args": {"command": "..."}} - Execute bash command
- {"tool": "read_file", "args": {"path": "..."}} - Read file
- {"tool": "write_file", "args": {"path": "...", "content": "..."}} - Write file
- {"tool": "submit", "args": {"patch": "..."}} - Submit solution

Think step by step. Explore, understand, fix, test, then submit your patch."""


# ==============================================================================
# Agent Rollout
# ==============================================================================

def run_agent_rollout(
    instance_id: str,
    problem_statement: str,
    config: AgenticGRPOConfig,
    tokenizer,
) -> AgentRollout:
    """
    Run a single agent rollout with proper data collection.

    Key points:
    - Captures token_ids and logprobs ONLY for model turns
    - Environment observations are stored separately (masked from loss)
    - Uses vLLM API for generation
    """
    rollout = AgentRollout(instance_id=instance_id)

    # Start container
    container_id = start_swebench_container(instance_id)
    if not container_id:
        rollout.exit_status = "container_failed"
        return rollout

    env = DockerToolEnvironment(container_id)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Fix this issue:\n\n{problem_statement[:3000]}"},
        ]

        for turn_idx in range(config.max_turns):
            try:
                response = call_vllm_with_logprobs(
                    url=config.vllm_url,
                    model=config.model_name,
                    messages=messages,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
            except Exception as e:
                logger.error(f"vLLM error: {e}")
                rollout.exit_status = "vllm_error"
                break

            content = response["content"]

            # Tokenize response to get token_ids
            if not response["token_ids"]:
                token_ids = tokenizer.encode(content, add_special_tokens=False)
            else:
                token_ids = response["token_ids"]

            logprobs = response["logprobs"]

            # Align logprobs with token_ids
            if len(logprobs) != len(token_ids):
                if len(logprobs) < len(token_ids):
                    logprobs = logprobs + [0.0] * (len(token_ids) - len(logprobs))
                else:
                    logprobs = logprobs[:len(token_ids)]

            # Store model turn data (INCLUDED in training)
            turn = TurnData(
                token_ids=token_ids,
                logprobs=logprobs,
                text=content,
            )
            rollout.turns.append(turn)

            messages.append({"role": "assistant", "content": content})

            # Parse tool call
            tool_call = parse_tool_call(content)

            if tool_call:
                tool_name, tool_args = tool_call

                if tool_name == "submit":
                    rollout.patch = tool_args.get("patch", "")
                    rollout.exit_status = "submitted"
                    break

                # Execute tool
                if tool_name == "bash":
                    observation = env.execute_bash(tool_args.get("command", ""))
                elif tool_name == "read_file":
                    observation = env.read_file(tool_args.get("path", ""))
                elif tool_name == "write_file":
                    observation = env.write_file(
                        tool_args.get("path", ""),
                        tool_args.get("content", "")
                    )
                else:
                    observation = f"Unknown tool: {tool_name}"

                # Store observation (MASKED from training)
                rollout.env_observations.append(observation[:5000])
                messages.append({"role": "user", "content": f"Result:\n{observation[:5000]}"})
            else:
                if "submit" in content.lower() or turn_idx >= config.max_turns - 1:
                    rollout.patch = env.get_git_diff()
                    rollout.exit_status = "completed"
                    break

        if not rollout.exit_status:
            rollout.exit_status = "max_turns"
            rollout.patch = env.get_git_diff()

    finally:
        stop_container(container_id)

    return rollout


# ==============================================================================
# swebench.harness Evaluation
# ==============================================================================

def evaluate_patch(instance_id: str, patch: str, timeout: int = 300) -> float:
    """Evaluate patch with swebench.harness. Returns +1.0 if resolved, -1.0 otherwise."""
    if not patch or not patch.strip():
        return -1.0

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            prediction = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "agentic-grpo",
            }
            f.write(json.dumps(prediction) + "\n")
            pred_file = f.name

        run_id = f"agrpo_{instance_id.replace('/', '_')}_{int(time.time())}"

        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--split", "test",
            "--predictions_path", pred_file,
            "--max_workers", "1",
            "--timeout", str(timeout),
            "--run_id", run_id,
            "--instance_ids", instance_id,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 120,
        )

        for pattern in ["agentic-grpo.*.json", f"*{run_id}*.json"]:
            matches = list(Path('.').glob(pattern))
            if matches:
                with open(matches[0]) as f:
                    data = json.load(f)
                resolved = data.get("resolved_ids", [])
                try:
                    matches[0].unlink()
                except:
                    pass
                if instance_id in resolved:
                    return 1.0
                break

        try:
            os.unlink(pred_file)
        except:
            pass

        return -1.0

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return -1.0


# ==============================================================================
# GRPO Advantage Computation
# ==============================================================================

def compute_grpo_advantages(rewards: list[float]) -> tuple[list[float], float, float]:
    """
    Compute GRPO group-relative advantages.

    Returns: (advantages, mean_reward, std_reward)
    """
    mean_reward = sum(rewards) / len(rewards)

    if len(rewards) > 1:
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = max(var ** 0.5, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        std_reward = 0.0
        advantages = [r - mean_reward for r in rewards]

    return advantages, mean_reward, std_reward


# ==============================================================================
# GRPO Training Step with Proper Masking
# ==============================================================================

def train_grpo_step(
    prompt: str,
    rollouts: list[AgentRollout],
    model,
    ref_model,
    tokenizer,
    optimizer,
    config: AgenticGRPOConfig,
    device: torch.device,
) -> dict:
    """
    GRPO training step with proper masking.

    Key insight: Only model-generated tokens contribute to gradient.
    Environment observations are context but do NOT contribute to loss.
    """
    # Compute GRPO advantages
    rewards = [r.reward for r in rollouts]
    advantages, mean_reward, std_reward = compute_grpo_advantages(rewards)

    logger.info(f"  Rewards: {rewards}, Mean: {mean_reward:.3f}, Std: {std_reward:.3f}")

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    n_valid = 0

    # Tokenize prompt once
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    prompt_ids = prompt_inputs["input_ids"][0].to(device)

    for rollout, advantage in zip(rollouts, advantages):
        model_token_ids = rollout.get_model_token_ids()
        model_logprobs = rollout.get_model_logprobs()

        if len(model_token_ids) == 0:
            continue

        # Truncate long sequences to prevent OOM
        if len(model_token_ids) > config.max_response_tokens:
            logger.warning(f"Truncating response from {len(model_token_ids)} to {config.max_response_tokens} tokens")
            model_token_ids = model_token_ids[:config.max_response_tokens]
            model_logprobs = model_logprobs[:config.max_response_tokens]

        response_ids = torch.tensor(model_token_ids, device=device)
        old_logprobs = torch.tensor(model_logprobs, device=device, dtype=torch.float32)

        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        prompt_len = len(prompt_ids)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            policy_outputs = model(full_ids, return_dict=True)
            with torch.no_grad():
                ref_outputs = ref_model(full_ids, return_dict=True)

        # Extract log probs for response tokens
        response_logits = policy_outputs.logits[0, prompt_len - 1:-1]
        ref_logits = ref_outputs.logits[0, prompt_len - 1:-1]

        policy_log_probs = F.log_softmax(response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        policy_token_log_probs = policy_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        # Align with old log probs
        if len(old_logprobs) != len(policy_token_log_probs):
            min_len = min(len(old_logprobs), len(policy_token_log_probs))
            old_logprobs = old_logprobs[:min_len]
            policy_token_log_probs = policy_token_log_probs[:min_len]
            ref_token_log_probs = ref_token_log_probs[:min_len]

        # GRPO loss using SLiME's utilities
        ppo_kl = old_logprobs - policy_token_log_probs
        advantages_tensor = torch.full_like(policy_token_log_probs, advantage)

        # Policy loss
        pg_losses, clipfrac = compute_policy_loss(
            ppo_kl=ppo_kl,
            advantages=advantages_tensor,
            eps_clip=config.eps_clip,
            eps_clip_high=config.eps_clip_high,
            eps_clip_c=None,
        )
        policy_loss = pg_losses.mean()

        # KL loss
        kl = compute_approx_kl(
            log_probs=policy_token_log_probs,
            log_probs_base=ref_token_log_probs,
            kl_loss_type=config.kl_loss_type,
        )
        kl_loss = kl.mean()

        # Total loss
        loss = policy_loss + config.kl_coef * kl_loss
        loss.backward()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()
        n_valid += 1

    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        "loss": total_loss / max(n_valid, 1),
        "policy_loss": total_policy_loss / max(n_valid, 1),
        "kl_loss": total_kl_loss / max(n_valid, 1),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_valid": n_valid,
    }


# ==============================================================================
# Model Loading Utilities
# ==============================================================================

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    use_lora: bool = True,
    lora_r: int = 16,
    gradient_checkpointing: bool = True,
):
    """Load model and tokenizer with optional LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_reference_model(model_name: str, device: torch.device):
    """Load reference model for KL computation."""
    from transformers import AutoModelForCausalLM

    logger.info("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


# ==============================================================================
# Training Data Loading
# ==============================================================================

def load_training_instances(
    train_file: str = "/home/gaokaizhang/slime/train_instances_id.txt",
    num_instances: int = None,
    test_mode: bool = False,
) -> list[dict]:
    """Load SWE-bench training instances."""
    from datasets import load_dataset

    train_path = Path(train_file)
    if train_path.exists():
        with open(train_path) as f:
            train_ids = [line.strip() for line in f if line.strip()]
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        id_to_instance = {x["instance_id"]: x for x in ds}
        train_instances = [id_to_instance[iid] for iid in train_ids if iid in id_to_instance]
    else:
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        train_instances = [x for x in ds if x["repo"] == "django/django"]

    if test_mode:
        train_instances = train_instances[:2]
        logger.info(f"Test mode: {len(train_instances)} instances")
    elif num_instances:
        train_instances = train_instances[:num_instances]
        logger.info(f"Training on {len(train_instances)} instances")
    else:
        logger.info(f"Training on {len(train_instances)} instances")

    return train_instances


# ==============================================================================
# Prompt Creation
# ==============================================================================

def create_training_prompt(instance: dict) -> str:
    """Create prompt for training from SWE-bench instance."""
    return f"Fix this issue in {instance['repo']}:\n\n{instance['problem_statement'][:3000]}"
