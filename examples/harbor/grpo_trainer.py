"""
Standalone GRPO Trainer for SWE-bench with Harbor evaluation.

This implements GRPO (Group Relative Policy Optimization) without requiring
the full SLiME/Megatron infrastructure. Uses:
- vLLM for inference (external server)
- Our vllm_agent for rollouts
- Harbor for evaluation
- PyTorch + DeepSpeed/FSDP for training

Based on Search-R1 GRPO hyperparameters.
"""

import asyncio
import json
import logging
import os
import sys
import time
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO training configuration (Search-R1 defaults)."""
    # Model
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    vllm_url: str = "http://localhost:8000"

    # GRPO hyperparameters (Search-R1)
    lr: float = 1e-6
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"  # k1, k2, k3, low_var_kl
    n_samples_per_prompt: int = 5
    temperature: float = 1.0
    gamma: float = 1.0  # No discounting
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    # Training
    num_rollouts: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_turns: int = 50
    max_response_len: int = 4096

    # Output
    output_dir: str = "outputs/grpo_training"
    save_interval: int = 10

    # Data
    train_instances_file: str = "train_instances_id.txt"
    test_instances_file: str = "test_instances_id.txt"


class GRPOSample:
    """Single GRPO sample with all required data."""

    def __init__(
        self,
        prompt_tokens: list[int],
        response_tokens: list[int],
        logprobs: list[float],
        reward: float,
        instance_id: str,
    ):
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.tokens = prompt_tokens + response_tokens  # Full sequence
        self.response_length = len(response_tokens)
        self.logprobs = logprobs  # π_old(a|s)
        self.reward = reward
        self.instance_id = instance_id


def compute_grpo_advantages(
    rewards: list[float],
    gamma: float = 1.0,
    normalize: bool = True,
) -> list[float]:
    """
    Compute GRPO advantages (group-relative).

    For GRPO, advantage = reward - mean(rewards_in_group)
    Optionally normalize by std.
    """
    mean_reward = sum(rewards) / len(rewards)

    if normalize and len(rewards) > 1:
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_reward = max(std_reward, 1e-8)  # Avoid division by zero
        advantages = [(r - mean_reward) / std_reward for r in rewards]
    else:
        advantages = [r - mean_reward for r in rewards]

    return advantages


def compute_kl_loss(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
    kl_loss_type: str = "low_var_kl",
) -> torch.Tensor:
    """
    Compute KL divergence loss.

    Args:
        log_probs_policy: log π(a|s) from current policy
        log_probs_ref: log π_ref(a|s) from reference policy
        kl_loss_type: Type of KL approximation

    Returns:
        KL loss tensor
    """
    if kl_loss_type == "k1":
        # Standard KL: E_π[log π - log π_ref]
        kl = log_probs_policy - log_probs_ref
    elif kl_loss_type == "k2":
        # Reverse KL: E_π_ref[log π_ref - log π]
        kl = log_probs_ref - log_probs_policy
    elif kl_loss_type == "k3":
        # Symmetric KL
        kl = 0.5 * (log_probs_policy - log_probs_ref) + 0.5 * (log_probs_ref - log_probs_policy)
    elif kl_loss_type == "low_var_kl":
        # Low variance KL from Search-R1
        # KL = 0.5 * (ratio - 1)^2 where ratio = π/π_ref
        ratio = torch.exp(log_probs_policy - log_probs_ref)
        kl = 0.5 * (ratio - 1) ** 2
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    return kl.mean()


def compute_policy_loss(
    log_probs_policy: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
) -> torch.Tensor:
    """
    Compute clipped policy loss (PPO-style).

    Args:
        log_probs_policy: log π(a|s) from current policy
        log_probs_old: log π_old(a|s) from rollout policy
        advantages: Advantage estimates
        eps_clip: Lower clip range
        eps_clip_high: Upper clip range (DAPO-style)

    Returns:
        Policy loss tensor
    """
    # Importance sampling ratio
    ratio = torch.exp(log_probs_policy - log_probs_old)

    # Clipped ratio with asymmetric bounds (DAPO-style)
    clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)

    # Policy loss (minimize negative advantage * ratio)
    loss1 = -advantages * ratio
    loss2 = -advantages * clipped_ratio

    # Take the more conservative loss
    loss = torch.max(loss1, loss2)

    return loss.mean()


class GRPOTrainer:
    """GRPO Trainer for SWE-bench."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (for training)
        logger.info(f"Loading model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Reference model (for KL)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1,
        )

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

    async def generate_rollouts(
        self,
        prompts: list[str],
        instance_ids: list[str],
    ) -> list[list[GRPOSample]]:
        """
        Generate rollouts using vLLM agent.

        Returns groups of samples (n_samples_per_prompt per prompt).
        """
        from examples.harbor.vllm_agent import VLLMAgentConfig, run_agent, get_tokenizer
        from examples.harbor.rollout import evaluate_with_harbor

        vllm_tokenizer = get_tokenizer(self.config.model_name)

        all_groups = []

        for prompt, instance_id in zip(prompts, instance_ids):
            logger.info(f"Generating rollouts for {instance_id}...")

            group = []

            for sample_idx in range(self.config.n_samples_per_prompt):
                config = VLLMAgentConfig(
                    api_url=self.config.vllm_url,
                    model_name=self.config.model_name,
                    max_tokens=self.config.max_response_len,
                    temperature=self.config.temperature,
                    max_turns=self.config.max_turns,
                )

                try:
                    result = await asyncio.to_thread(
                        run_agent,
                        prompt,
                        config,
                        workdir="/tmp",
                        tokenizer=vllm_tokenizer,
                    )

                    # Encode prompt
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

                    # Evaluate reward
                    args = Namespace(use_harbor_eval=True)
                    sample = Namespace(metadata={})
                    reward = await evaluate_with_harbor(
                        instance_id=instance_id,
                        patch=result.patch,
                        sample=sample,
                        args=args,
                    )

                    grpo_sample = GRPOSample(
                        prompt_tokens=prompt_tokens,
                        response_tokens=result.completion_token_ids,
                        logprobs=result.logprobs,
                        reward=reward,
                        instance_id=instance_id,
                    )
                    group.append(grpo_sample)

                    logger.info(f"  Sample {sample_idx}: reward={reward}, tokens={len(result.completion_token_ids)}")

                except Exception as e:
                    logger.error(f"  Sample {sample_idx} failed: {e}")
                    # Add failed sample with negative reward
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                    grpo_sample = GRPOSample(
                        prompt_tokens=prompt_tokens,
                        response_tokens=[],
                        logprobs=[],
                        reward=-1.0,
                        instance_id=instance_id,
                    )
                    group.append(grpo_sample)

            all_groups.append(group)

        return all_groups

    def compute_log_probs(
        self,
        model: nn.Module,
        tokens: torch.Tensor,
        response_start: int,
    ) -> torch.Tensor:
        """Compute log probabilities for response tokens."""
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(tokens, return_dict=True)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Get log probs of actual tokens
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Only return response token log probs
        response_log_probs = token_log_probs[..., response_start-1:]

        return response_log_probs

    def train_step(self, groups: list[list[GRPOSample]]) -> dict[str, float]:
        """
        Single GRPO training step.

        Args:
            groups: List of sample groups (each group = samples for same prompt)

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_samples = 0

        for group in groups:
            if not group or all(len(s.response_tokens) == 0 for s in group):
                continue

            # Compute GRPO advantages
            rewards = [s.reward for s in group]
            advantages = compute_grpo_advantages(rewards, gamma=self.config.gamma)

            for sample, advantage in zip(group, advantages):
                if len(sample.response_tokens) == 0:
                    continue

                # Prepare input
                tokens = torch.tensor([sample.tokens], device=self.device)
                response_start = len(sample.prompt_tokens)

                # Compute log probs from current policy
                log_probs_policy = self.compute_log_probs(self.model, tokens, response_start)

                # Compute log probs from reference policy
                with torch.no_grad():
                    log_probs_ref = self.compute_log_probs(self.ref_model, tokens, response_start)

                # Get rollout log probs
                log_probs_old = torch.tensor(sample.logprobs, device=self.device)

                # Ensure same length
                min_len = min(log_probs_policy.shape[-1], log_probs_old.shape[-1], log_probs_ref.shape[-1])
                log_probs_policy = log_probs_policy[..., :min_len]
                log_probs_ref = log_probs_ref[..., :min_len]
                log_probs_old = log_probs_old[:min_len]

                # Compute losses
                advantage_tensor = torch.tensor(advantage, device=self.device)

                policy_loss = compute_policy_loss(
                    log_probs_policy.mean(),
                    log_probs_old.mean(),
                    advantage_tensor,
                    eps_clip=self.config.eps_clip,
                    eps_clip_high=self.config.eps_clip_high,
                )

                kl_loss = compute_kl_loss(
                    log_probs_policy.mean(),
                    log_probs_ref.mean(),
                    kl_loss_type=self.config.kl_loss_type,
                )

                loss = policy_loss + self.config.kl_loss_coef * kl_loss

                # Accumulate gradients
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.config.gradient_accumulation_steps
                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_loss.item()
                num_samples += 1

        # Update weights
        if num_samples > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": total_loss / max(num_samples, 1),
            "policy_loss": total_policy_loss / max(num_samples, 1),
            "kl_loss": total_kl_loss / max(num_samples, 1),
            "num_samples": num_samples,
            "mean_reward": sum(s.reward for g in groups for s in g) / max(sum(len(g) for g in groups), 1),
        }

    def save_checkpoint(self, rollout_id: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint_{rollout_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        with open(os.path.join(checkpoint_dir, "grpo_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    async def train(self):
        """Main training loop."""
        from examples.harbor.data_source import DjangoTrainDataSource, BUG_SOLVING_PROMPT

        logger.info("=" * 60)
        logger.info("GRPO Training")
        logger.info("=" * 60)

        # Load training data
        logger.info("Loading training data...")
        ds = DjangoTrainDataSource(limit=self.config.num_rollouts)
        logger.info(f"Loaded {len(ds)} instances")

        # Training loop
        for rollout_id in range(self.config.num_rollouts):
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Rollout {rollout_id + 1}/{self.config.num_rollouts}")
            logger.info(f"{'=' * 40}")

            # Get batch of prompts
            batch_samples = ds.get_samples(self.config.batch_size)

            prompts = []
            instance_ids = []
            for group in batch_samples:
                sample = group[0]  # All samples in group have same prompt
                prompts.append(sample.prompt)
                instance_ids.append(sample.metadata["instance_id"])

            # Generate rollouts
            logger.info("Generating rollouts...")
            groups = await self.generate_rollouts(prompts, instance_ids)

            # Train on rollouts
            logger.info("Training...")
            metrics = self.train_step(groups)

            logger.info(f"Metrics: loss={metrics['loss']:.4f}, "
                       f"policy_loss={metrics['policy_loss']:.4f}, "
                       f"kl_loss={metrics['kl_loss']:.4f}, "
                       f"mean_reward={metrics['mean_reward']:.3f}")

            # Save checkpoint
            if (rollout_id + 1) % self.config.save_interval == 0:
                self.save_checkpoint(rollout_id + 1)

        # Save final model
        self.save_checkpoint(self.config.num_rollouts)
        logger.info("Training complete!")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training for SWE-bench")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--vllm-url", type=str, default=os.environ.get("VLLM_URL", "http://localhost:8000"))
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-samples-per-prompt", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_training")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-loss-coef", type=float, default=0.001)
    args = parser.parse_args()

    config = GRPOConfig(
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        num_rollouts=args.num_rollouts,
        batch_size=args.batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
        output_dir=args.output_dir,
        lr=args.lr,
        kl_loss_coef=args.kl_loss_coef,
    )

    trainer = GRPOTrainer(config)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
