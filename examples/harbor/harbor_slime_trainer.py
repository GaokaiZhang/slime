#!/usr/bin/env python
"""
Harbor + SLiME Trainer

Integrates Harbor (trajectory generation) with SLiME (RL training).

Architecture:
    Harbor CLI (qwen-code agent) → Trajectories → SLiME GRPO Training

Key insight:
- Log probs are NOT collected during Harbor rollout
- Log probs are recomputed at training time via forward pass
- This enables using any Harbor agent (qwen-code, openhands, etc.)

Usage:
    python harbor_slime_trainer.py \\
        --model Qwen/Qwen3-Coder-30B-A3B \\
        --dataset swebench-verified@1.0 \\
        --n-rollouts 50
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for Harbor + SLiME training."""

    # Model
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B"

    # Harbor agent
    agent: str = "qwen-coder"
    n_concurrent: int = 4

    # Dataset
    dataset: str = "swebench-verified@1.0"
    task_filter: str | None = None  # e.g., "django__*" for Django tasks only

    # Training
    n_rollouts: int = 50
    n_samples_per_prompt: int = 2  # GRPO group size
    lr: float = 1e-6
    kl_coef: float = 0.001
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    # LoRA
    use_lora: bool = True
    lora_r: int = 16

    # Output
    output_dir: str = "outputs/harbor_slime"
    jobs_dir: str = "jobs"
    save_every: int = 10


class HarborSlimeTrainer:
    """
    Trainer that uses Harbor for rollouts and SLiME utilities for GRPO training.

    Workflow:
    1. Use Harbor CLI to run qwen-code agent on tasks
    2. Parse trajectories (tokens, reward)
    3. Forward pass to compute current log probs
    4. Compute GRPO loss and update

    No log probs from inference - computed at training time.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config

        # Training state (lazy init)
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = None

        # Harbor generator (lazy init)
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            from harbor_rollout import HarborRolloutGenerator, HarborConfig

            harbor_config = HarborConfig(
                model=self.config.model_name,
                agent=self.config.agent,
                n_concurrent=self.config.n_concurrent,
                jobs_dir=self.config.jobs_dir,
                export_traces=True,
            )

            self._generator = HarborRolloutGenerator(
                config=harbor_config,
                tokenizer=self.tokenizer,
            )

        return self._generator

    def setup_models(self):
        """Load model, reference model, and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Apply LoRA if configured
        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_r * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA with r={self.config.lora_r}")

        # Load reference model (frozen)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1,
        )

    def compute_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        response_start: int,
    ) -> torch.Tensor:
        """
        Compute log probabilities for response tokens.

        This is the key function - log probs are computed here at training time,
        NOT during Harbor rollout.

        Args:
            model: The model to compute log probs with
            input_ids: Full sequence token IDs [1, seq_len]
            response_start: Index where response starts

        Returns:
            Log probabilities for each response token
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits

        # Get log probs for response tokens
        # Shift: logits[t] predicts token[t+1]
        response_logits = logits[0, response_start - 1:-1]  # [response_len, vocab]
        response_tokens = input_ids[0, response_start:]  # [response_len]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)

        return token_log_probs

    def compute_grpo_loss(
        self,
        samples: list,
        rewards: list[float],
    ) -> dict[str, float]:
        """
        Compute GRPO loss from samples.

        Uses SLiME's ppo_utils for loss computation.

        Args:
            samples: List of SLiME Sample objects
            rewards: List of rewards for each sample

        Returns:
            Dict with loss metrics
        """
        # Import SLiME utilities
        try:
            from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
        except ImportError:
            logger.warning("SLiME ppo_utils not available, using local implementation")
            compute_approx_kl = self._compute_approx_kl
            compute_policy_loss = self._compute_policy_loss

        # Compute advantages (group-relative, no sigma normalization for SSR)
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]

        logger.info(f"  Rewards: {[f'{r:.2f}' for r in rewards]}")
        logger.info(f"  Mean: {mean_reward:.3f}, Advantages: {[f'{a:.3f}' for a in advantages]}")

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_valid = 0

        for sample, advantage in zip(samples, advantages):
            if not sample.tokens or sample.response_length == 0:
                continue

            # Prepare input
            input_ids = torch.tensor(sample.tokens, device=self.device).unsqueeze(0)
            response_start = len(sample.tokens) - sample.response_length

            # Compute current policy log probs
            policy_log_probs = self.compute_log_probs(
                self.model, input_ids, response_start
            )

            # Compute reference log probs
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(
                    self.ref_model, input_ids, response_start
                )

            # For on-policy training, old_log_probs ≈ policy_log_probs
            # We use detached policy_log_probs as "old" since Harbor didn't provide them
            old_log_probs = policy_log_probs.detach()

            # PPO KL (old - new, for ratio computation)
            ppo_kl = old_log_probs - policy_log_probs

            # Advantage tensor
            advantages_tensor = torch.full_like(policy_log_probs, advantage)

            # Policy loss
            pg_losses, clipfrac = compute_policy_loss(
                ppo_kl=ppo_kl,
                advantages=advantages_tensor,
                eps_clip=self.config.eps_clip,
                eps_clip_high=self.config.eps_clip_high,
                eps_clip_c=None,
            )
            policy_loss = pg_losses.mean()

            # KL loss (vs reference)
            kl = compute_approx_kl(
                log_probs=policy_log_probs,
                log_probs_base=ref_log_probs,
                kl_loss_type="low_var_kl",
            )
            kl_loss = kl.mean()

            # Total loss
            loss = policy_loss + self.config.kl_coef * kl_loss
            loss.backward()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            n_valid += 1

        # Update
        if n_valid > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "loss": total_loss / max(n_valid, 1),
            "policy_loss": total_policy_loss / max(n_valid, 1),
            "kl_loss": total_kl_loss / max(n_valid, 1),
            "mean_reward": mean_reward,
            "n_valid": n_valid,
        }

    @staticmethod
    def _compute_approx_kl(log_probs, log_probs_base, kl_loss_type="low_var_kl", **kwargs):
        """Fallback KL computation if SLiME not available."""
        log_ratio = log_probs.float() - log_probs_base.float()
        if kl_loss_type == "low_var_kl":
            log_ratio = -log_ratio
            kl = log_ratio.exp() - 1 - log_ratio
            kl = torch.clamp(kl, min=-10, max=10)
        else:
            kl = log_ratio
        return kl

    @staticmethod
    def _compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
        """Fallback policy loss if SLiME not available."""
        ratio = (-ppo_kl).exp()
        pg_losses1 = -ratio * advantages
        pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
        pg_losses = torch.maximum(pg_losses1, pg_losses2)
        clipfrac = torch.gt(pg_losses2, pg_losses1).float()
        return pg_losses, clipfrac

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup models
        self.setup_models()

        # Metrics tracking
        all_metrics = []
        total_resolved = 0
        total_samples = 0
        start_time = time.time()

        # Generate rollouts using Harbor
        logger.info(f"Generating {self.config.n_rollouts} rollouts using Harbor...")
        logger.info(f"Agent: {self.config.agent}, Model: {self.config.model_name}")

        # Generate all samples at once (Harbor handles batching internally)
        all_samples = self.generator.generate(
            dataset=self.config.dataset,
            n_tasks=self.config.n_rollouts,
        )

        if not all_samples:
            logger.error("No samples generated!")
            return []

        # Group samples for GRPO (n_samples_per_prompt per group)
        groups = []
        for i in range(0, len(all_samples), self.config.n_samples_per_prompt):
            group = all_samples[i:i + self.config.n_samples_per_prompt]
            if len(group) == self.config.n_samples_per_prompt:
                groups.append(group)

        logger.info(f"Created {len(groups)} groups for GRPO training")

        # Training loop
        for group_idx, group in enumerate(groups):
            logger.info(f"\n[{group_idx + 1}/{len(groups)}] Training on group")

            rewards = [s.reward for s in group]
            total_samples += len(group)
            total_resolved += sum(1 for r in rewards if r > 0)

            # Compute GRPO loss and update
            metrics = self.compute_grpo_loss(group, rewards)

            logger.info(
                f"  Loss: {metrics['loss']:.4f} "
                f"(policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})"
            )

            # Record metrics
            metrics["group_idx"] = group_idx
            metrics["rewards"] = rewards
            all_metrics.append(metrics)

            # Save checkpoint
            if (group_idx + 1) % self.config.save_every == 0:
                checkpoint_path = os.path.join(
                    self.config.output_dir, f"checkpoint_{group_idx + 1}"
                )
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"  Saved checkpoint to {checkpoint_path}")

            # Save metrics
            with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)

        # Final summary
        elapsed = time.time() - start_time
        resolve_rate = total_resolved / total_samples if total_samples > 0 else 0

        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Total groups: {len(groups)}")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Total resolved: {total_resolved} ({resolve_rate*100:.1f}%)")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")

        # Save final model
        final_path = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Saved final model to {final_path}")

        return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Harbor + SLiME GRPO Trainer")

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B",
                        help="Model name (default: Qwen3-Coder-30B)")

    # Agent
    parser.add_argument("--agent", default="qwen-coder",
                        help="Harbor agent (default: qwen-coder)")

    # Dataset
    parser.add_argument("--dataset", default="swebench-verified@1.0",
                        help="Harbor dataset")
    parser.add_argument("--n-rollouts", type=int, default=50,
                        help="Number of rollouts")
    parser.add_argument("--n-samples", type=int, default=2,
                        help="Samples per prompt for GRPO")

    # Training
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=16)

    # Concurrency
    parser.add_argument("--n-concurrent", type=int, default=4,
                        help="Harbor concurrent trials")

    # Output
    parser.add_argument("--output-dir", default="outputs/harbor_slime")
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--save-every", type=int, default=10)

    args = parser.parse_args()

    config = TrainerConfig(
        model_name=args.model,
        agent=args.agent,
        dataset=args.dataset,
        n_rollouts=args.n_rollouts,
        n_samples_per_prompt=args.n_samples,
        n_concurrent=args.n_concurrent,
        lr=args.lr,
        kl_coef=args.kl_coef,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        output_dir=args.output_dir,
        jobs_dir=args.jobs_dir,
        save_every=args.save_every,
    )

    trainer = HarborSlimeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
