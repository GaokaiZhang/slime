#!/usr/bin/env python
"""
GRPO Trainer using mini-swe-agent-plus.

This trainer uses mini-swe-agent-plus (multi-turn agent with tools) instead of
simple single-turn generation. The agent interacts with the codebase via bash
commands and a text-edit tool.

Architecture:
- vLLM on Modal: Model inference with logprobs
- Local Docker: swebench environment for agent execution
- Local: GRPO training loop

Usage:
    # First deploy vLLM on Modal
    modal deploy examples/grpo/modal_vllm.py

    # Run training
    python examples/grpo/mini_swe_grpo_trainer.py \
        --num-rollouts 50 \
        --n-samples 2 \
        --vllm-url https://your-modal-vllm-endpoint.modal.run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "submodules" / "mini-swe-agent-plus" / "src"))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GRPORollout:
    """A single GRPO rollout from mini-swe-agent-plus."""
    instance_id: str
    prompt_tokens: list[int] = field(default_factory=list)
    response_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    patch: str = ""
    reward: float = 0.0
    n_turns: int = 0
    exit_status: str = ""


class MiniSweGRPOTrainer:
    """
    GRPO trainer using mini-swe-agent-plus as the agent.

    Key differences from simple generation:
    1. Multi-turn agent with tools (bash, edit)
    2. Agent runs in swebench Docker container
    3. Token_ids and logprobs captured across all turns
    """

    def __init__(
        self,
        model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
        vllm_url: str = "http://localhost:8000",
        temperature: float = 1.0,
        lr: float = 1e-6,
        kl_coef: float = 0.001,
        use_lora: bool = True,
        lora_r: int = 16,
        agent_step_limit: int = 30,
        agent_config_path: str = None,
    ):
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.temperature = temperature
        self.lr = lr
        self.kl_coef = kl_coef
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.agent_step_limit = agent_step_limit
        self.agent_config_path = agent_config_path or str(
            Path(__file__).parent.parent.parent /
            "submodules" / "mini-swe-agent-plus" / "src" /
            "minisweagent" / "config" / "extra" / "swebench_add_edit_tool.yaml"
        )

        # Training state
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = None

    def setup_models(self):
        """Load model, reference model, and tokenizer for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model for training
        logger.info(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_r * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA with r={self.lora_r}")

        # Load reference model (frozen)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
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
            lr=self.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1,
        )

    def run_agent(self, instance: dict) -> GRPORollout:
        """
        Run mini-swe-agent-plus on an instance and capture rollout data.

        Uses swebench container for environment.
        """
        from grpo_core import setup_swebench_container, cleanup_container

        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]

        logger.info(f"    Running mini-swe-agent-plus on {instance_id}")

        # Setup swebench container
        container_name, repo_path = setup_swebench_container(instance_id)
        if not container_name:
            logger.error(f"    Failed to setup container for {instance_id}")
            return GRPORollout(
                instance_id=instance_id,
                exit_status="ContainerSetupFailed",
                reward=-1.0,
            )

        try:
            # Import agent components
            from grpo_agent import run_agent_for_grpo

            # Run agent (container_name for Docker, repo_path is working_dir inside container)
            result = run_agent_for_grpo(
                instance_id=instance_id,
                problem_statement=problem_statement,
                container_name=container_name,
                model_name=self.model_name,
                vllm_url=self.vllm_url,
                temperature=self.temperature,
                max_tokens=2048,
                step_limit=self.agent_step_limit,
                config_path=self.agent_config_path,
                working_dir=repo_path,
            )

            return GRPORollout(
                instance_id=instance_id,
                prompt_tokens=result["prompt_tokens"],
                response_tokens=result["response_tokens"],
                logprobs=result["logprobs"],
                patch=result["patch"],
                n_turns=result["n_turns"],
                exit_status=result["exit_status"],
            )

        except Exception as e:
            logger.error(f"    Agent error: {e}")
            return GRPORollout(
                instance_id=instance_id,
                exit_status=f"Error: {e}",
                reward=-1.0,
            )

        finally:
            cleanup_container(container_name)

    def evaluate_rollout(self, rollout: GRPORollout, instance_id: str, timeout: int = 300) -> float:
        """Evaluate rollout with swebench.harness."""
        from grpo_core import evaluate_with_swebench

        if not rollout.patch:
            logger.info(f"    No patch generated")
            return -1.0

        reward = evaluate_with_swebench(instance_id, rollout.patch, timeout)
        return reward

    def compute_grpo_loss(
        self,
        rollouts: list[GRPORollout],
        rewards: list[float],
    ) -> dict:
        """
        Compute GRPO loss from rollouts.

        Uses SLiME's ppo_utils for loss computation.
        """
        import torch.nn.functional as F
        from slime.utils.ppo_utils import (
            compute_approx_kl,
            compute_policy_loss,
        )

        # Compute advantages (group-relative)
        mean_reward = sum(rewards) / len(rewards)
        if len(rewards) > 1:
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_reward = max(variance ** 0.5, 1e-8)
            advantages = [(r - mean_reward) / std_reward for r in rewards]
        else:
            std_reward = 0.0
            advantages = [r - mean_reward for r in rewards]

        logger.info(f"  Rewards: {rewards}")
        logger.info(f"  Mean: {mean_reward:.3f}, Std: {std_reward:.3f}")
        logger.info(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_valid = 0

        # GRPO hyperparameters
        eps_clip = 0.2
        eps_clip_high = 0.28

        for rollout, reward, advantage in zip(rollouts, rewards, advantages):
            if not rollout.response_tokens or not rollout.logprobs:
                continue

            # Build full sequence
            prompt_ids = torch.tensor(rollout.prompt_tokens, device=self.device)
            response_ids = torch.tensor(rollout.response_tokens, device=self.device)
            full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)

            # Forward through policy model
            policy_outputs = self.model(full_ids, return_dict=True)
            policy_logits = policy_outputs.logits

            # Forward through reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(full_ids, return_dict=True)
                ref_logits = ref_outputs.logits

            # Compute log probs for response tokens
            prompt_length = len(rollout.prompt_tokens)
            response_start = prompt_length - 1
            response_end = full_ids.shape[1] - 1

            policy_log_probs = F.log_softmax(
                policy_logits[0, response_start:response_end], dim=-1
            )
            ref_log_probs = F.log_softmax(
                ref_logits[0, response_start:response_end], dim=-1
            )

            # Gather logprobs for generated tokens
            policy_token_log_probs = policy_log_probs.gather(
                -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_log_probs = ref_log_probs.gather(
                -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Old logprobs from generation
            old_log_probs = torch.tensor(rollout.logprobs, device=self.device)

            # Align lengths
            min_len = min(len(policy_token_log_probs), len(old_log_probs))
            policy_token_log_probs = policy_token_log_probs[:min_len]
            ref_token_log_probs = ref_token_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]

            # Policy loss (PPO-style with DAPO clipping)
            advantages_tensor = torch.full_like(policy_token_log_probs, advantage)
            ppo_kl = old_log_probs - policy_token_log_probs
            pg_losses, clipfrac = compute_policy_loss(
                ppo_kl=ppo_kl,
                advantages=advantages_tensor,
                eps_clip=eps_clip,
                eps_clip_high=eps_clip_high,
                eps_clip_c=None,
            )
            policy_loss = pg_losses.mean()

            # KL loss (low-variance)
            kl = compute_approx_kl(
                log_probs=policy_token_log_probs,
                log_probs_base=ref_token_log_probs,
                kl_loss_type="low_var_kl",
            )
            kl_loss = kl.mean()

            # Total loss
            loss = policy_loss + self.kl_coef * kl_loss
            loss.backward()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            n_valid += 1

        # Update weights
        if n_valid > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "loss": total_loss / max(n_valid, 1),
            "policy_loss": total_policy_loss / max(n_valid, 1),
            "kl_loss": total_kl_loss / max(n_valid, 1),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_valid": n_valid,
        }

    def train(
        self,
        instances: list[dict],
        n_samples: int = 2,
        output_dir: str = "outputs/mini_swe_grpo",
        save_every: int = 10,
        eval_timeout: int = 300,
    ):
        """
        Main training loop.

        Args:
            instances: List of SWE-bench instances
            n_samples: Samples per instance for GRPO
            output_dir: Output directory
            save_every: Save checkpoint every N rollouts
            eval_timeout: Evaluation timeout in seconds
        """
        os.makedirs(output_dir, exist_ok=True)

        self.setup_models()

        all_metrics = []
        total_resolved = 0
        total_samples = 0
        start_time = time.time()

        for rollout_idx, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            logger.info(f"\n[{rollout_idx + 1}/{len(instances)}] {instance_id}")

            # Generate n_samples rollouts
            rollouts = []
            rewards = []

            for sample_idx in range(n_samples):
                logger.info(f"  Sample {sample_idx + 1}/{n_samples}")

                # Run agent
                rollout = self.run_agent(instance)
                rollouts.append(rollout)

                # Evaluate
                reward = self.evaluate_rollout(rollout, instance_id, eval_timeout)
                rollout.reward = reward
                rewards.append(reward)

                if reward > 0:
                    total_resolved += 1
                total_samples += 1

                logger.info(f"    Exit: {rollout.exit_status}, Turns: {rollout.n_turns}, Reward: {reward}")

            # Compute GRPO loss and update
            metrics = self.compute_grpo_loss(rollouts, rewards)

            logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")

            # Record metrics
            metrics["instance_id"] = instance_id
            metrics["rollout_idx"] = rollout_idx
            metrics["rewards"] = rewards
            all_metrics.append(metrics)

            # Save checkpoint
            if (rollout_idx + 1) % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{rollout_idx + 1}")
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"  Saved checkpoint to {checkpoint_path}")

            # Save metrics
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)

        # Final summary
        elapsed = time.time() - start_time
        resolve_rate = total_resolved / total_samples if total_samples > 0 else 0

        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Total rollouts: {len(instances)}")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Total resolved: {total_resolved} ({resolve_rate*100:.1f}%)")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")

        return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Mini-SWE-Agent GRPO Trainer")
    parser.add_argument("--model-name", type=str, default="Kwai-Klear/Klear-AgentForge-8B-SFT")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs/mini_swe_grpo")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-timeout", type=int, default=300)
    parser.add_argument("--agent-step-limit", type=int, default=30)
    args = parser.parse_args()

    # Load data
    from datasets import load_dataset
    logger.info("Loading SWE-bench data...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
    django_instances = [x for x in ds if x["repo"] == "django/django"]
    train_instances = django_instances[:args.num_rollouts]
    logger.info(f"Training on {len(train_instances)} Django instances")

    # Create trainer
    trainer = MiniSweGRPOTrainer(
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        temperature=args.temperature,
        lr=args.lr,
        kl_coef=args.kl_coef,
        agent_step_limit=args.agent_step_limit,
    )

    # Train
    trainer.train(
        instances=train_instances,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        save_every=args.save_every,
        eval_timeout=args.eval_timeout,
    )


if __name__ == "__main__":
    main()
