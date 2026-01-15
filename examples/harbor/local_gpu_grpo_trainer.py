#!/usr/bin/env python
"""
Local GPU GRPO Trainer with swebench.harness Evaluation.

This implements full GRPO training on local GPU with:
1. Model loaded on local GPU (with LoRA for memory efficiency)
2. Inference using local vLLM server OR HuggingFace generate
3. swebench.harness for evaluation (Docker required)
4. Weight updates on local GPU

NO heuristic rewards - only swebench.harness evaluation.

Uses IDENTICAL GRPO implementation as hybrid_grpo_trainer.py via grpo_core.py.

Prerequisites:
- Local GPU with 24GB+ VRAM (for 8B model with LoRA)
- Docker with SWE-bench images pulled
- swebench package installed

Usage:
    # Option 1: Use local vLLM for inference (recommended for speed)
    # First start vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model Kwai-Klear/Klear-AgentForge-8B-SFT \
        --port 8000 --dtype bfloat16

    # Then run training:
    python examples/harbor/local_gpu_grpo_trainer.py \
        --num-rollouts 50 \
        --n-samples 4 \
        --vllm-url http://localhost:8000

    # Option 2: Use HuggingFace generate (slower but no vLLM needed)
    python examples/harbor/local_gpu_grpo_trainer.py \
        --num-rollouts 50 \
        --n-samples 4 \
        --use-hf-generate

Environment variables:
    HF_TOKEN: HuggingFace token for model access
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import shared GRPO implementation
sys.path.insert(0, str(Path(__file__).parent))
from grpo_core import (
    GRPOConfig as BaseGRPOConfig,
    RolloutSample,
    extract_patch,
    evaluate_with_swebench,
    compute_grpo_advantages,
    compute_kl_loss,
    compute_policy_loss,
    create_swebench_prompt,
    setup_lora,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LocalGRPOConfig(BaseGRPOConfig):
    """Extended GRPO config for local GPU training."""
    # Inference
    vllm_url: Optional[str] = None  # If None, use HF generate
    use_hf_generate: bool = False

    # Training
    num_rollouts: int = 50

    # Output
    output_dir: str = "outputs/local_gpu_grpo"
    save_every: int = 10


class LocalGPUGRPOTrainer:
    """
    Local GPU GRPO trainer with swebench.harness evaluation.

    Uses the SAME GRPO implementation as hybrid_grpo_trainer.py.
    """

    def __init__(self, config: LocalGRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This trainer requires a GPU.")

        logger.info(f"Device: {self.device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load training model
        logger.info(f"Loading training model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )

        if config.use_lora:
            logger.info(f"Applying LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
            self.model = setup_lora(self.model, config)
            self.model.print_trainable_parameters()

        # Load reference model for KL computation
        logger.info("Loading reference model for KL...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer (Search-R1 settings)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1,
        )

        # Check vLLM if specified
        if config.vllm_url and not config.use_hf_generate:
            self._check_vllm_connection()

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _check_vllm_connection(self):
        """Check if vLLM server is accessible."""
        import requests
        try:
            resp = requests.get(f"{self.config.vllm_url}/health", timeout=10)
            logger.info(f"vLLM server connected: {resp.status_code}")
        except Exception as e:
            logger.warning(f"vLLM server not accessible: {e}")
            logger.warning("Will fall back to HuggingFace generate")
            self.config.use_hf_generate = True

    def generate_with_vllm(self, prompt: str) -> dict:
        """Generate response using vLLM API."""
        import requests

        url = f"{self.config.vllm_url}/v1/chat/completions"
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]

            # Extract logprobs
            logprobs = []
            tokens = []
            if "logprobs" in choice and choice["logprobs"]:
                for item in choice["logprobs"].get("content", []):
                    logprobs.append(item.get("logprob", 0.0))
                    tokens.append(item.get("token", ""))

            # Convert tokens to IDs
            token_ids = []
            for token in tokens:
                if token:
                    encoded = self.tokenizer.encode(token, add_special_tokens=False)
                    if encoded:
                        token_ids.append(encoded[0])

            return {
                "content": content,
                "logprobs": logprobs,
                "token_ids": token_ids,
            }

        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return {"content": "", "logprobs": [], "token_ids": []}

    def generate_with_hf(self, prompt: str) -> dict:
        """Generate response using HuggingFace generate."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_len,
        ).to(self.device)

        prompt_length = inputs["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract response tokens
        response_ids = outputs.sequences[0, prompt_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Compute log probabilities
        logprobs = []
        for i, score in enumerate(outputs.scores):
            if i < len(response_ids):
                probs = F.log_softmax(score[0], dim=-1)
                token_id = response_ids[i].item()
                logprobs.append(probs[token_id].item())

        return {
            "content": response_text,
            "logprobs": logprobs,
            "token_ids": response_ids.tolist(),
        }

    def generate_rollout(self, prompt: str, instance_id: str) -> RolloutSample:
        """Generate a single rollout and evaluate with swebench.harness."""
        # Generate response
        if self.config.use_hf_generate or not self.config.vllm_url:
            result = self.generate_with_hf(prompt)
        else:
            result = self.generate_with_vllm(prompt)

        response_text = result["content"]
        logprobs = result["logprobs"]
        token_ids = result["token_ids"]

        # Extract patch from response (using shared function from grpo_core)
        patch = extract_patch(response_text)

        # Evaluate with swebench.harness (ONLY reward function - NO HEURISTICS)
        reward = evaluate_with_swebench(
            instance_id,
            patch,
            self.config.eval_timeout
        )

        return RolloutSample(
            response_text=response_text,
            token_ids=token_ids,
            logprobs=logprobs,
            reward=reward,
            instance_id=instance_id,
            patch=patch,
        )

    def generate_group(
        self,
        prompt: str,
        instance_id: str,
        n_samples: int,
    ) -> list[RolloutSample]:
        """Generate a group of rollouts for GRPO."""
        samples = []
        for i in range(n_samples):
            logger.info(f"  Generating sample {i+1}/{n_samples}...")
            sample = self.generate_rollout(prompt, instance_id)
            samples.append(sample)
            logger.info(f"    reward={sample.reward:.1f}, tokens={len(sample.token_ids)}, patch_len={len(sample.patch)}")
        return samples

    def compute_grpo_loss(
        self,
        prompt: str,
        samples: list[RolloutSample],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute GRPO loss for a group of samples.

        Uses the SAME GRPO implementation as hybrid_grpo_trainer.py:
        - Group-relative advantages: (r_i - mean) / std
        - PPO-style clipping with DAPO asymmetric bounds [0.8, 1.28]
        - Low-variance KL: 0.5 * (ratio - 1)^2
        """
        # Compute group-relative advantages using shared function from grpo_core
        rewards = [s.reward for s in samples]
        advantages, mean_reward, std_reward = compute_grpo_advantages(rewards, normalize=True)

        # Tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_len,
        ).to(self.device)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        total_loss = torch.tensor(0.0, device=self.device)
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_valid = 0

        for sample, advantage in zip(samples, advantages):
            if len(sample.token_ids) == 0 or len(sample.logprobs) == 0:
                continue

            # Build full sequence
            response_ids = torch.tensor(sample.token_ids, device=self.device)
            full_ids = torch.cat([
                prompt_inputs["input_ids"][0],
                response_ids
            ]).unsqueeze(0)

            # Forward through policy model
            self.model.train()
            policy_outputs = self.model(full_ids, return_dict=True)
            policy_logits = policy_outputs.logits

            # Forward through reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(full_ids, return_dict=True)
                ref_logits = ref_outputs.logits

            # Compute log probs for response tokens
            response_start = prompt_length - 1
            response_end = full_ids.shape[1] - 1

            policy_log_probs = F.log_softmax(
                policy_logits[0, response_start:response_end], dim=-1
            )
            ref_log_probs = F.log_softmax(
                ref_logits[0, response_start:response_end], dim=-1
            )

            # Get log probs of actual tokens
            policy_token_log_probs = policy_log_probs.gather(
                -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_log_probs = ref_log_probs.gather(
                -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Old log probs from generation
            old_log_probs = torch.tensor(sample.logprobs, device=self.device)

            # Align lengths
            min_len = min(len(policy_token_log_probs), len(old_log_probs))
            policy_token_log_probs = policy_token_log_probs[:min_len]
            ref_token_log_probs = ref_token_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]

            # =========================================================================
            # GRPO Loss (Using SLiME's ppo_utils.py via grpo_core.py)
            # =========================================================================

            # Expand advantage to match log_probs shape
            advantages_tensor = torch.full_like(policy_token_log_probs, advantage)

            # Policy loss using SLiME's implementation
            policy_loss, clipfrac = compute_policy_loss(
                policy_token_log_probs,
                old_log_probs,
                advantages_tensor,
                eps_clip=self.config.eps_clip,
                eps_clip_high=self.config.eps_clip_high,
            )

            # KL loss using SLiME's implementation
            kl_loss = compute_kl_loss(
                policy_token_log_probs,
                ref_token_log_probs,
                kl_loss_type=getattr(self.config, 'kl_loss_type', 'low_var_kl'),
            )

            # Total loss
            loss = policy_loss + self.config.kl_coef * kl_loss

            total_loss = total_loss + loss
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        metrics = {
            "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0,
            "policy_loss": total_policy_loss / max(n_valid, 1),
            "kl_loss": total_kl_loss / max(n_valid, 1),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_valid": n_valid,
        }

        return total_loss, metrics

    def train(self):
        """Main training loop."""
        logger.info("=" * 70)
        logger.info("Local GPU GRPO Training with swebench.harness Evaluation")
        logger.info("=" * 70)
        logger.info(f"\nSearch-R1 GRPO Hyperparameters:")
        logger.info(f"  Learning rate: {self.config.lr}")
        logger.info(f"  KL coefficient: {self.config.kl_coef}")
        logger.info(f"  Temperature: {self.config.temperature}")
        logger.info(f"  Samples per prompt: {self.config.n_samples_per_prompt}")
        logger.info(f"  eps_clip: [{1-self.config.eps_clip:.2f}, {1+self.config.eps_clip_high:.2f}]")
        logger.info(f"\nInference mode: {'HuggingFace generate' if self.config.use_hf_generate else 'vLLM API'}")
        logger.info(f"Evaluation: swebench.harness (Docker) - NO HEURISTICS")

        # Load data
        logger.info("\nLoading SWE-bench data...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        django_instances = [x for x in ds if x["repo"] == "django/django"]
        train_instances = django_instances[:self.config.num_rollouts]
        logger.info(f"Training on {len(train_instances)} Django instances")

        # Training loop
        all_metrics = []
        start_time = time.time()
        total_resolved = 0
        total_samples = 0

        logger.info("\n" + "=" * 70)
        logger.info("Starting GRPO Training Loop")
        logger.info("=" * 70)

        for rollout_idx, instance in enumerate(tqdm(train_instances, desc="Training")):
            instance_id = instance["instance_id"]
            # Use shared prompt function from grpo_core
            prompt = create_swebench_prompt(instance)

            logger.info(f"\n[{rollout_idx + 1}/{len(train_instances)}] {instance_id}")

            # Generate group of samples
            samples = self.generate_group(
                prompt,
                instance_id,
                self.config.n_samples_per_prompt
            )

            # Count resolved
            resolved_in_group = sum(1 for s in samples if s.reward > 0)
            total_resolved += resolved_in_group
            total_samples += len(samples)

            # Compute GRPO loss
            loss, metrics = self.compute_grpo_loss(prompt, samples)

            # Backward pass
            if loss.requires_grad:
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            # Update weights
            if (rollout_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log metrics
            metrics["instance_id"] = instance_id
            metrics["rollout_idx"] = rollout_idx
            metrics["resolved_in_group"] = resolved_in_group
            all_metrics.append(metrics)

            logger.info(f"  Loss: {metrics['loss']:.4f} (policy={metrics['policy_loss']:.4f}, kl={metrics['kl_loss']:.4f})")
            logger.info(f"  Reward: mean={metrics['mean_reward']:.3f}, std={metrics['std_reward']:.3f}")
            logger.info(f"  Resolved: {resolved_in_group}/{len(samples)} in this group")

            # Save checkpoint
            if (rollout_idx + 1) % self.config.save_every == 0:
                checkpoint_dir = os.path.join(
                    self.config.output_dir, f"checkpoint_{rollout_idx + 1}"
                )
                logger.info(f"\nSaving checkpoint to {checkpoint_dir}")
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)

                # Save metrics
                with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
                    json.dump(all_metrics, f, indent=2)

        # Final summary
        elapsed = time.time() - start_time
        resolve_rate = total_resolved / total_samples if total_samples > 0 else 0

        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Total rollouts: {len(train_instances)}")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Total resolved: {total_resolved} ({resolve_rate*100:.1f}%)")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info(f"Avg reward: {sum(m['mean_reward'] for m in all_metrics) / len(all_metrics):.3f}")

        # Save final model
        final_dir = os.path.join(self.config.output_dir, "final_model")
        logger.info(f"\nSaving final model to {final_dir}")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save all metrics
        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Save training summary
        summary = {
            "total_rollouts": len(train_instances),
            "total_samples": total_samples,
            "total_resolved": total_resolved,
            "resolve_rate": resolve_rate,
            "elapsed_minutes": elapsed / 60,
            "avg_reward": sum(m['mean_reward'] for m in all_metrics) / len(all_metrics),
            "config": {
                "model_name": self.config.model_name,
                "lr": self.config.lr,
                "kl_coef": self.config.kl_coef,
                "n_samples_per_prompt": self.config.n_samples_per_prompt,
                "temperature": self.config.temperature,
                "use_lora": self.config.use_lora,
            }
        }
        with open(os.path.join(self.config.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def main():
    parser = argparse.ArgumentParser(description="Local GPU GRPO Training with swebench.harness")

    # Model
    parser.add_argument("--model-name", type=str,
                       default="Kwai-Klear/Klear-AgentForge-8B-SFT",
                       help="HuggingFace model name")

    # Inference
    parser.add_argument("--vllm-url", type=str, default=None,
                       help="vLLM server URL (e.g., http://localhost:8000)")
    parser.add_argument("--use-hf-generate", action="store_true",
                       help="Use HuggingFace generate instead of vLLM")

    # Training
    parser.add_argument("--num-rollouts", type=int, default=50,
                       help="Number of training instances")
    parser.add_argument("--n-samples", type=int, default=4,
                       help="Samples per prompt (group size)")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.001,
                       help="KL coefficient")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")

    # LoRA
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="Use LoRA for memory efficiency")
    parser.add_argument("--no-lora", action="store_false", dest="use_lora",
                       help="Disable LoRA (requires more VRAM)")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")

    # Evaluation
    parser.add_argument("--eval-timeout", type=int, default=300,
                       help="Timeout per swebench evaluation (seconds)")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/local_gpu_grpo",
                       help="Output directory")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N rollouts")

    args = parser.parse_args()

    config = LocalGRPOConfig(
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        use_hf_generate=args.use_hf_generate,
        num_rollouts=args.num_rollouts,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )

    trainer = LocalGPUGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
