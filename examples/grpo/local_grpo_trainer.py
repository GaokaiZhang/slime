#!/usr/bin/env python
"""
Local GRPO Trainer with swebench.harness Evaluation.

This implements GRPO training locally with:
1. vLLM server (Modal or local) for inference
2. swebench.harness for accurate reward computation
3. Local weight updates with LoRA
4. Docker-based evaluation

Architecture:
- Inference: vLLM server (Modal A100 or local)
- Evaluation: Local Docker with swebench.harness
- Training: Local GPU with LoRA (efficient training)

Usage:
    # Deploy vLLM server first
    modal deploy examples/grpo/modal_vllm.py

    # Run training with swebench evaluation
    export VLLM_URL="https://susvibes-mitigation--slime-grpo-vllm-serve-vllm.modal.run"
    python examples/grpo/local_grpo_trainer.py \
        --num-rollouts 50 \
        --n-samples 4 \
        --use-swebench-eval

    # For testing without Docker (heuristic rewards only)
    python examples/grpo/local_grpo_trainer.py \
        --num-rollouts 10 \
        --n-samples 4
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO training configuration (Search-R1 defaults)."""
    # Model
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    vllm_url: str = "http://localhost:8000"

    # GRPO hyperparameters (Search-R1)
    lr: float = 1e-6
    kl_coef: float = 0.001
    n_samples_per_prompt: int = 4
    temperature: float = 1.0
    gamma: float = 1.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    # Training
    num_rollouts: int = 50
    gradient_accumulation_steps: int = 4
    max_new_tokens: int = 1024
    max_prompt_len: int = 2048

    # LoRA
    use_lora: bool = True
    lora_r: int = 16

    # Evaluation
    use_swebench_eval: bool = False
    eval_timeout: int = 300

    # Output
    output_dir: str = "outputs/local_grpo"
    save_every: int = 10


@dataclass
class RolloutSample:
    """Single rollout sample."""
    response_text: str = ""
    token_ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    reward: float = -1.0


def call_vllm_generate(
    vllm_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> dict:
    """Call vLLM server to generate response with logprobs."""
    url = f"{vllm_url}/v1/chat/completions"

    messages = [
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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
            logprobs_content = choice["logprobs"].get("content", [])
            for item in logprobs_content:
                logprobs.append(item.get("logprob", 0.0))
                tokens.append(item.get("token", ""))

        return {
            "content": content,
            "logprobs": logprobs,
            "tokens": tokens,
        }

    except Exception as e:
        logger.error(f"vLLM call failed: {e}")
        return {"content": "", "logprobs": [], "tokens": []}


def evaluate_with_swebench(instance_id: str, patch: str, timeout: int = 300) -> float:
    """
    Evaluate patch using swebench.harness.

    Returns:
        +1.0 if patch resolves the issue
        -1.0 otherwise
    """
    if not patch or not patch.strip():
        return -1.0

    try:
        # Create temporary prediction file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            prediction = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "grpo-local",
            }
            f.write(json.dumps(prediction) + "\n")
            pred_file = f.name

        # Run swebench evaluation
        run_id = f"grpo_{instance_id.replace('/', '_')}_{int(time.time())}"

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

        logger.info(f"Running swebench evaluation for {instance_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )

        # Find evaluation result file
        eval_file = None
        for pattern in [
            f"grpo-local.{run_id}.json",
            f"*.{run_id}.json",
        ]:
            matches = list(Path('.').glob(pattern))
            if matches:
                eval_file = matches[0]
                break

        if eval_file and eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            resolved_ids = eval_data.get("resolved_ids", [])

            # Clean up
            eval_file.unlink()
            os.unlink(pred_file)

            if instance_id in resolved_ids:
                logger.info(f"[{instance_id}] RESOLVED!")
                return 1.0
            else:
                logger.info(f"[{instance_id}] Not resolved")
                return -1.0
        else:
            logger.warning(f"[{instance_id}] No evaluation result found")
            os.unlink(pred_file)
            return -1.0

    except subprocess.TimeoutExpired:
        logger.error(f"[{instance_id}] Evaluation timeout")
        return -1.0
    except Exception as e:
        logger.error(f"[{instance_id}] Evaluation error: {e}")
        return -1.0


def compute_heuristic_reward(response: str) -> float:
    """Compute heuristic reward based on patch quality."""
    import re

    has_diff_header = ("--- a/" in response or "--- " in response) and \
                      ("+++ b/" in response or "+++ " in response)
    has_hunk_header = "@@" in response

    if has_diff_header and has_hunk_header:
        file_pattern = r'---\s+[ab]/[\w/\._-]+\.py'
        if re.search(file_pattern, response):
            return 1.0
        return 0.5

    if has_diff_header or has_hunk_header:
        return 0.0

    if "```diff" in response.lower():
        return 0.0

    return -1.0


def extract_patch(response: str) -> str:
    """Extract git diff patch from response."""
    import re

    # Try to find diff block
    diff_pattern = r'```diff\n(.*?)```'
    match = re.search(diff_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find raw diff
    lines = response.split('\n')
    diff_lines = []
    in_diff = False

    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
            if line.strip() == '' and len(diff_lines) > 5:
                # End of diff hunk
                break

    if diff_lines:
        return '\n'.join(diff_lines)

    return response


class LocalGRPOTrainer:
    """Local GRPO trainer with swebench evaluation."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model for training
        logger.info(f"Loading model for training: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        if config.use_lora:
            from peft import LoraConfig, get_peft_model

            logger.info(f"Applying LoRA with r={config.lora_r}")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_r * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Reference model for KL
        logger.info("Loading reference model for KL...")
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
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1,
        )

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def generate_rollout(
        self,
        prompt: str,
        instance_id: str,
    ) -> RolloutSample:
        """Generate a single rollout using vLLM server."""
        result = call_vllm_generate(
            vllm_url=self.config.vllm_url,
            model_name=self.config.model_name,
            prompt=prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        response_text = result["content"]
        logprobs = result["logprobs"]
        tokens = result["tokens"]

        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token:
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                if encoded:
                    token_ids.append(encoded[0])

        # Compute reward
        if self.config.use_swebench_eval:
            patch = extract_patch(response_text)
            reward = evaluate_with_swebench(
                instance_id, patch, self.config.eval_timeout
            )
        else:
            reward = compute_heuristic_reward(response_text)

        return RolloutSample(
            response_text=response_text,
            token_ids=token_ids,
            logprobs=logprobs,
            reward=reward,
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
            sample = self.generate_rollout(prompt, instance_id)
            samples.append(sample)
            logger.info(f"  Sample {i+1}: reward={sample.reward:.2f}, tokens={len(sample.token_ids)}")
        return samples

    def compute_grpo_loss(
        self,
        prompt: str,
        samples: list[RolloutSample],
    ) -> tuple[torch.Tensor, dict]:
        """Compute GRPO loss for a group of samples."""
        # Compute advantages
        rewards = [s.reward for s in samples]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_reward = max(std_reward, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        # Tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_len,
        ).to(self.device)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        total_loss = 0.0
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

            # Old log probs
            old_log_probs = torch.tensor(sample.logprobs, device=self.device)

            # Align lengths
            min_len = min(len(policy_token_log_probs), len(old_log_probs))
            policy_token_log_probs = policy_token_log_probs[:min_len]
            ref_token_log_probs = ref_token_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]

            # Policy loss
            ratio = torch.exp(policy_token_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.eps_clip,
                1 + self.config.eps_clip_high
            )
            advantage_tensor = torch.tensor(advantage, device=self.device)
            policy_loss = torch.max(
                -advantage_tensor * ratio,
                -advantage_tensor * clipped_ratio
            ).mean()

            # KL loss (low-variance)
            kl_ratio = torch.exp(policy_token_log_probs - ref_token_log_probs)
            kl_loss = 0.5 * ((kl_ratio - 1) ** 2).mean()

            # Total loss
            loss = policy_loss + self.config.kl_coef * kl_loss

            total_loss += loss
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
            "n_valid": n_valid,
        }

        return total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(0.0), metrics

    def train(self):
        """Main training loop."""
        logger.info("=" * 70)
        logger.info("Local GRPO Training with swebench.harness Evaluation")
        logger.info("=" * 70)

        # Check vLLM connection
        try:
            resp = requests.get(f"{self.config.vllm_url}/health", timeout=10)
            logger.info(f"vLLM server status: {resp.json()}")
        except Exception as e:
            logger.error(f"Cannot connect to vLLM server: {e}")
            logger.error(f"Please deploy vLLM first: modal deploy examples/grpo/modal_vllm.py")
            return

        # Load data
        logger.info("Loading SWE-bench data...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        django_instances = [x for x in ds if x["repo"] == "django/django"]
        train_instances = django_instances[:self.config.num_rollouts]
        logger.info(f"Training on {len(train_instances)} instances")

        # Create prompt
        def create_prompt(instance):
            return f"""You are an expert software engineer. Fix this bug and provide a git diff patch.

## Repository: {instance["repo"]}
## Problem Statement
{instance["problem_statement"][:2000]}

## Instructions
Provide ONLY a git diff patch that fixes this issue. Format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
-old line
+new line
```

Your patch:"""

        # Training loop
        all_metrics = []
        start_time = time.time()

        for rollout_idx, instance in enumerate(tqdm(train_instances, desc="Training")):
            instance_id = instance["instance_id"]
            prompt = create_prompt(instance)

            logger.info(f"\n[{rollout_idx + 1}/{len(train_instances)}] {instance_id}")

            # Generate group
            samples = self.generate_group(
                prompt, instance_id, self.config.n_samples_per_prompt
            )

            # Compute loss
            self.model.train()
            loss, metrics = self.compute_grpo_loss(prompt, samples)

            # Backward
            if loss.requires_grad:
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            # Update weights
            if (rollout_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log
            metrics["instance_id"] = instance_id
            metrics["rollout_idx"] = rollout_idx
            all_metrics.append(metrics)

            logger.info(f"  Loss: {metrics['loss']:.4f}, Reward: {metrics['mean_reward']:.3f}")

            # Save checkpoint
            if (rollout_idx + 1) % self.config.save_every == 0:
                checkpoint_dir = os.path.join(
                    self.config.output_dir, f"checkpoint_{rollout_idx + 1}"
                )
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)

                with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
                    json.dump(all_metrics, f, indent=2)

        # Save final model
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info(f"Avg reward: {sum(m['mean_reward'] for m in all_metrics) / len(all_metrics):.3f}")

        final_dir = os.path.join(self.config.output_dir, "final_model")
        logger.info(f"Saving final model to {final_dir}")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Local GRPO Training")
    parser.add_argument("--model-name", type=str,
                       default="Kwai-Klear/Klear-AgentForge-8B-SFT")
    parser.add_argument("--vllm-url", type=str,
                       default=os.environ.get("VLLM_URL", "http://localhost:8000"))
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs/local_grpo")
    parser.add_argument("--use-swebench-eval", action="store_true",
                       help="Use swebench.harness for evaluation (requires Docker)")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    config = GRPOConfig(
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        num_rollouts=args.num_rollouts,
        n_samples_per_prompt=args.n_samples,
        lr=args.lr,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
        output_dir=args.output_dir,
        use_swebench_eval=args.use_swebench_eval,
        use_lora=args.use_lora,
    )

    trainer = LocalGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
