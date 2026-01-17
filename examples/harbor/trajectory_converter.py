#!/usr/bin/env python
"""
Trajectory Converter: Harbor Output -> SLiME Sample

Converts Harbor job output (JSON trajectories) to SLiME Sample format
for RL training. Log probs are NOT needed from Harbor - they are
recomputed at training time by SLiME.

Usage:
    from trajectory_converter import HarborTrajectoryConverter

    converter = HarborTrajectoryConverter(tokenizer)
    samples = converter.load_job("jobs/my-job")
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryTurn:
    """A single turn in a multi-turn trajectory."""
    role: str  # "user" or "assistant"
    content: str
    token_ids: list[int] | None = None


@dataclass
class HarborTrajectory:
    """Parsed Harbor trajectory with all turns."""
    task_name: str
    instance_id: str
    turns: list[TrajectoryTurn]
    reward: float
    patch: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_resolved(self) -> bool:
        return self.reward > 0


class HarborTrajectoryConverter:
    """
    Converts Harbor job outputs to SLiME Sample format.

    Harbor outputs trajectories as JSON files with:
    - messages: list of {role, content} dicts
    - reward: float (from verifier)
    - metadata: task info

    SLiME expects:
    - tokens: list[int] - full sequence token IDs
    - response: str - generated text
    - response_length: int
    - reward: float
    - (optional) rollout_log_probs - NOT needed, computed at training time
    """

    def __init__(self, tokenizer, chat_template: bool = True):
        """
        Args:
            tokenizer: HuggingFace tokenizer for the model being trained
            chat_template: Whether to apply chat template when tokenizing
        """
        self.tokenizer = tokenizer
        self.chat_template = chat_template

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def load_job(self, job_dir: str | Path) -> list["Sample"]:
        """
        Load all trajectories from a Harbor job directory.

        Args:
            job_dir: Path to Harbor job output directory

        Returns:
            List of SLiME Sample objects
        """
        from slime.utils.types import Sample

        job_dir = Path(job_dir)
        samples = []

        # Harbor stores trials in subdirectories
        trial_dirs = list(job_dir.glob("trial-*"))
        if not trial_dirs:
            # Try flat structure
            trial_dirs = [job_dir]

        for trial_dir in trial_dirs:
            trajectory_file = trial_dir / "trajectory.json"
            if not trajectory_file.exists():
                # Try ATIF format
                trajectory_file = trial_dir / "trajectory.atif.json"

            if trajectory_file.exists():
                try:
                    sample = self._load_trajectory(trajectory_file)
                    if sample is not None:
                        samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to load {trajectory_file}: {e}")

        logger.info(f"Loaded {len(samples)} trajectories from {job_dir}")
        return samples

    def _load_trajectory(self, trajectory_file: Path) -> "Sample | None":
        """Load a single trajectory file and convert to Sample."""
        from slime.utils.types import Sample

        with open(trajectory_file) as f:
            data = json.load(f)

        # Handle different trajectory formats
        if "messages" in data:
            # Standard format
            messages = data["messages"]
            reward = data.get("reward", 0.0)
        elif "trajectory" in data:
            # ATIF format
            trajectory = data["trajectory"]
            messages = []
            for step in trajectory:
                if "observation" in step:
                    messages.append({"role": "user", "content": step["observation"]})
                if "action" in step:
                    messages.append({"role": "assistant", "content": step["action"]})
            reward = data.get("reward", data.get("metrics", {}).get("reward", 0.0))
        else:
            logger.warning(f"Unknown trajectory format in {trajectory_file}")
            return None

        # Convert to tokens
        if self.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Manual concatenation
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n".join(parts)
            tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Extract prompt and response
        # For multi-turn, prompt is first user message, response is everything after
        prompt_messages = []
        response_messages = []
        found_first_assistant = False

        for msg in messages:
            if msg["role"] == "assistant" and not found_first_assistant:
                found_first_assistant = True
            if found_first_assistant:
                response_messages.append(msg)
            else:
                prompt_messages.append(msg)

        # Tokenize prompt to get prompt length
        if prompt_messages and self.chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_length = len(prompt_tokens)
        else:
            prompt_length = 0

        response_length = len(tokens) - prompt_length

        # Build response text
        response_text = ""
        for msg in response_messages:
            response_text += msg.get("content", "") + "\n"
        response_text = response_text.strip()

        # Create Sample
        sample = Sample(
            prompt=prompt_messages[0]["content"] if prompt_messages else "",
            tokens=tokens,
            response=response_text,
            response_length=response_length,
            reward=reward,
            status=Sample.Status.COMPLETED,
            metadata={
                "source": "harbor",
                "trajectory_file": str(trajectory_file),
                "n_turns": len(messages),
            },
        )

        return sample

    def load_traces_parquet(self, parquet_file: str | Path) -> list["Sample"]:
        """
        Load trajectories from Harbor's exported Parquet traces file.

        Args:
            parquet_file: Path to traces.parquet file

        Returns:
            List of SLiME Sample objects
        """
        from slime.utils.types import Sample

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas pyarrow")

        df = pd.read_parquet(parquet_file)
        samples = []

        for _, row in df.iterrows():
            # Extract messages from ShareGPT or messages column
            if "messages" in row:
                messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
            elif "sharegpt" in row:
                messages = json.loads(row["sharegpt"]) if isinstance(row["sharegpt"], str) else row["sharegpt"]
            else:
                continue

            reward = row.get("reward", 0.0)

            # Tokenize
            if self.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                tokens = self.tokenizer.encode(text, add_special_tokens=True)

            # Response text (all assistant turns)
            response_text = "\n".join([
                m["content"] for m in messages if m["role"] == "assistant"
            ])

            sample = Sample(
                prompt=messages[0]["content"] if messages else "",
                tokens=tokens,
                response=response_text,
                response_length=len(tokens),  # Simplified; could compute exact
                reward=reward,
                status=Sample.Status.COMPLETED,
                metadata={"source": "harbor_parquet"},
            )
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {parquet_file}")
        return samples


def convert_harbor_to_slime(
    job_dir: str,
    tokenizer_name: str = "Qwen/Qwen3-Coder-30B-A3B",
    output_file: str = None,
) -> list:
    """
    Convenience function to convert Harbor job output to SLiME samples.

    Args:
        job_dir: Path to Harbor job directory
        tokenizer_name: HuggingFace tokenizer name
        output_file: Optional path to save samples as JSON

    Returns:
        List of Sample objects
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    converter = HarborTrajectoryConverter(tokenizer)
    samples = converter.load_job(job_dir)

    if output_file:
        with open(output_file, "w") as f:
            json.dump([s.to_dict() for s in samples], f, indent=2)
        logger.info(f"Saved {len(samples)} samples to {output_file}")

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Harbor trajectories to SLiME format")
    parser.add_argument("job_dir", help="Harbor job directory")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Coder-30B-A3B", help="Tokenizer name")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    samples = convert_harbor_to_slime(args.job_dir, args.tokenizer, args.output)

    print(f"Converted {len(samples)} trajectories")
    resolved = sum(1 for s in samples if s.reward > 0)
    print(f"Resolved: {resolved}/{len(samples)} ({100*resolved/len(samples):.1f}%)")
