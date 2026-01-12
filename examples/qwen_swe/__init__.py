"""
Qwen-Code Agent Integration with SLiME GRPO for SWE-bench Tasks.

This module provides the integration between qwen-code agent framework
and SLiME's GRPO training for SWE-bench bug-solving tasks.

Components:
- rollout.py: Custom generate function for agent rollouts
- rewards.py: SWE-bench evaluation-based reward function
- data_source.py: Django instance loader
- qwen_agent.py: Qwen-code agent wrapper
- prompts.py: Bug solving prompt templates
"""

from .rollout import generate
from .rewards import swebench_reward

__all__ = ["generate", "swebench_reward"]
