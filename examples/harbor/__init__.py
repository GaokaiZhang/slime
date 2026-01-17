"""
Harbor + SLiME Integration

This module provides integration between Harbor (agent evaluation framework)
and SLiME (RL training framework) for training coding agents.

Key insight: Harbor handles trajectory generation, SLiME handles training.
Log probs are recomputed at training time, so Harbor only needs to provide:
- tokens (token IDs)
- response (text)
- reward (from verification)
"""
