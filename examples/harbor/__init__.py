"""
Harbor + SLiME Integration for GRPO Training on SWE-bench.

Architecture:
- SLiME: GRPO training (loss computation, model updates, Megatron)
- Harbor: Agent rollouts (terminus-2), Docker environments, verification

This module provides a thin integration layer:
- rollout.py: Uses Harbor's Trial API with terminus-2 agent
- data_source.py: SLiME DataSource for SWE-Bench_Verified

Custom code is minimal - only loss mask computation from Harbor's RolloutDetail.
"""

from .data_source import (
    SWEBenchVerifiedDataSource,
    DjangoTrainDataSource,
    create_data_source,
)
from .rollout import (
    generate,
    generate_group,
    compute_loss_mask_from_rollout_details,
)

__all__ = [
    "SWEBenchVerifiedDataSource",
    "DjangoTrainDataSource",
    "create_data_source",
    "generate",
    "generate_group",
    "compute_loss_mask_from_rollout_details",
]
