# Self-play SWE-RL (SSR) Implementation for SLiME
# Based on arXiv:2512.18552

"""
SSR (Self-play SWE-RL) Module

Core SSR idea:
- One policy model, instantiated into two roles via prompting:
  (1) Bug Injector (proposer) and (2) Bug Solver (repair agent).
- Both roles share weights and are jointly updated with RL.
- Bugs are specified by a *bug artifact* (patches + test runner + parser), not natural-language issues.

This module provides:
- Bug Injector agent: Injects bugs into code and creates bug artifacts
- Bug Solver agent: Fixes bugs given buggy repo and oracle tests
- Bug artifact validation
- Reward functions for SSR training
- Docker-based sandbox execution

Usage:
    from examples.ssr import (
        BugArtifact,
        validate_bug_artifact,
        compute_solver_reward,
        compute_injector_reward,
        DockerSandbox,
    )

    # Or use the rollout function with slime
    # --rollout_function_path examples.ssr.rollout:generate
"""

__all__ = [
    # Core components
    "BugArtifact",
    "validate_bug_artifact",
    "compute_solver_reward",
    "compute_injector_reward",
    "DockerSandbox",
    "ssr_generate",
    # Prompts
    "BUG_INJECTOR_REMOVAL_PROMPT",
    "BUG_INJECTOR_HISTORY_PROMPT",
    "BUG_INJECTOR_DIRECT_PROMPT",
    "BUG_SOLVER_PROMPT",
    # GPU inference
    "LocalGPUEngine",
    "SGLangEngine",
    "create_inference_engine",
    "InferenceConfig",
    # SWE-bench harness
    "evaluate_patch",
    "evaluate_solver_patch",
    "evaluate_bug_artifact",
    "run_tests_in_container",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ("BUG_INJECTOR_REMOVAL_PROMPT", "BUG_INJECTOR_HISTORY_PROMPT",
                "BUG_INJECTOR_DIRECT_PROMPT", "BUG_SOLVER_PROMPT"):
        from .prompts import (
            BUG_INJECTOR_REMOVAL_PROMPT,
            BUG_INJECTOR_HISTORY_PROMPT,
            BUG_INJECTOR_DIRECT_PROMPT,
            BUG_SOLVER_PROMPT,
        )
        return locals()[name]

    if name in ("BugArtifact", "validate_bug_artifact"):
        from .bug_artifact import BugArtifact, validate_bug_artifact
        return locals()[name]

    if name in ("compute_solver_reward", "compute_injector_reward"):
        from .rewards import compute_solver_reward, compute_injector_reward
        return locals()[name]

    if name == "DockerSandbox":
        from .docker_sandbox import DockerSandbox
        return DockerSandbox

    if name == "ssr_generate":
        from .rollout import generate
        return generate

    # GPU inference
    if name in ("LocalGPUEngine", "SGLangEngine", "create_inference_engine", "InferenceConfig"):
        from .gpu_inference import (
            LocalGPUEngine,
            SGLangEngine,
            create_inference_engine,
            InferenceConfig,
        )
        return locals()[name]

    # SWE-bench harness
    if name in ("evaluate_patch", "evaluate_solver_patch", "evaluate_bug_artifact", "run_tests_in_container"):
        from .swebench_harness import (
            evaluate_patch,
            evaluate_solver_patch,
            evaluate_bug_artifact,
            run_tests_in_container,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
