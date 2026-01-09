"""
SSR Reward Functions for Bug Injector and Bug Solver agents.

Based on SSR paper (arXiv:2512.18552) Section on Rewards.
"""

from typing import Any

# Default SSR hyperparameters
DEFAULT_ALPHA = 0.8
DEFAULT_GROUP_SIZE = 8


def compute_solver_reward(
    all_tests_pass: bool,
    **kwargs,
) -> float:
    """
    Compute reward for the Bug Solver agent.

    Solver Reward (terminal, per attempt):
    - +1 if all oracle tests pass
    - -1 otherwise

    Args:
        all_tests_pass: Whether all oracle tests pass after applying solver's patch

    Returns:
        Reward value (+1 or -1)
    """
    return 1.0 if all_tests_pass else -1.0


def compute_injector_reward(
    validation_passed: bool,
    solve_rate: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    **kwargs,
) -> float:
    """
    Compute reward for the Bug Injector agent.

    Injector Reward (per bug artifact):
    Let s be the solve rate = (#successful solver attempts) / (group_size)

    - -1.0 if consistency validation fails
    - -alpha if s == 0 or s == 1
    - 1 - (1 + alpha) * s if 0 < s < 1

    The reward function encourages:
    - Valid bug artifacts (validation must pass)
    - Moderate difficulty bugs (solve_rate between 0 and 1)
    - Discourages too-easy bugs (solve_rate = 1) and too-hard bugs (solve_rate = 0)

    Args:
        validation_passed: Whether bug artifact passed consistency validation
        solve_rate: Fraction of solver attempts that succeeded (0 to 1)
        alpha: Penalty for extreme solve rates (default 0.8)

    Returns:
        Reward value
    """
    if not validation_passed:
        return -1.0

    if solve_rate is None:
        # If no solve attempts yet, return 0 (neutral)
        return 0.0

    if solve_rate == 0.0 or solve_rate == 1.0:
        return -alpha

    return 1.0 - (1.0 + alpha) * solve_rate


def compute_group_injector_reward(
    validation_passed: bool,
    solver_results: list[bool],
    alpha: float = DEFAULT_ALPHA,
    **kwargs,
) -> float:
    """
    Compute injector reward given a group of solver attempts.

    Args:
        validation_passed: Whether bug artifact passed consistency validation
        solver_results: List of booleans indicating solver success for each attempt
        alpha: Penalty for extreme solve rates

    Returns:
        Reward value
    """
    if not solver_results:
        return compute_injector_reward(validation_passed, None, alpha)

    successful = sum(1 for r in solver_results if r)
    solve_rate = successful / len(solver_results)

    return compute_injector_reward(validation_passed, solve_rate, alpha)


class SSRRewardModel:
    """
    Complete reward model for SSR training.

    Handles both injector and solver rewards, with support for:
    - Group-based reward computation
    - Reward normalization (optional)
    - Detailed reward breakdown for logging
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        group_size: int = DEFAULT_GROUP_SIZE,
        normalize_rewards: bool = False,
    ):
        self.alpha = alpha
        self.group_size = group_size
        self.normalize_rewards = normalize_rewards

        # Track statistics for normalization
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0
        self._count = 0

    def solver_reward(
        self,
        all_tests_pass: bool,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Compute solver reward with metadata.

        Returns dict with:
        - score: The reward value
        - all_tests_pass: Whether all tests passed
        """
        score = compute_solver_reward(all_tests_pass)
        return {
            "score": score,
            "all_tests_pass": all_tests_pass,
        }

    def injector_reward(
        self,
        validation_passed: bool,
        solver_results: list[bool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Compute injector reward with metadata.

        Returns dict with:
        - score: The reward value
        - validation_passed: Whether validation passed
        - solve_rate: The solve rate (if solver results provided)
        - num_solvers: Number of solver attempts
        - num_successful: Number of successful solver attempts
        """
        if solver_results is None:
            solve_rate = None
            num_solvers = 0
            num_successful = 0
        else:
            num_solvers = len(solver_results)
            num_successful = sum(1 for r in solver_results if r)
            solve_rate = num_successful / num_solvers if num_solvers > 0 else None

        score = compute_injector_reward(validation_passed, solve_rate, self.alpha)

        return {
            "score": score,
            "validation_passed": validation_passed,
            "solve_rate": solve_rate,
            "num_solvers": num_solvers,
            "num_successful": num_successful,
        }

    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        if not self.normalize_rewards:
            return reward

        # Update statistics
        self._reward_sum += reward
        self._reward_sq_sum += reward ** 2
        self._count += 1

        if self._count < 2:
            return reward

        mean = self._reward_sum / self._count
        var = (self._reward_sq_sum / self._count) - (mean ** 2)
        std = max(var ** 0.5, 1e-6)

        return (reward - mean) / std


# Async wrapper for use with slime rollout
async def async_solver_reward(args, sample, **kwargs):
    """
    Async reward function for solver in slime rollout.

    Expected sample.metadata:
    - all_tests_pass: bool
    """
    all_tests_pass = sample.metadata.get("all_tests_pass", False)
    return compute_solver_reward(all_tests_pass)


async def async_injector_reward(args, sample, **kwargs):
    """
    Async reward function for injector in slime rollout.

    Expected sample.metadata:
    - validation_passed: bool
    - solver_results: list[bool] (optional)
    """
    validation_passed = sample.metadata.get("validation_passed", False)
    solver_results = sample.metadata.get("solver_results", None)

    if solver_results:
        return compute_group_injector_reward(validation_passed, solver_results)
    else:
        return compute_injector_reward(validation_passed)
