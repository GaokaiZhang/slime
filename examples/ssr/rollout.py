"""
SSR Rollout - Custom generation function for SLiME.

Implements multi-turn agent interaction with docker sandboxes for:
- Bug Injector: Interacts with repo to create bug artifacts
- Bug Solver: Fixes bugs given oracle test patches

Based on SSR paper (arXiv:2512.18552).
"""

import asyncio
import json
import logging
import re
from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .bug_artifact import BugArtifact, validate_bug_artifact
from .docker_sandbox import AsyncDockerSandbox, DockerSandboxConfig
from .prompts import format_injector_prompt, format_solver_prompt
from .rewards import compute_injector_reward, compute_solver_reward

logger = logging.getLogger(__name__)

# Tool patterns for parsing agent output
TOOL_PATTERNS = {
    "bash": re.compile(r"<tool:\s*bash>(.*?)</tool>", re.DOTALL),
    "submit": re.compile(r"<tool:\s*submit>(.*?)</tool>", re.DOTALL),
    "write": re.compile(r"<tool:\s*write\s+([^>]+)>(.*?)</tool>", re.DOTALL),
    "read": re.compile(r"<tool:\s*read\s+([^>]+)>(.*?)</tool>", re.DOTALL),
}


# SSR Configuration
SSR_CONFIGS = {
    "max_turns": 32,  # Maximum agent turns
    "test_timeout": 90,  # Test execution timeout
    "injector_timeout": 300,  # Bug injector timeout
    "solver_timeout": 300,  # Bug solver timeout
    "min_passing_tests": 5,
    "min_changed_files": 1,
    "min_num_tests_to_break": 1,
}


def parse_tool_call(response: str) -> tuple[str | None, str, dict[str, str]]:
    """
    Parse tool call from agent response.

    Returns:
        Tuple of (tool_name, tool_args, tool_kwargs)
        Returns (None, "", {}) if no tool call found
    """
    for tool_name, pattern in TOOL_PATTERNS.items():
        match = pattern.search(response)
        if match:
            if tool_name in ["write", "read"]:
                return tool_name, match.group(1).strip(), {"content": match.group(2)}
            else:
                return tool_name, match.group(1).strip(), {}

    return None, "", {}


def format_tool_result(tool_name: str, result: str, error: str | None = None) -> str:
    """Format tool execution result for agent."""
    if error:
        return f"\n<tool_result: {tool_name}>\nError: {error}\n</tool_result>\n"
    return f"\n<tool_result: {tool_name}>\n{result}\n</tool_result>\n"


async def execute_tool(
    sandbox: AsyncDockerSandbox,
    tool_name: str,
    tool_args: str,
    tool_kwargs: dict[str, str],
) -> tuple[str, bool]:
    """
    Execute a tool in the sandbox.

    Returns:
        Tuple of (result_string, is_done)
    """
    is_done = False

    try:
        if tool_name == "bash":
            # Execute bash command
            result = await sandbox.exec_command(tool_args, timeout=60)
            return format_tool_result("bash", result), False

        elif tool_name == "submit":
            # Submission - signals end of interaction
            is_done = True
            return format_tool_result("submit", "Submission received."), True

        elif tool_name == "write":
            # Write file
            file_path = tool_args
            content = tool_kwargs.get("content", "")
            await asyncio.to_thread(sandbox._sandbox.write_file, file_path, content)
            return format_tool_result("write", f"File written: {file_path}"), False

        elif tool_name == "read":
            # Read file
            file_path = tool_args
            content = await asyncio.to_thread(sandbox._sandbox.read_file, file_path)
            return format_tool_result("read", content), False

        else:
            return format_tool_result(tool_name, "", f"Unknown tool: {tool_name}"), False

    except Exception as e:
        logger.warning(f"Tool execution failed: {tool_name} - {e}")
        return format_tool_result(tool_name, "", str(e)), False


async def generate_injector(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Generate bug artifact using Bug Injector agent.

    The injector interacts with a docker sandbox to:
    1. Explore the codebase (with GOLD PATCH already applied)
    2. Create test infrastructure
    3. Inject NEW bugs into the CLEAN code
    4. Weaken tests to hide bugs
    5. Submit bug artifact

    IMPORTANT: The Docker container starts with the ORIGINAL SWE-bench bug.
    We MUST apply the gold_patch FIRST to get clean code before bug injection.
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get instance info from sample
    instance_id = sample.metadata.get("instance_id", "")
    repo_path = sample.metadata.get("repo_path", "/testbed")
    injector_type = sample.metadata.get("injector_type", "removal")

    # CRITICAL: Get gold patch to fix original bug first
    gold_patch = sample.metadata.get("gold_patch", "")
    if not gold_patch:
        logger.warning(f"No gold_patch provided for {instance_id} - original bug will remain!")

    # Format prompt
    system_prompt = format_injector_prompt(
        prompt_type=injector_type,
        repo_root=repo_path,
        min_passing_tests=SSR_CONFIGS["min_passing_tests"],
        min_changed_files=SSR_CONFIGS["min_changed_files"],
        min_num_tests_to_break=SSR_CONFIGS["min_num_tests_to_break"],
    )

    # Initialize conversation
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    response = ""
    response_token_ids = []
    loss_masks = []
    turn = 0

    # Start docker sandbox with gold_patch applied
    sandbox_config = DockerSandboxConfig()
    sandbox = AsyncDockerSandbox(sandbox_config)

    try:
        # Start container and apply gold_patch to fix original bug
        # This gives us a CLEAN codebase for bug injection
        await sandbox.start(instance_id=instance_id, gold_patch=gold_patch)

        while turn < SSR_CONFIGS["max_turns"]:
            turn += 1

            # Check total length
            total_length = len(prompt_token_ids) + len(response_token_ids)
            max_context = getattr(args, "rollout_max_context_len", 32768)
            if total_length >= max_context:
                sample.status = Sample.Status.TRUNCATED
                break

            # Generate next response chunk
            current_token_ids = prompt_token_ids + response_token_ids
            payload = {
                "input_ids": current_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)

            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            # Extract response tokens and log probs
            if "output_token_logprobs" in output["meta_info"]:
                cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                cur_response = state.tokenizer.decode(cur_token_ids)

                if sample.rollout_log_probs is None:
                    sample.rollout_log_probs = []
                sample.rollout_log_probs += cur_log_probs
            else:
                cur_response = output["text"]
                cur_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

            response += cur_response
            response_token_ids += cur_token_ids
            loss_masks += [1] * len(cur_token_ids)

            # Check for tool call
            tool_name, tool_args, tool_kwargs = parse_tool_call(cur_response)

            if tool_name is None:
                # No tool call, check if length limit reached
                if output["meta_info"]["finish_reason"]["type"] == "length":
                    break
                continue

            # Execute tool
            tool_result, is_done = await execute_tool(sandbox, tool_name, tool_args, tool_kwargs)

            if is_done:
                # Parse and validate submission
                artifact = BugArtifact.from_submission(response, repo_path)
                validation_result = validate_bug_artifact(
                    artifact,
                    sandbox._sandbox,
                    min_passing_tests=SSR_CONFIGS["min_passing_tests"],
                    min_changed_files=SSR_CONFIGS["min_changed_files"],
                    min_failing_tests=SSR_CONFIGS["min_num_tests_to_break"],
                )

                sample.metadata["validation_passed"] = validation_result.valid
                sample.metadata["validation_errors"] = validation_result.errors
                sample.metadata["bug_artifact"] = artifact.to_dict()
                sample.status = Sample.Status.COMPLETED
                break

            # Add tool result to response
            result_token_ids = state.tokenizer(tool_result, add_special_tokens=False)["input_ids"]
            response += tool_result
            response_token_ids += result_token_ids
            loss_masks += [0] * len(result_token_ids)  # Don't train on tool results

            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(result_token_ids)

    finally:
        await sandbox.stop()

    # Set sample attributes
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Compute reward
    validation_passed = sample.metadata.get("validation_passed", False)
    sample.reward = compute_injector_reward(validation_passed)

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    return sample


async def generate_solver(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Generate bug fix using Bug Solver agent.

    The solver interacts with a buggy docker sandbox to:
    1. Understand the failing tests
    2. Explore and debug the code
    3. Implement a fix
    4. Submit the fix patch
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get bug artifact from sample metadata
    bug_artifact_data = sample.metadata.get("bug_artifact", {})
    bug_artifact = BugArtifact.from_dict(bug_artifact_data)

    instance_id = sample.metadata.get("instance_id", "")
    repo_path = sample.metadata.get("repo_path", "/testbed")

    # Format prompt with oracle test patch
    oracle_test_patch = bug_artifact.get_oracle_test_patch()
    system_prompt = format_solver_prompt(oracle_test_patch, repo_path)

    # Initialize conversation
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    response = ""
    response_token_ids = []
    loss_masks = []
    turn = 0

    # Start docker sandbox
    sandbox_config = DockerSandboxConfig()
    sandbox = AsyncDockerSandbox(sandbox_config)

    try:
        await sandbox.start(instance_id=instance_id)

        # Apply bug patch and test patch to create buggy state
        await sandbox.apply_patch(bug_artifact.bug_patch)
        await sandbox.apply_patch(bug_artifact.test_patch)

        # CRITICAL: Wipe git history to prevent information leakage
        # Solver must NOT see commit history (could reveal bug via git log/diff)
        await sandbox.wipe_git_history()

        while turn < SSR_CONFIGS["max_turns"]:
            turn += 1

            # Check total length
            total_length = len(prompt_token_ids) + len(response_token_ids)
            max_context = getattr(args, "rollout_max_context_len", 32768)
            if total_length >= max_context:
                sample.status = Sample.Status.TRUNCATED
                break

            # Generate next response chunk
            current_token_ids = prompt_token_ids + response_token_ids
            payload = {
                "input_ids": current_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)

            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            # Extract response tokens and log probs
            if "output_token_logprobs" in output["meta_info"]:
                cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                cur_response = state.tokenizer.decode(cur_token_ids)

                if sample.rollout_log_probs is None:
                    sample.rollout_log_probs = []
                sample.rollout_log_probs += cur_log_probs
            else:
                cur_response = output["text"]
                cur_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

            response += cur_response
            response_token_ids += cur_token_ids
            loss_masks += [1] * len(cur_token_ids)

            # Check for tool call
            tool_name, tool_args, tool_kwargs = parse_tool_call(cur_response)

            if tool_name is None:
                if output["meta_info"]["finish_reason"]["type"] == "length":
                    break
                continue

            # Execute tool
            tool_result, is_done = await execute_tool(sandbox, tool_name, tool_args, tool_kwargs)

            if is_done and tool_name == "submit":
                # Parse submitted patch and verify tests pass
                patch_path = tool_args.strip().split("\n")[0]
                solver_patch = await asyncio.to_thread(sandbox._sandbox.read_file, patch_path)

                # Reset and apply solver's patch
                await sandbox.reset()
                await sandbox.apply_patch(bug_artifact.bug_patch)
                await sandbox.apply_patch(bug_artifact.test_patch)
                await sandbox.apply_patch(solver_patch)

                # Run oracle tests (with test_patch reverted)
                await sandbox.apply_patch(bug_artifact.test_patch, reverse=True)

                all_pass, test_results = await sandbox.check_tests_pass(
                    bug_artifact.test_script,
                    bug_artifact.parse_test_output,
                )

                sample.metadata["all_tests_pass"] = all_pass
                sample.metadata["test_results"] = test_results
                sample.metadata["solver_patch"] = solver_patch
                sample.status = Sample.Status.COMPLETED
                break

            # Add tool result to response
            result_token_ids = state.tokenizer(tool_result, add_special_tokens=False)["input_ids"]
            response += tool_result
            response_token_ids += result_token_ids
            loss_masks += [0] * len(result_token_ids)

            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(result_token_ids)

    finally:
        await sandbox.stop()

    # Set sample attributes
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Compute reward
    all_tests_pass = sample.metadata.get("all_tests_pass", False)
    sample.reward = compute_solver_reward(all_tests_pass)

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    return sample


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """
    Main generate function for SSR rollout.

    Routes to injector or solver based on sample metadata.
    """
    agent_type = sample.metadata.get("agent_type", "injector")

    if agent_type == "injector":
        return await generate_injector(args, sample, sampling_params)
    elif agent_type == "solver":
        return await generate_solver(args, sample, sampling_params)
    else:
        logger.error(f"Unknown agent type: {agent_type}")
        sample.status = Sample.Status.FAILED
        sample.reward = -1.0
        return sample


# Reward function for slime rm_hub
async def reward_func(args, sample, **kwargs):
    """
    Compute reward for SSR sample.

    Works for both injector and solver based on metadata.
    """
    agent_type = sample.metadata.get("agent_type", "solver")

    if agent_type == "injector":
        validation_passed = sample.metadata.get("validation_passed", False)
        solver_results = sample.metadata.get("solver_results", None)

        if solver_results:
            successful = sum(1 for r in solver_results if r)
            solve_rate = successful / len(solver_results)
            return compute_injector_reward(validation_passed, solve_rate)
        else:
            return compute_injector_reward(validation_passed)

    else:  # solver
        all_tests_pass = sample.metadata.get("all_tests_pass", False)
        return compute_solver_reward(all_tests_pass)
