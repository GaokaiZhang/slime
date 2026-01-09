"""
SSR Training Script for SLiME.

Runs Self-play SWE-RL training with Bug Injector and Bug Solver agents.

Usage:
    python examples/ssr/run_ssr.py --config examples/ssr/ssr_config.yaml

    # Or with command line arguments
    python examples/ssr/run_ssr.py \
        --hf_checkpoint Kwai-Klear/Klear-AgentForge-8B-SFT \
        --ssr_data_path /home/gaokaizhang/SWE-sft/data/sft/train.jsonl \
        --ssr_agent_type both \
        --rollout_batch_size 16 \
        --n_samples_per_prompt 8
"""

import argparse
import logging
import os
import sys

# Add slime to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SSR Training with SLiME")

    # Model configuration
    parser.add_argument(
        "--hf_checkpoint",
        type=str,
        default="Kwai-Klear/Klear-AgentForge-8B-SFT",
        help="HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="HF_TOKEN_PLACEHOLDER",
        help="HuggingFace token",
    )

    # SSR configuration
    parser.add_argument(
        "--ssr_data_path",
        type=str,
        default="/home/gaokaizhang/SWE-sft/data/sft/train.jsonl",
        help="Path to SWE-bench data",
    )
    parser.add_argument(
        "--ssr_agent_type",
        type=str,
        choices=["injector", "solver", "both"],
        default="both",
        help="Agent type to train",
    )
    parser.add_argument(
        "--ssr_injector_type",
        type=str,
        choices=["removal", "history", "direct"],
        default="removal",
        help="Bug injection strategy",
    )
    parser.add_argument(
        "--ssr_group_size",
        type=int,
        default=8,
        help="Number of solver attempts per bug artifact",
    )
    parser.add_argument(
        "--ssr_alpha",
        type=float,
        default=0.8,
        help="Penalty for extreme solve rates",
    )

    # Rollout configuration
    parser.add_argument("--rollout_batch_size", type=int, default=16)
    parser.add_argument("--n_samples_per_prompt", type=int, default=8)
    parser.add_argument("--rollout_max_response_len", type=int, default=16384)
    parser.add_argument("--rollout_max_context_len", type=int, default=32768)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_top_p", type=float, default=0.95)
    parser.add_argument("--rollout_top_k", type=int, default=-1)

    # Training configuration
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--advantage_estimator", type=str, default="grpo")

    # Infrastructure
    parser.add_argument("--rollout_num_gpus", type=int, default=1)
    parser.add_argument("--rollout_num_gpus_per_engine", type=int, default=1)
    parser.add_argument("--num_gpus_per_node", type=int, default=1)
    parser.add_argument("--sglang_server_concurrency", type=int, default=32)

    # Paths
    parser.add_argument("--output_dir", type=str, default="./ssr_output")
    parser.add_argument("--save_debug_rollout_data", type=str, default=None)

    # Test mode
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with minimal config")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def setup_ssr_args(args):
    """Set up additional args required by slime."""

    # Required paths for slime
    args.data_source_path = "examples.ssr.data_source:create_data_source"
    args.rollout_function_path = "examples.ssr.rollout:generate"
    args.eval_function_path = "slime.rollout.sglang_rollout:eval_rollout"
    args.custom_generate_function_path = "examples.ssr.rollout:generate"

    # Reward configuration
    args.custom_reward_post_process_path = None
    args.reward_key = None
    args.group_rm = True

    # Generation settings
    args.rollout_stop = ["</tool>", "<|im_end|>"]
    args.rollout_stop_token_ids = None
    args.rollout_skip_special_tokens = False
    args.apply_chat_template = False
    args.apply_chat_template_kwargs = {}

    # Router settings
    args.sglang_router_ip = None
    args.sglang_router_port = None
    args.use_slime_router = False

    # Other required args
    args.rollout_global_dataset = True
    args.ci_test = False
    args.partial_rollout = False
    args.mask_offpolicy_in_partial_rollout = False
    args.dynamic_sampling_filter_path = None
    args.over_sampling_batch_size = args.rollout_batch_size
    args.disable_rollout_trim_samples = False
    args.rollout_sample_filter_path = None
    args.rollout_all_samples_process_path = None
    args.sglang_enable_deterministic_inference = False
    args.rollout_seed = 42
    args.sglang_dp_size = None
    args.sglang_speculative_algorithm = None
    args.use_rollout_routing_replay = False
    args.multimodal_keys = None
    args.eval_datasets = []

    return args


def test_docker_sandbox():
    """Test docker sandbox functionality."""
    from examples.ssr.docker_sandbox import DockerSandbox, DockerSandboxConfig

    logger.info("Testing docker sandbox...")

    config = DockerSandboxConfig(
        image_name="swebench/sweb.eval.x86_64.django_1776_django-17087",
        image_tag="latest",
    )

    try:
        with DockerSandbox(config) as sandbox:
            # Test basic commands
            result = sandbox.exec_command("pwd")
            logger.info(f"PWD: {result.strip()}")

            result = sandbox.exec_command("ls -la")
            logger.info(f"LS output: {result[:200]}...")

            result = sandbox.exec_command("git status")
            logger.info(f"Git status: {result[:200]}...")

        logger.info("Docker sandbox test passed!")
        return True
    except Exception as e:
        logger.error(f"Docker sandbox test failed: {e}")
        return False


def test_bug_artifact():
    """Test bug artifact creation and validation."""
    from examples.ssr.bug_artifact import BugArtifact, validate_bug_artifact

    logger.info("Testing bug artifact...")

    # Create a minimal test artifact
    artifact = BugArtifact(
        test_files=["tests/test_example.py"],
        test_script="#!/bin/bash\npytest tests/test_example.py -v",
        parse_test_output='''import sys, json
input_text = sys.stdin.read()
results = {}
for line in input_text.split("\\n"):
    if "PASSED" in line:
        test_name = line.split(" ")[0]
        results[test_name] = "passed"
    elif "FAILED" in line:
        test_name = line.split(" ")[0]
        results[test_name] = "failed"
print(json.dumps(results, indent=2))
''',
        bug_patch="diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-return x + 1\n+return x",
        test_patch="diff --git a/tests/test_example.py b/tests/test_example.py\n--- a/tests/test_example.py\n+++ b/tests/test_example.py\n@@ -1 +1 @@\n-assert func(1) == 2\n+assert func(1) >= 1",
    )

    logger.info(f"Created artifact with {len(artifact.test_files)} test files")
    logger.info(f"Code files touched: {artifact.get_code_files_touched()}")
    logger.info(f"Test files touched: {artifact.get_test_files_touched()}")

    # Test oracle patch generation
    oracle = artifact.get_oracle_test_patch()
    logger.info(f"Oracle test patch: {oracle[:200]}...")

    logger.info("Bug artifact test passed!")
    return True


def test_prompts():
    """Test prompt formatting."""
    from examples.ssr.prompts import format_injector_prompt, format_solver_prompt

    logger.info("Testing prompts...")

    # Test injector prompt
    injector_prompt = format_injector_prompt(
        prompt_type="removal",
        repo_root="/testbed",
        min_passing_tests=5,
        min_changed_files=1,
    )
    logger.info(f"Injector prompt length: {len(injector_prompt)}")
    assert "{min_passing_tests}" not in injector_prompt
    assert "/testbed" in injector_prompt

    # Test solver prompt
    oracle_patch = "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new"
    solver_prompt = format_solver_prompt(oracle_patch, "/testbed")
    logger.info(f"Solver prompt length: {len(solver_prompt)}")
    assert oracle_patch in solver_prompt

    logger.info("Prompts test passed!")
    return True


def test_rewards():
    """Test reward functions."""
    from examples.ssr.rewards import (
        compute_solver_reward,
        compute_injector_reward,
        SSRRewardModel,
    )

    logger.info("Testing rewards...")

    # Solver rewards
    assert compute_solver_reward(True) == 1.0
    assert compute_solver_reward(False) == -1.0

    # Injector rewards
    assert compute_injector_reward(False) == -1.0  # Validation failed
    assert compute_injector_reward(True, 0.0) == -0.8  # Too hard
    assert compute_injector_reward(True, 1.0) == -0.8  # Too easy
    assert abs(compute_injector_reward(True, 0.5) - 0.1) < 0.001  # 1 - 1.8 * 0.5 = 0.1

    # Reward model
    model = SSRRewardModel()
    result = model.solver_reward(True)
    assert result["score"] == 1.0
    assert result["all_tests_pass"] == True

    result = model.injector_reward(True, [True, True, False, False])
    assert result["solve_rate"] == 0.5

    logger.info("Rewards test passed!")
    return True


def run_tests(args):
    """Run all tests."""
    logger.info("Running SSR tests...")

    tests = [
        ("Prompts", test_prompts),
        ("Rewards", test_rewards),
        ("Bug Artifact", test_bug_artifact),
    ]

    if not args.test_mode:
        tests.append(("Docker Sandbox", test_docker_sandbox))

    passed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                logger.info(f"[PASS] {name}")
                passed += 1
            else:
                logger.error(f"[FAIL] {name}")
        except Exception as e:
            logger.error(f"[FAIL] {name}: {e}")

    logger.info(f"Tests passed: {passed}/{len(tests)}")
    return passed == len(tests)


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set HF token
    os.environ["HF_TOKEN"] = args.hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    if args.test_mode:
        # Run tests only
        success = run_tests(args)
        sys.exit(0 if success else 1)

    # Setup full args
    args = setup_ssr_args(args)

    logger.info("=" * 60)
    logger.info("SSR Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.hf_checkpoint}")
    logger.info(f"Data: {args.ssr_data_path}")
    logger.info(f"Agent type: {args.ssr_agent_type}")
    logger.info(f"Batch size: {args.rollout_batch_size}")
    logger.info(f"Samples per prompt: {args.n_samples_per_prompt}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # Import slime components
    try:
        from slime.ray.rollout import RolloutManager
        from slime.ray.train_actor import TrainActor

        logger.info("SLiME components imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import SLiME: {e}")
        logger.info("Running in standalone test mode instead...")
        run_tests(args)
        return

    # Run training
    logger.info("Starting SSR training...")
    # Full training would be orchestrated by slime's main training loop
    # For now, we can run a test rollout

    from examples.ssr.data_source import create_data_source

    data_source = create_data_source(args)
    logger.info(f"Data source created with {len(data_source)} instances")

    # Get a test batch
    batch = data_source.get_samples(2)
    logger.info(f"Got {len(batch)} sample groups")

    for i, group in enumerate(batch):
        logger.info(f"Group {i}: {len(group)} samples")
        for j, sample in enumerate(group):
            logger.info(f"  Sample {j}: {sample.metadata.get('instance_id')}, type={sample.metadata.get('agent_type')}")


if __name__ == "__main__":
    main()
