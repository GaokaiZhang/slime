#!/usr/bin/env python3
"""
SWE-bench GRPO Training Script for SLiME.

Runs GRPO training with qwen-code agent on SWE-bench instances.

Usage:
    # With local vLLM:
    python examples/qwen_swe/run_qwen_swe.py \
        --vllm_url http://localhost:8000 \
        --hf_checkpoint Qwen/Qwen3-Coder-30B-A3B-Instruct

    # With config file:
    python examples/qwen_swe/run_qwen_swe.py --config examples/qwen_swe/config.yaml
"""

import argparse
import logging
import os
import sys

# Add slime to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from slime.utils.arguments import parse_args as slime_parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_qwen_swe_args(parser):
    """Add SWE-bench specific arguments."""
    group = parser.add_argument_group("SWE-bench Configuration")
    
    # Data configuration
    group.add_argument(
        "--swe_instance_file",
        type=str,
        default="/home/gaokaizhang/SWE-sft/data/raw/splits/train_201_django.txt",
        help="Path to file with SWE-bench instance IDs",
    )
    group.add_argument(
        "--swe_split",
        type=str,
        default=None,
        help="SWE-bench split type (all, part_a, part_b)",
    )
    
    # vLLM configuration
    group.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL for inference",
    )
    
    # Qwen agent configuration
    group.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="Model name for qwen-code CLI",
    )
    group.add_argument(
        "--qwen_max_turns",
        type=int,
        default=50,
        help="Maximum turns per agent session",
    )
    group.add_argument(
        "--qwen_timeout",
        type=int,
        default=1800,
        help="Timeout per rollout in seconds",
    )
    
    # Evaluation configuration
    group.add_argument(
        "--eval_timeout",
        type=int,
        default=900,
        help="Timeout for swebench evaluation in seconds",
    )
    
    return parser


def setup_qwen_swe_args(args):
    """Set up additional args required by SLiME for SWE-bench training."""
    
    # Data source path
    args.data_source_path = "examples.qwen_swe.data_source:create_data_source"
    
    # Rollout function paths
    args.rollout_function_path = "examples.qwen_swe.rollout:generate"
    args.custom_generate_function_path = "examples.qwen_swe.rollout:generate"
    
    # Evaluation function (use default sglang eval)
    if not hasattr(args, "eval_function_path") or args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path
    
    # Reward configuration
    args.custom_rm_path = None
    args.reward_key = None
    args.group_rm = True  # Compute rewards within groups for GRPO
    
    # Loss mask configuration
    args.loss_mask_type = "qwen3"
    
    # Generation settings
    args.rollout_stop = ["<|im_end|>", "</tool>"]
    args.rollout_stop_token_ids = None
    args.rollout_skip_special_tokens = False
    
    # Chat template settings
    args.apply_chat_template = False
    args.apply_chat_template_kwargs = {}
    
    # Dataset settings
    args.rollout_global_dataset = True
    args.prompt_data = None  # We use custom data source
    args.input_key = "prompt"
    args.label_key = None
    args.metadata_key = None
    args.tool_key = None
    args.multimodal_keys = None
    
    # Rollout settings defaults
    if not hasattr(args, "rollout_batch_size") or args.rollout_batch_size is None:
        args.rollout_batch_size = 4
    if not hasattr(args, "n_samples_per_prompt") or args.n_samples_per_prompt is None:
        args.n_samples_per_prompt = 2
    if not hasattr(args, "rollout_shuffle"):
        args.rollout_shuffle = True
    if not hasattr(args, "rollout_max_response_len") or args.rollout_max_response_len is None:
        args.rollout_max_response_len = 16384
    if not hasattr(args, "rollout_max_prompt_len") or args.rollout_max_prompt_len is None:
        args.rollout_max_prompt_len = 8192
        
    # Temperature settings
    if not hasattr(args, "rollout_temperature") or args.rollout_temperature is None:
        args.rollout_temperature = 0.7
    if not hasattr(args, "rollout_top_p") or args.rollout_top_p is None:
        args.rollout_top_p = 0.95
    if not hasattr(args, "rollout_top_k") or args.rollout_top_k is None:
        args.rollout_top_k = -1
    
    # Partial rollout settings
    args.partial_rollout = False
    args.mask_offpolicy_in_partial_rollout = False
    
    # Filter settings
    args.dynamic_sampling_filter_path = None
    args.rollout_sample_filter_path = None
    args.rollout_all_samples_process_path = None
    args.buffer_filter_path = None
    
    # Other required args
    args.ci_test = False
    args.over_sampling_batch_size = args.rollout_batch_size
    args.disable_rollout_trim_samples = False
    args.sglang_enable_deterministic_inference = False
    args.rollout_seed = 42
    
    # GRPO/advantage settings
    if not hasattr(args, "advantage_estimator") or args.advantage_estimator is None:
        args.advantage_estimator = "grpo"
    
    # Training settings defaults
    if not hasattr(args, "train_backend") or args.train_backend is None:
        args.train_backend = "fsdp"
    if not hasattr(args, "lr") or args.lr is None:
        args.lr = 1e-6
    if not hasattr(args, "global_batch_size") or args.global_batch_size is None:
        args.global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt
    
    # Set vLLM URL in environment for qwen_agent
    os.environ["VLLM_URL"] = args.vllm_url
    
    return args


def main():
    """Main entry point."""
    # First parse our custom args
    custom_parser = argparse.ArgumentParser(add_help=False)
    add_qwen_swe_args(custom_parser)
    custom_args, remaining = custom_parser.parse_known_args()
    
    # Then parse SLiME args
    sys.argv = [sys.argv[0]] + remaining
    args = slime_parse_args()
    
    # Merge custom args
    for key, value in vars(custom_args).items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    # Setup SWE-bench specific args
    args = setup_qwen_swe_args(args)
    
    logger.info("=" * 60)
    logger.info("SWE-bench GRPO Training with SLiME")
    logger.info("=" * 60)
    logger.info(f"Model: {args.hf_checkpoint}")
    logger.info(f"vLLM URL: {args.vllm_url}")
    logger.info(f"Instance file: {args.swe_instance_file}")
    logger.info(f"Rollout batch size: {args.rollout_batch_size}")
    logger.info(f"Samples per prompt: {args.n_samples_per_prompt}")
    logger.info(f"Advantage estimator: {args.advantage_estimator}")
    logger.info(f"Train backend: {args.train_backend}")
    logger.info("=" * 60)
    
    # Import and run training
    from train import train
    train(args)


if __name__ == "__main__":
    main()
