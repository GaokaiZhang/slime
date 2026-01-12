#!/usr/bin/env python3
"""
Start local vLLM server for SWE-bench GRPO training.

Usage:
    python examples/qwen_swe/start_vllm.py --port 8000 --tp 8
"""

import argparse
import subprocess
import sys


def start_vllm_server(
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    port: int = 8000,
    tensor_parallel: int = 8,
    max_model_len: int = 32768,
):
    """Start vLLM server with tool calling support."""
    cmd = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-model-len", str(max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--trust-remote-code",
    ]

    print("Starting vLLM server:")
    print(f"  Model: {model_name}")
    print(f"  Port: {port}")
    print(f"  Tensor parallel: {tensor_parallel}")
    print(f"  URL: http://localhost:{port}/v1")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Server failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start local vLLM inference server")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model length")

    args = parser.parse_args()

    start_vllm_server(
        model_name=args.model,
        port=args.port,
        tensor_parallel=args.tp,
        max_model_len=args.max_model_len,
    )
