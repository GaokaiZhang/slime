#!/usr/bin/env python
"""
Tinker-to-OpenAI Proxy Server

This server exposes an OpenAI-compatible API that forwards requests to Tinker's
native SDK. This allows Harbor agents to use Tinker for inference while
collecting logprobs for PPO training.

Usage:
    # Start the proxy server
    export TINKER_API_KEY="tml-..."
    python examples/harbor/tinker_proxy.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # Then run Harbor with the proxy
    python examples/harbor/harbor_grpo_tinker.py \
        --agent qwen-coder \
        --ak base_url=http://localhost:8000/v1 \
        --ak api_key=dummy \
        --num-rollouts 50

Architecture:
    Harbor Agent (qwen-coder)
        │
        │ OpenAI API calls (POST /v1/chat/completions)
        ▼
    ┌─────────────────────────┐
    │ Tinker Proxy Server     │  ← This script
    │ (localhost:8000)        │
    └─────────────────────────┘
        │
        │ Tinker SDK (sample_async)
        ▼
    ┌─────────────────────────┐
    │ Tinker Cloud (GPU)      │
    └─────────────────────────┘
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")
    exit(1)

# Tinker imports
try:
    import tinker
    import tinker.types as types
except ImportError:
    print("Tinker SDK not installed. Install with: pip install tinker")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app = FastAPI(title="Tinker-to-OpenAI Proxy")
tinker_client = None
tokenizer = None
model_name = None
logprobs_store = {}  # Store logprobs for PPO training


@dataclass
class TinkerProxyState:
    """Global state for the proxy server."""
    service_client: Any = None
    training_client: Any = None
    sampling_client: Any = None
    tokenizer: Any = None
    model_name: str = ""
    lora_rank: int = 32
    # Store logprobs for each request (for PPO training)
    logprobs_cache: Dict[str, Dict] = field(default_factory=dict)


state = TinkerProxyState()


async def init_tinker(model_name: str, lora_rank: int = 32, enable_training: bool = True):
    """Initialize Tinker client."""
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY environment variable not set")

    logger.info(f"Initializing Tinker with model: {model_name}")

    state.model_name = model_name
    state.lora_rank = lora_rank
    state.service_client = tinker.ServiceClient()

    if enable_training:
        # Create training client (enables weight updates)
        logger.info("Creating training client...")
        state.training_client = state.service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
            train_mlp=True,
            train_attn=True,
        )
        state.tokenizer = state.training_client.get_tokenizer()

        # Create sampling client from training client (shares weights)
        logger.info("Creating sampling client from training client...")
        state.sampling_client = state.training_client.save_weights_and_get_sampling_client(
            name=f"proxy-checkpoint-{int(time.time())}"
        )
    else:
        # Sampling only (no training)
        logger.info("Creating standalone sampling client...")
        state.sampling_client = state.service_client.create_sampling_client(
            base_model=model_name
        )
        state.tokenizer = state.sampling_client.get_tokenizer()

    logger.info("Tinker initialization complete")


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to a prompt string."""
    # Simple chat format - adjust based on model requirements
    prompt_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}")

    # Add assistant prefix for generation
    prompt_parts.append("<|assistant|>\n")

    return "\n".join(prompt_parts)


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tinker",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Forwards requests to Tinker's native SDK and returns responses
    in OpenAI format. Also stores logprobs for PPO training.
    """
    if state.sampling_client is None:
        raise HTTPException(status_code=503, detail="Tinker client not initialized")

    body = await request.json()

    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 1.0)
    stream = body.get("stream", False)
    request_logprobs = body.get("logprobs", False)
    top_logprobs = body.get("top_logprobs", 0)

    # Format messages into prompt
    prompt = format_chat_prompt(messages)

    # Tokenize prompt
    prompt_tokens = state.tokenizer.encode(prompt)
    encoded_chunk = tinker.EncodedTextChunk(tokens=prompt_tokens)
    model_input = types.ModelInput(chunks=[encoded_chunk])

    # Generate with Tinker
    try:
        result = await state.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    except Exception as e:
        logger.error(f"Tinker sampling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Extract results
    sequence = result.sequences[0]
    response_tokens = sequence.tokens
    response_logprobs = sequence.logprobs
    response_text = state.tokenizer.decode(response_tokens, skip_special_tokens=True)

    # Generate unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Store logprobs for PPO training (can be retrieved later)
    state.logprobs_cache[completion_id] = {
        "prompt_tokens": prompt_tokens,
        "response_tokens": list(response_tokens),
        "logprobs": list(response_logprobs),
        "text": response_text,
        "timestamp": time.time(),
    }

    # Clean old cache entries (keep last 1000)
    if len(state.logprobs_cache) > 1000:
        oldest_keys = sorted(state.logprobs_cache.keys(),
                           key=lambda k: state.logprobs_cache[k]["timestamp"])[:100]
        for k in oldest_keys:
            del state.logprobs_cache[k]

    # Build OpenAI-compatible response
    if stream:
        return StreamingResponse(
            generate_stream_response(completion_id, response_text),
            media_type="text/event-stream"
        )

    # Build logprobs in OpenAI format if requested
    logprobs_content = None
    if request_logprobs:
        logprobs_content = {
            "content": [
                {
                    "token": state.tokenizer.decode([tok]),
                    "logprob": float(lp),
                    "bytes": None,
                    "top_logprobs": []
                }
                for tok, lp in zip(response_tokens, response_logprobs)
            ]
        }

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": state.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "logprobs": logprobs_content,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(response_tokens),
            "total_tokens": len(prompt_tokens) + len(response_tokens),
        }
    }

    return JSONResponse(content=response)


async def generate_stream_response(completion_id: str, text: str):
    """Generate streaming response chunks."""
    # Initial chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": state.model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Content chunks (simulate streaming by chunking the text)
    chunk_size = 10
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i+chunk_size]
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": state.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)  # Small delay to simulate streaming

    # Final chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": state.model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/logprobs/{completion_id}")
async def get_logprobs(completion_id: str):
    """
    Retrieve stored logprobs for a completion.

    This is a custom endpoint (not OpenAI standard) for PPO training.
    """
    if completion_id not in state.logprobs_cache:
        raise HTTPException(status_code=404, detail="Completion not found")

    return state.logprobs_cache[completion_id]


@app.get("/v1/logprobs")
async def list_logprobs():
    """List all stored logprobs (for debugging)."""
    return {
        "count": len(state.logprobs_cache),
        "ids": list(state.logprobs_cache.keys())[-20:]  # Last 20
    }


@app.post("/v1/train/ppo")
async def train_ppo(request: Request):
    """
    Perform a PPO training step with stored logprobs.

    This is a custom endpoint for GRPO training.

    Request body:
    {
        "completion_ids": ["chatcmpl-xxx", ...],
        "rewards": [1.0, -1.0, ...],
        "learning_rate": 1e-6
    }
    """
    if state.training_client is None:
        raise HTTPException(status_code=503, detail="Training not enabled")

    body = await request.json()
    completion_ids = body.get("completion_ids", [])
    rewards = body.get("rewards", [])
    learning_rate = body.get("learning_rate", 1e-6)

    if len(completion_ids) != len(rewards):
        raise HTTPException(status_code=400, detail="completion_ids and rewards must have same length")

    # Gather rollouts from cache
    rollouts = []
    for cid, reward in zip(completion_ids, rewards):
        if cid not in state.logprobs_cache:
            logger.warning(f"Completion {cid} not found in cache, skipping")
            continue

        cached = state.logprobs_cache[cid]
        rollouts.append({
            "tokens": cached["response_tokens"],
            "logprobs": cached["logprobs"],
            "reward": reward,
        })

    if not rollouts:
        raise HTTPException(status_code=400, detail="No valid completions found")

    # Compute GRPO advantages
    import numpy as np

    rewards_arr = [r["reward"] for r in rollouts]
    mean_reward = sum(rewards_arr) / len(rewards_arr)
    variance = sum((r - mean_reward) ** 2 for r in rewards_arr) / len(rewards_arr)
    std_reward = max(variance ** 0.5, 1e-8)
    advantages = [(r - mean_reward) / std_reward for r in rewards_arr]

    # Create PPO training data
    data = []
    for rollout, advantage in zip(rollouts, advantages):
        response_tokens = list(rollout["tokens"])
        response_logprobs = list(rollout["logprobs"])

        encoded_chunk = tinker.EncodedTextChunk(tokens=response_tokens)
        model_input = types.ModelInput(chunks=[encoded_chunk])

        tokens_arr = np.array(response_tokens, dtype=np.int64)
        logprobs_arr = np.array(response_logprobs, dtype=np.float32)
        advantages_arr = np.full(len(response_tokens), advantage, dtype=np.float32)

        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(tokens_arr),
                "logprobs": tinker.TensorData.from_numpy(logprobs_arr),
                "advantages": tinker.TensorData.from_numpy(advantages_arr),
            }
        )
        data.append(datum)

    # Forward-backward with PPO loss
    fwd_bwd_future = state.training_client.forward_backward(
        data=data,
        loss_fn="ppo",
        loss_fn_config={
            "clip_low_threshold": 0.8,
            "clip_high_threshold": 1.28,
        }
    )
    fwd_bwd_result = fwd_bwd_future.result()

    # Optimizer step
    optim_future = state.training_client.optim_step(
        types.AdamParams(learning_rate=learning_rate)
    )
    optim_result = optim_future.result()

    # Update sampling client with new weights
    state.sampling_client = state.training_client.save_weights_and_get_sampling_client(
        name=f"proxy-checkpoint-{int(time.time())}"
    )

    return {
        "status": "success",
        "n_samples": len(rollouts),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "advantages": advantages,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": state.model_name,
        "training_enabled": state.training_client is not None,
        "logprobs_cached": len(state.logprobs_cache),
    }


def main():
    parser = argparse.ArgumentParser(description="Tinker-to-OpenAI Proxy Server")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                       help="Tinker model to use")
    parser.add_argument("--lora-rank", type=int, default=32,
                       help="LoRA rank for training")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to")
    parser.add_argument("--no-training", action="store_true",
                       help="Disable training (inference only)")

    args = parser.parse_args()

    # Initialize Tinker on startup
    logger.info(f"Starting Tinker proxy with model: {args.model}")

    async def startup():
        await init_tinker(
            model_name=args.model,
            lora_rank=args.lora_rank,
            enable_training=not args.no_training
        )

    @app.on_event("startup")
    async def on_startup():
        await startup()

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
