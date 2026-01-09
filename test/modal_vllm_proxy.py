#!/usr/bin/env python3
"""
Modal vLLM Proxy Server for Qwen3-Coder.

This script runs a vLLM server on Modal GPU and exposes it via FastAPI ASGI.
Much more reliable than @modal.web_server for external access.

Usage:
    # Deploy the server (keeps running)
    modal deploy test/modal_vllm_proxy.py

    # Or run temporarily
    modal serve test/modal_vllm_proxy.py

    # Test the endpoint
    curl https://YOUR_MODAL_APP_URL/v1/models
"""

import modal
import os
import time
import subprocess
import threading
import queue

# Modal app
app = modal.App("qwen3-coder-proxy")

# Hugging Face token
HF_TOKEN = "HF_TOKEN_PLACEHOLDER"

# Model configuration
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MAX_MODEL_LEN = 32768

# Create image with vLLM and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "transformers",
        "torch",
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "httpx",
        "requests",
    )
)

# Volume for caching model weights
model_cache = modal.Volume.from_name("qwen3-coder-cache", create_if_missing=True)

# Global vLLM process manager
vllm_process = None
vllm_ready = False


def start_vllm_background():
    """Start vLLM in the background and wait for it to be ready."""
    global vllm_process, vllm_ready
    import requests

    if vllm_process is not None:
        return

    print(f"Starting vLLM server for {MODEL_NAME}...")

    env = {
        **os.environ,
        "HF_TOKEN": HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
    }

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--host", "127.0.0.1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]

    vllm_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Log output in background thread
    def log_output():
        for line in iter(vllm_process.stdout.readline, b''):
            print(f"[vLLM] {line.decode().rstrip()}")

    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()

    # Wait for server to be ready
    print("Waiting for vLLM server to be ready...")
    for i in range(600):  # Up to 10 minutes
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                print(f"vLLM server ready after {i+1}s!")
                vllm_ready = True
                return
        except Exception:
            pass

        try:
            response = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"vLLM server ready (via /v1/models) after {i+1}s!")
                vllm_ready = True
                return
        except Exception:
            pass

        if (i + 1) % 30 == 0:
            print(f"Still waiting... {i+1}s")
        time.sleep(1)

    raise RuntimeError("vLLM server failed to start within 10 minutes")


# Create FastAPI app
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

fastapi_app = FastAPI()


@fastapi_app.on_event("startup")
async def startup_event():
    """Start vLLM server when FastAPI starts."""
    start_vllm_background()


@fastapi_app.get("/health")
async def health():
    """Health check endpoint."""
    global vllm_ready
    if not vllm_ready:
        return JSONResponse({"status": "loading", "model": MODEL_NAME}, status_code=503)
    return {"status": "healthy", "model": MODEL_NAME, "ready": True}


@fastapi_app.get("/v1/models")
async def list_models():
    """List available models."""
    global vllm_ready
    if not vllm_ready:
        return JSONResponse({"error": "Model still loading"}, status_code=503)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8000/v1/models", timeout=30)
        return Response(content=response.content, media_type="application/json")


@fastapi_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to vLLM."""
    global vllm_ready
    if not vllm_ready:
        return JSONResponse({"error": "Model still loading"}, status_code=503)

    body = await request.body()
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            content=body,
            headers=headers,
            timeout=300,
        )
        return Response(
            content=response.content,
            media_type=response.headers.get("content-type", "application/json"),
        )


@fastapi_app.post("/v1/completions")
async def completions(request: Request):
    """Proxy completions to vLLM."""
    global vllm_ready
    if not vllm_ready:
        return JSONResponse({"error": "Model still loading"}, status_code=503)

    body = await request.body()
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8000/v1/completions",
            content=body,
            headers=headers,
            timeout=300,
        )
        return Response(
            content=response.content,
            media_type=response.headers.get("content-type", "application/json"),
        )


@fastapi_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(request: Request, path: str):
    """Proxy all other requests to vLLM."""
    global vllm_ready
    if not vllm_ready:
        return JSONResponse({"error": "Model still loading"}, status_code=503)

    body = await request.body()

    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=request.method,
            url=f"http://127.0.0.1:8000/{path}",
            content=body if body else None,
            headers=dict(request.headers),
            timeout=300,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN, "HUGGING_FACE_HUB_TOKEN": HF_TOKEN})],
    volumes={"/root/.cache/huggingface": model_cache},
    scaledown_window=900,  # Keep warm for 15 minutes
)
@modal.asgi_app()
def serve_vllm():
    """Serve vLLM via ASGI."""
    return fastapi_app


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN, "HUGGING_FACE_HUB_TOKEN": HF_TOKEN})],
    volumes={"/root/.cache/huggingface": model_cache},
)
def test_model():
    """Test that the model loads and generates correctly."""
    from vllm import LLM, SamplingParams

    print(f"Loading model {MODEL_NAME}...")

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print("Model loaded! Testing generation...")

    prompts = ["Hello, I am a"]
    sampling_params = SamplingParams(max_tokens=50, temperature=0.7)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

    return {"success": True, "model": MODEL_NAME}


@app.local_entrypoint()
def main(action: str = "info"):
    """
    Main entrypoint.

    Args:
        action: info (show URL info), test (test model loading)
    """
    if action == "test":
        print("Testing model loading...")
        result = test_model.remote()
        print(f"Result: {result}")
    elif action == "info":
        print(f"Model: {MODEL_NAME}")
        print(f"Max model len: {MAX_MODEL_LEN}")
        print("\nTo deploy the server, run:")
        print("  modal deploy test/modal_vllm_proxy.py")
        print("\nTo serve temporarily, run:")
        print("  modal serve test/modal_vllm_proxy.py")
        print("\nThe endpoint will be at:")
        print("  https://susvibes-mitigation--qwen3-coder-proxy-serve-vllm.modal.run")
