"""
Modal vLLM server for Harbor GRPO training.

Deploys a vLLM server on Modal with A100-80GB GPU for inference.
Uses logprobs for RL training.
"""

import modal
import os
import subprocess
import threading
import time

# Modal app
app = modal.App("slime-grpo-vllm")

# Configuration
DEFAULT_MODEL_NAME = "Kwai-Klear/Klear-AgentForge-8B-SFT"
DEFAULT_MAX_MODEL_LEN = 32768

# Create image with vLLM and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "transformers>=4.48.0",
        "torch>=2.4.0",
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "httpx",
        "requests",
    )
)

# Volume for caching model weights
model_cache = modal.Volume.from_name("slime-grpo-cache", create_if_missing=True)

# HuggingFace token secret
hf_secret = modal.Secret.from_name("hf-token-swe")

# Global vLLM state
vllm_process = None
vllm_ready = False


def get_config():
    """Get configuration from environment."""
    return {
        "model_name": os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)),
    }


def create_fastapi_app():
    """Create FastAPI app inside Modal container."""
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import httpx

    app = FastAPI(title="Harbor GRPO vLLM Server")

    def start_vllm_background():
        """Start vLLM OpenAI-compatible server in background."""
        global vllm_process, vllm_ready
        import requests

        if vllm_process is not None:
            return

        config = get_config()
        model_name = config["model_name"]
        max_model_len = config["max_model_len"]

        print(f"Starting vLLM server for {model_name}...")

        hf_token = os.environ.get("HF_TOKEN", "")
        env = {
            **os.environ,
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        }

        # vLLM command - no auto-tool-choice (we handle tools in prompts)
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", "8000",
            "--host", "127.0.0.1",
            "--max-model-len", str(max_model_len),
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.9",
        ]

        print(f"Command: {' '.join(cmd)}")

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

    @app.on_event("startup")
    async def startup_event():
        """Start vLLM server when FastAPI starts."""
        start_vllm_background()

    @app.get("/")
    async def root():
        """Root endpoint."""
        config = get_config()
        return {
            "status": "ok" if vllm_ready else "loading",
            "model": config["model_name"],
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        config = get_config()
        if not vllm_ready:
            return JSONResponse(
                {"status": "loading", "model": config["model_name"]},
                status_code=503
            )
        return {"status": "healthy", "model": config["model_name"], "ready": True}

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        if not vllm_ready:
            return JSONResponse({"error": "Model still loading"}, status_code=503)

        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8000/v1/models", timeout=30)
            return Response(content=response.content, media_type="application/json")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Proxy chat completions to vLLM."""
        if not vllm_ready:
            return JSONResponse({"error": "Model still loading"}, status_code=503)

        body = await request.body()
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                content=body,
                headers=headers,
                timeout=600,  # 10 minutes for long completions
            )
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "application/json"),
            )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_all(request: Request, path: str):
        """Proxy all other requests to vLLM."""
        if not vllm_ready:
            return JSONResponse({"error": "Model still loading"}, status_code=503)

        body = await request.body()

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=f"http://127.0.0.1:8000/{path}",
                content=body if body else None,
                headers=dict(request.headers),
                timeout=600,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )

    return app


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,  # 4 hours
    secrets=[hf_secret],
    volumes={"/root/.cache/huggingface": model_cache},
    scaledown_window=900,  # Keep warm for 15 minutes
    min_containers=1,
    max_containers=4,
)
@modal.asgi_app()
def serve_vllm():
    """Serve vLLM via ASGI."""
    return create_fastapi_app()


@app.local_entrypoint()
def main(action: str = "info"):
    """Main entrypoint."""
    config = get_config()

    if action == "info":
        print("=" * 60)
        print("Harbor GRPO vLLM Server")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Model: {config['model_name']}")
        print(f"  Max model len: {config['max_model_len']}")
        print("\nTo deploy the server, run:")
        print("  modal deploy examples/grpo/modal_vllm.py")
        print("\nTo serve temporarily, run:")
        print("  modal serve examples/grpo/modal_vllm.py")
