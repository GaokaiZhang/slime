"""
GPU Inference Module for SSR - Supports both local GPU and Modal.

Provides unified interface for model inference:
- Local GPU: Direct transformers/SGLang inference
- Modal GPU: Remote function calls

Usage:
    # Local GPU
    engine = LocalGPUEngine(model_name="Kwai-Klear/Klear-AgentForge-8B-SFT")
    response = engine.generate(prompt, max_tokens=500)

    # Modal GPU (when no local GPU)
    engine = ModalGPUEngine(model_name="Kwai-Klear/Klear-AgentForge-8B-SFT")
    response = engine.generate(prompt, max_tokens=500)
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT"
    hf_token: str = ""

    # Generation parameters
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1

    # Model configuration
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True

    # Stop tokens
    stop_strings: list = field(default_factory=lambda: ["</tool>", "<|im_end|>"])


class BaseInferenceEngine(ABC):
    """Base class for inference engines."""

    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_strings: list | None = None,
    ) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_strings: list | None = None,
    ) -> str:
        """Async generate response from prompt."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class LocalGPUEngine(BaseInferenceEngine):
    """
    Local GPU inference engine using transformers.

    Supports direct GPU inference when CUDA is available.
    """

    def __init__(self, config: InferenceConfig | None = None):
        super().__init__(config)
        self._check_cuda()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            self._has_cuda = torch.cuda.is_available()
            if self._has_cuda:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, will use CPU (slow)")
            return self._has_cuda
        except ImportError:
            logger.error("PyTorch not installed")
            return False

    def initialize(self) -> None:
        """Initialize local model and tokenizer."""
        if self._initialized:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading model: {self.config.model_name}")

        # Set HF token
        if self.config.hf_token:
            os.environ["HF_TOKEN"] = self.config.hf_token

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token or None,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token or None,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )

        logger.info(f"Model loaded: {self.model.config.model_type}")
        self._initialized = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_strings: list | None = None,
    ) -> str:
        """Generate response using local GPU."""
        if not self._initialized:
            self.initialize()

        import torch

        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature
        stops = stop_strings or self.config.stop_strings

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else None,
                do_sample=temp > 0,
                top_p=self.config.top_p if temp > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=stops if stops else None,
                tokenizer=self.tokenizer if stops else None,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_strings: list | None = None,
    ) -> str:
        """Async generate (runs in thread pool)."""
        return await asyncio.to_thread(
            self.generate,
            prompt,
            max_new_tokens,
            temperature,
            stop_strings,
        )

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        self._initialized = False


class SGLangEngine(BaseInferenceEngine):
    """
    SGLang-based inference engine for high-throughput generation.

    Supports both local SGLang server and remote server.
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        server_url: str = "http://localhost:8000",
    ):
        super().__init__(config)
        self.server_url = server_url
        self._process = None

    def initialize(self) -> None:
        """Start SGLang server if not running."""
        if self._initialized:
            return

        import requests

        # Check if server is already running
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"SGLang server already running at {self.server_url}")
                self._initialized = True
                return
        except:
            pass

        # Start server
        logger.info(f"Starting SGLang server for {self.config.model_name}")
        self._start_server()
        self._initialized = True

    def _start_server(self) -> None:
        """Start SGLang server subprocess."""
        import subprocess
        import time
        import requests

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.config.model_name,
            "--port", "8000",
            "--host", "0.0.0.0",
            "--dtype", self.config.dtype,
            "--trust-remote-code",
        ]

        if self.config.hf_token:
            os.environ["HF_TOKEN"] = self.config.hf_token

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        max_wait = 300
        start = time.time()
        while time.time() - start < max_wait:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("SGLang server ready")
                    return
            except:
                pass
            time.sleep(5)

        raise RuntimeError("SGLang server failed to start")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_strings: list | None = None,
    ) -> str:
        """Generate using SGLang server."""
        if not self._initialized:
            self.initialize()

        import requests

        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
            },
        }

        if stop_strings or self.config.stop_strings:
            payload["sampling_params"]["stop"] = stop_strings or self.config.stop_strings

        response = requests.post(
            f"{self.server_url}/generate",
            json=payload,
            timeout=120,
        )

        result = response.json()
        return result.get("text", "")

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_strings: list | None = None,
    ) -> str:
        """Async generate using aiohttp."""
        if not self._initialized:
            self.initialize()

        import aiohttp

        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
            },
        }

        if stop_strings or self.config.stop_strings:
            payload["sampling_params"]["stop"] = stop_strings or self.config.stop_strings

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                result = await response.json()
                return result.get("text", "")

    def cleanup(self) -> None:
        """Stop SGLang server."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None
        self._initialized = False


def create_inference_engine(
    backend: str = "auto",
    config: InferenceConfig | None = None,
    **kwargs,
) -> BaseInferenceEngine:
    """
    Create inference engine based on available hardware.

    Args:
        backend: "auto", "local", "sglang", or "modal"
        config: Inference configuration
        **kwargs: Additional backend-specific arguments

    Returns:
        BaseInferenceEngine instance
    """
    config = config or InferenceConfig()

    if backend == "auto":
        # Auto-detect best available backend
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Using local GPU backend")
                return LocalGPUEngine(config)
        except ImportError:
            pass

        # Fall back to SGLang if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                logger.info("Using SGLang backend")
                return SGLangEngine(config, **kwargs)
        except:
            pass

        # Default to local (will be slow on CPU)
        logger.warning("No GPU detected, using local backend (may be slow)")
        return LocalGPUEngine(config)

    elif backend == "local":
        return LocalGPUEngine(config)

    elif backend == "sglang":
        return SGLangEngine(config, **kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}")


# Convenience function for quick inference
def generate(
    prompt: str,
    model_name: str = "Kwai-Klear/Klear-AgentForge-8B-SFT",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    backend: str = "auto",
) -> str:
    """
    Quick inference function for single generations.

    Args:
        prompt: Input prompt
        model_name: HuggingFace model name
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        backend: Inference backend ("auto", "local", "sglang")

    Returns:
        Generated text
    """
    config = InferenceConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    engine = create_inference_engine(backend=backend, config=config)
    try:
        engine.initialize()
        return engine.generate(prompt)
    finally:
        engine.cleanup()


if __name__ == "__main__":
    # Test local GPU inference
    import torch

    print("=" * 60)
    print("GPU Inference Module Test")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test with a simple prompt
    config = InferenceConfig(
        model_name="Kwai-Klear/Klear-AgentForge-8B-SFT",
        hf_token="HF_TOKEN_PLACEHOLDER",
        max_new_tokens=100,
    )

    engine = create_inference_engine(backend="auto", config=config)

    try:
        engine.initialize()

        test_prompt = """<|im_start|>system
You are a helpful coding assistant.
<|im_end|>
<|im_start|>user
Write a simple Python function that adds two numbers.
<|im_end|>
<|im_start|>assistant
"""

        print("\nGenerating response...")
        response = engine.generate(test_prompt, max_new_tokens=200)
        print("\n" + "-" * 60)
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)

    finally:
        engine.cleanup()
