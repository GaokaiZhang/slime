"""
Simple vLLM-based SWE agent for GRPO training.

This agent:
1. Calls vLLM API directly (OpenAI-compatible)
2. Captures completion_token_ids and logprobs for GRPO
3. Uses tool-calling format similar to qwen-code/openhands
4. Works with any vLLM-served model (Qwen, Llama, etc.)

This is a lightweight alternative to terminus-2/openhands that provides
the token-level data needed for GRPO training.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Tool definitions (similar to qwen-code/openhands)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your solution when the task is complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "The git diff patch of your changes"}
                },
                "required": ["patch"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an expert software engineer. You have access to the following tools:

1. bash(command): Execute bash commands
2. read_file(path): Read file contents
3. write_file(path, content): Write to files
4. submit(patch): Submit your solution

When you want to use a tool, output a JSON object with "tool" and "args" keys.
Example: {"tool": "bash", "args": {"command": "ls -la"}}

Think step by step. Explore the codebase, understand the issue, implement a fix, and test it.
When done, use submit() with your git diff patch."""


@dataclass
class AgentResult:
    """Result from running the agent."""
    patch: str = ""
    trajectory: dict = field(default_factory=dict)
    completion_token_ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    total_tokens: int = 0
    exit_status: str = "completed"


@dataclass
class VLLMAgentConfig:
    """Configuration for the vLLM agent."""
    api_url: str = "http://localhost:8000"
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    max_tokens: int = 4096
    temperature: float = 1.0
    max_turns: int = 50
    timeout: int = 1800
    # Optional: path to tokenizer for fallback token ID extraction
    # If vLLM doesn't support return_token_ids, we can use the tokenizer
    tokenizer_path: str | None = None


# Global tokenizer cache
_TOKENIZER_CACHE: dict[str, Any] = {}


def get_tokenizer(model_name: str):
    """Get or create tokenizer for a model (cached)."""
    if model_name not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer for {model_name}...")
            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            _TOKENIZER_CACHE[model_name] = None
    return _TOKENIZER_CACHE[model_name]


def call_vllm(
    api_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    tokenizer=None,
) -> dict:
    """
    Call vLLM API and get response with logprobs and token IDs.

    Returns dict with:
    - content: str
    - completion_token_ids: list[int]
    - logprobs: list[float]

    Token ID extraction methods (in order of preference):
    1. vLLM extra_body.return_token_ids=True -> provider_specific_fields.token_ids
    2. Fallback: use tokenizer to convert token strings to IDs
    """
    url = f"{api_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,  # Request logprobs for GRPO
        "top_logprobs": 1,
        # vLLM-specific: request token IDs in response
        # This is NOT part of standard OpenAI API, but vLLM supports it
        "extra_body": {
            "return_token_ids": True,
        },
    }

    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()

    choice = data["choices"][0]
    content = choice["message"]["content"]

    # Extract token IDs and logprobs
    token_ids = []
    logprobs = []
    tokens = []  # Token strings for fallback

    # Method 1: Try to get token_ids from provider_specific_fields (vLLM with return_token_ids=True)
    # vLLM returns this when extra_body.return_token_ids=True
    provider_token_ids = None
    if "provider_specific_fields" in choice:
        provider_token_ids = choice["provider_specific_fields"].get("token_ids")
    # Also check at top level of choice (some vLLM versions)
    if not provider_token_ids and "token_ids" in choice:
        provider_token_ids = choice["token_ids"]

    # Extract logprobs (always available when logprobs=True)
    if "logprobs" in choice and choice["logprobs"]:
        logprobs_content = choice["logprobs"].get("content", [])
        for item in logprobs_content:
            logprobs.append(item.get("logprob", 0.0))
            tokens.append(item.get("token", ""))

    # Use provider token_ids if available
    if provider_token_ids and isinstance(provider_token_ids, list):
        token_ids = provider_token_ids
        logger.debug(f"Got {len(token_ids)} token IDs from vLLM provider_specific_fields")
    # Method 2: Fallback - use tokenizer to convert token strings
    elif tokenizer is not None and tokens:
        try:
            # Encode each token individually
            # Note: This may not perfectly match due to tokenization edge cases
            for token_str in tokens:
                if token_str:
                    encoded = tokenizer.encode(token_str, add_special_tokens=False)
                    if encoded:
                        token_ids.append(encoded[0])
                    else:
                        token_ids.append(0)  # Unknown token
                else:
                    token_ids.append(0)
            logger.debug(f"Converted {len(token_ids)} tokens using tokenizer fallback")
        except Exception as e:
            logger.warning(f"Tokenizer fallback failed: {e}")
            token_ids = []
    else:
        # No token IDs available - log warning
        if not provider_token_ids:
            logger.warning(
                "vLLM did not return token_ids. Ensure vLLM version supports "
                "extra_body.return_token_ids=True, or provide a tokenizer for fallback."
            )

    return {
        "content": content,
        "completion_token_ids": token_ids,
        "logprobs": logprobs,
        "tokens": tokens,  # Include token strings for debugging
        "usage": data.get("usage", {}),
    }


def parse_tool_call(content: str) -> tuple[str, dict] | None:
    """Parse tool call from model output."""
    # Try to find JSON objects in the response
    # Look for patterns like {"tool": "...", "args": {...}}

    # First try: find any JSON object
    try:
        # Find all potential JSON objects (including nested)
        depth = 0
        start = -1
        for i, c in enumerate(content):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        candidate = content[start:i+1]
                        data = json.loads(candidate)
                        if "tool" in data and "args" in data:
                            return data["tool"], data["args"]
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass

    # Fallback: simple regex
    json_pattern = r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}'
    matches = re.findall(json_pattern, content, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if "tool" in data and "args" in data:
                return data["tool"], data["args"]
        except json.JSONDecodeError:
            continue

    return None


def execute_tool(
    tool_name: str,
    args: dict,
    workdir: str = "/testbed",
    container_id: str | None = None,
) -> str:
    """
    Execute a tool and return the result.

    If container_id is provided, executes within the Docker container.
    Otherwise, executes locally.
    """
    try:
        if tool_name == "bash":
            cmd = args.get("command", "")
            if container_id:
                # Execute in Docker container
                docker_cmd = ["docker", "exec", "-w", workdir, container_id, "bash", "-c", cmd]
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            output = result.stdout + result.stderr
            return output[:10000]  # Truncate long outputs

        elif tool_name == "read_file":
            path = args.get("path", "")
            if not path.startswith("/"):
                path = os.path.join(workdir, path)

            if container_id:
                docker_cmd = ["docker", "exec", container_id, "cat", path]
                result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    return f"Error reading file: {result.stderr}"
                return result.stdout[:50000]
            else:
                with open(path, "r") as f:
                    return f.read()[:50000]

        elif tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path.startswith("/"):
                path = os.path.join(workdir, path)

            if container_id:
                # Write file in container using cat with heredoc
                # Escape content for shell
                escaped_content = content.replace("'", "'\"'\"'")
                write_cmd = f"mkdir -p $(dirname {path}) && cat > {path} << 'EOF'\n{content}\nEOF"
                docker_cmd = ["docker", "exec", "-w", workdir, container_id, "bash", "-c", write_cmd]
                result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    return f"Error writing file: {result.stderr}"
                return f"Written to {path}"
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                return f"Written to {path}"

        elif tool_name == "submit":
            return "SUBMIT:" + args.get("patch", "")

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        return f"Error: {str(e)}"


def start_docker_container(
    instance_id: str,
    image: str | None = None,
) -> str | None:
    """
    Start a Docker container for SWE-bench instance.

    Returns container ID if successful, None otherwise.
    """
    from examples.harbor.swebench_utils import get_docker_image

    if image is None:
        image = get_docker_image(instance_id)

    try:
        # Start container with workdir at /testbed
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--workdir", "/testbed",
                image,
                "sleep", "3600"  # Keep container alive for 1 hour
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            container_id = result.stdout.strip()[:12]
            logger.info(f"Started Docker container {container_id} for {instance_id}")
            return container_id
        else:
            logger.error(f"Failed to start container: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Failed to start Docker container: {e}")
        return None


def stop_docker_container(container_id: str) -> None:
    """Stop and remove a Docker container."""
    try:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)
        logger.info(f"Stopped Docker container {container_id}")
    except Exception as e:
        logger.warning(f"Failed to stop container {container_id}: {e}")


def run_agent(
    instruction: str,
    config: VLLMAgentConfig,
    workdir: str = "/testbed",
    tokenizer=None,
    instance_id: str | None = None,
    use_docker: bool = False,
) -> AgentResult:
    """
    Run the vLLM agent on a task.

    Args:
        instruction: Task instruction (problem statement)
        config: Agent configuration
        workdir: Working directory for tool execution
        tokenizer: Optional tokenizer for fallback token ID extraction
        instance_id: SWE-bench instance ID (for Docker image selection)
        use_docker: Whether to run tools in Docker container

    Returns:
        AgentResult with patch, trajectory, token_ids, and logprobs
    """
    result = AgentResult()

    # Get tokenizer for fallback (lazy load if not provided)
    if tokenizer is None:
        tokenizer_path = config.tokenizer_path or config.model_name
        tokenizer = get_tokenizer(tokenizer_path)

    # Start Docker container if requested
    container_id = None
    if use_docker and instance_id:
        container_id = start_docker_container(instance_id)
        if container_id:
            logger.info(f"Running agent in Docker container {container_id}")
        else:
            logger.warning("Failed to start Docker container, running locally")

    try:
        # Initialize conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        # Collect all token IDs and logprobs across turns
        all_token_ids = []
        all_logprobs = []

        # Trajectory steps
        steps = [
            {"step_id": 1, "source": "user", "message": instruction}
        ]

        for turn in range(config.max_turns):
            try:
                # Call vLLM with tokenizer for fallback
                response = call_vllm(
                    api_url=config.api_url,
                    model=config.model_name,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    tokenizer=tokenizer,
                )

                content = response["content"]

                # Collect token data for GRPO
                if response["completion_token_ids"]:
                    all_token_ids.extend(response["completion_token_ids"])
                    all_logprobs.extend(response["logprobs"])

                result.total_tokens += response["usage"].get("total_tokens", 0)

                # Add assistant message to history
                messages.append({"role": "assistant", "content": content})

                # Parse tool call
                tool_call = parse_tool_call(content)

                step = {
                    "step_id": len(steps) + 1,
                    "source": "agent",
                    "message": content,
                }

                if tool_call:
                    tool_name, tool_args = tool_call
                    step["tool_call"] = {"name": tool_name, "args": tool_args}

                    # Check for submit
                    if tool_name == "submit":
                        result.patch = tool_args.get("patch", "")
                        step["observation"] = "Solution submitted"
                        steps.append(step)
                        logger.info(f"Agent submitted solution after {turn + 1} turns")
                        break

                    # Execute tool (in Docker if container_id is set)
                    tool_result = execute_tool(tool_name, tool_args, workdir, container_id)
                    step["observation"] = tool_result[:5000]

                    # Add tool result to conversation
                    messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

                steps.append(step)

                # Check for natural completion
                if not tool_call and ("submit" in content.lower() or "done" in content.lower()):
                    logger.info(f"Agent completed naturally after {turn + 1} turns")
                    break

            except Exception as e:
                logger.error(f"Turn {turn + 1} failed: {e}")
                result.exit_status = "failed"
                break
        else:
            result.exit_status = "max_turns"
            logger.warning(f"Agent reached max turns ({config.max_turns})")

        # Get patch from Docker if we submitted
        if container_id and result.patch:
            # Get the actual git diff from container
            try:
                git_result = subprocess.run(
                    ["docker", "exec", "-w", workdir, container_id, "git", "diff"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if git_result.returncode == 0 and git_result.stdout.strip():
                    result.patch = git_result.stdout
                    logger.info(f"Captured git diff patch: {len(result.patch)} chars")
            except Exception as e:
                logger.warning(f"Failed to get git diff: {e}")

        # Build trajectory
        result.trajectory = {
            "schema_version": "ATIF-v1.5",
            "agent": {
                "name": "vllm-swe-agent",
                "model_name": config.model_name,
            },
            "steps": steps,
        }

        result.completion_token_ids = all_token_ids
        result.logprobs = all_logprobs

    finally:
        # Clean up Docker container
        if container_id:
            stop_docker_container(container_id)

    return result


# For use with Harbor's Trial API
class VLLMAgent:
    """Harbor-compatible vLLM agent wrapper."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        model_name: str | None = None,
        api_url: str | None = None,
        max_turns: int = 50,
        temperature: float = 1.0,
        collect_rollout_details: bool = True,
        **kwargs,
    ):
        self.config = VLLMAgentConfig(
            api_url=api_url or os.environ.get("VLLM_URL", "http://localhost:8000"),
            model_name=model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            max_turns=max_turns,
            temperature=temperature,
        )
        self._collect_rollout_details = collect_rollout_details
        self._result: AgentResult | None = None

    @staticmethod
    def name() -> str:
        return "vllm-swe-agent"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        """Setup is a no-op for this agent."""
        pass

    async def run(self, instruction: str, environment, context) -> None:
        """Run the agent on a task."""
        import asyncio

        # Get workdir from environment if available
        workdir = getattr(environment, "workdir", "/testbed")

        # Run agent (blocking call wrapped in executor)
        loop = asyncio.get_event_loop()
        self._result = await loop.run_in_executor(
            None,
            lambda: run_agent(instruction, self.config, workdir)
        )

        # Populate context for Harbor
        if context and self._collect_rollout_details:
            context.rollout_details = [{
                "completion_token_ids": [self._result.completion_token_ids],
                "logprobs": [self._result.logprobs],
            }]
            context.metadata = {
                "trajectory": self._result.trajectory,
                "patch": self._result.patch,
            }


if __name__ == "__main__":
    # Quick test
    import sys

    logging.basicConfig(level=logging.INFO)

    config = VLLMAgentConfig(
        api_url=os.environ.get("VLLM_URL", "http://localhost:8000"),
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        max_turns=3,
    )

    result = run_agent(
        instruction="List the files in the current directory and tell me what you see.",
        config=config,
        workdir=os.getcwd(),
    )

    print(f"Exit status: {result.exit_status}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Token IDs collected: {len(result.completion_token_ids)}")
    print(f"Logprobs collected: {len(result.logprobs)}")
    print(f"Steps: {len(result.trajectory['steps'])}")
