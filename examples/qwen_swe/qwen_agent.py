"""
Qwen-code agent wrapper for running in SWE-bench Docker containers.

This module provides a wrapper around the qwen-code CLI for use with
SLiME's GRPO training pipeline.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QwenAgentConfig:
    """Configuration for qwen-code agent."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  # 30B MoE model
    api_base_url: str = ""  # Modal vLLM server URL
    api_key: str = "not-needed"  # For OpenAI-compatible endpoints

    # Agent parameters
    max_turns: int = 100
    timeout: int = 3600  # 1 hour per instance

    # Container configuration
    container_name_prefix: str = "qwen_swe"


@dataclass
class AgentResult:
    """Result from running qwen-code agent."""

    success: bool
    patch: str
    exit_status: str  # "submitted", "completed_no_patch", "failed", "timeout"
    messages: List[Dict[str, str]]  # Conversation history (converted)
    raw_events: List[Dict[str, Any]]  # Raw JSON events for loss masking
    duration: float
    returncode: int
    stdout: str
    stderr: str


def get_docker_image(instance_id: str) -> str:
    """Get the SWE-bench docker image name for an instance."""
    # Docker doesn't allow double underscore, so replace with magic token
    id_docker = instance_id.replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{id_docker}:latest"


def setup_container(
    instance_id: str,
    suffix: str = "",
    timeout: int = 300,
) -> str:
    """
    Start a Docker container for the given instance with qwen-code installed.

    Args:
        instance_id: SWE-bench instance ID (e.g., "django__django-16255")
        suffix: Optional suffix for container name (for parallel runs)
        timeout: Timeout for container setup in seconds

    Returns:
        Container name
    """
    image = get_docker_image(instance_id)
    container_name = f"qwen_swe_{instance_id.replace('__', '_')}{suffix}"

    # Remove existing container if any
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )

    # Start new container
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "-w", "/testbed",
            image,
            "tail", "-f", "/dev/null"
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    container_id = result.stdout.strip()
    logger.info(f"[{instance_id}] Started container {container_id[:12]} from {image}")

    # Install Node.js 20
    logger.info(f"[{instance_id}] Installing Node.js 20...")
    install_cmd = """
apt-get update && apt-get install -y ca-certificates curl gnupg && \
mkdir -p /etc/apt/keyrings && \
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
echo 'deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main' | tee /etc/apt/sources.list.d/nodesource.list && \
apt-get update && apt-get install -y nodejs
"""
    subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", install_cmd],
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    # Install qwen-code CLI
    logger.info(f"[{instance_id}] Installing qwen-code CLI...")
    subprocess.run(
        ["docker", "exec", container_name, "npm", "install", "-g", "@qwen-code/qwen-code@latest"],
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    # Verify installation
    result = subprocess.run(
        ["docker", "exec", container_name, "qwen", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info(f"[{instance_id}] Qwen CLI version: {result.stdout.strip()}")

    return container_name


def run_qwen_agent(
    container_name: str,
    instance_id: str,
    prompt: str,
    config: QwenAgentConfig,
) -> AgentResult:
    """
    Run qwen-code CLI in the container.

    Args:
        container_name: Docker container name
        instance_id: SWE-bench instance ID
        prompt: Task prompt for the agent
        config: Agent configuration

    Returns:
        AgentResult with patch and trajectory
    """
    # Write prompt to a temp file to avoid shell escaping issues
    prompt_file = "/tmp/qwen_prompt.txt"
    output_file = "/tmp/qwen_output.json"

    # First, write the prompt to a file in the container
    write_cmd = ["docker", "exec", "-i", container_name, "bash", "-c", f"cat > {prompt_file}"]
    write_result = subprocess.run(
        write_cmd,
        input=prompt,
        capture_output=True,
        text=True,
    )

    # Now run qwen with the prompt file
    cmd = [
        "docker", "exec",
        "-e", f"OPENAI_API_KEY={config.api_key}",
        "-e", f"OPENAI_BASE_URL={config.api_base_url}/v1",
        "-e", f"OPENAI_MODEL={config.model_name}",
        container_name,
        "bash", "-c",
        f'qwen --auth-type openai -y --output-format json --max-session-turns {config.max_turns} "$(cat {prompt_file})" > {output_file} 2>&1'
    ]

    logger.info(f"[{instance_id}] Running qwen-code with max {config.max_turns} turns...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        duration = time.time() - start_time
        logger.info(f"[{instance_id}] Completed in {duration:.1f}s (rc={result.returncode})")

        # Read output from the temp file
        read_result = subprocess.run(
            ["docker", "exec", container_name, "cat", output_file],
            capture_output=True,
            text=True,
        )
        logger.info(f"[{instance_id}] Output file: {len(read_result.stdout)} chars")
        result_stdout = read_result.stdout if read_result.returncode == 0 else ""
        result_stderr = result.stderr

        # Parse JSON output from the temp file
        events = []
        output = result_stdout.strip()
        output_source = "file"

        # If output file was empty, try stderr
        if not output:
            output = result_stderr.strip()
            output_source = "stderr"

        logger.info(f"[{instance_id}] {output_source} chars: {len(output)}")
        logger.info(f"[{instance_id}] {output_source} preview: {output[:200]}...")

        # Try different parsing strategies
        if output:
            # Strategy 1: Try as JSON array (qwen-code --output-format json outputs array)
            try:
                parsed = json.loads(output)
                if isinstance(parsed, list):
                    events = parsed
                    logger.info(f"[{instance_id}] Parsed as JSON array: {len(events)} events")
                else:
                    events = [parsed]
                    logger.info(f"[{instance_id}] Parsed as single JSON object")
            except json.JSONDecodeError:
                # Strategy 2: Try as newline-separated JSON
                lines = output.split("\n")
                logger.info(f"[{instance_id}] Trying JSONL parsing: {len(lines)} lines")
                parse_errors = 0
                for line in lines:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            parse_errors += 1
                logger.info(f"[{instance_id}] Parsed {len(events)} events from JSONL, {parse_errors} errors")

        # Convert events to messages
        messages = _convert_events_to_messages(events)

        # Extract patch
        patch = extract_patch(container_name, instance_id)

        # Determine exit status
        if patch:
            exit_status = "submitted"
        elif result.returncode == 0:
            exit_status = "completed_no_patch"
        else:
            exit_status = "failed"

        return AgentResult(
            success=bool(patch),
            patch=patch,
            exit_status=exit_status,
            messages=messages,
            raw_events=events,
            duration=duration,
            returncode=result.returncode,
            stdout=result_stdout,
            stderr=result_stderr,
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.warning(f"[{instance_id}] Timeout after {duration:.1f}s")
        return AgentResult(
            success=False,
            patch="",
            exit_status="timeout",
            messages=[],
            raw_events=[],
            duration=duration,
            returncode=-1,
            stdout="",
            stderr=f"Timeout after {config.timeout}s",
        )


def extract_patch(container_name: str, instance_id: str) -> str:
    """Extract git diff from the container."""
    result = subprocess.run(
        ["docker", "exec", container_name, "git", "diff"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def extract_trajectory(container_name: str, instance_id: str) -> List[Dict]:
    """
    Extract qwen-code chat records from the container.

    qwen-code stores records at ~/.qwen/tmp/<project_id>/chats/<session_id>.jsonl
    """
    # Find the chat records file
    find_cmd = "find /root/.qwen -name '*.jsonl' -type f 2>/dev/null | head -1"
    result = subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", find_cmd],
        capture_output=True,
        text=True,
    )

    jsonl_path = result.stdout.strip()
    if not jsonl_path:
        logger.warning(f"[{instance_id}] No qwen-code chat records found")
        return []

    # Read the JSONL file
    result = subprocess.run(
        ["docker", "exec", container_name, "cat", jsonl_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"[{instance_id}] Failed to read chat records")
        return []

    records = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    logger.info(f"[{instance_id}] Extracted {len(records)} chat records")
    return records


def cleanup_container(container_name: str):
    """Stop and remove the container."""
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )


def _convert_events_to_messages(events: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert qwen-code events to standard message format.

    Args:
        events: List of qwen-code JSON events

    Returns:
        List of messages with role and content
    """
    messages = []

    for event in events:
        # Skip non-dict events (qwen-code might output arrays in some cases)
        if not isinstance(event, dict):
            continue

        event_type = event.get("type", "")
        message = event.get("message", {})

        if not message:
            continue

        role = message.get("role", "")
        parts = message.get("parts", [])

        # Extract text content from parts
        content_parts = []
        for part in parts:
            if isinstance(part, str):
                content_parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    content_parts.append(part["text"])
                elif "functionCall" in part:
                    # Format tool call
                    fc = part["functionCall"]
                    content_parts.append(f"[Tool Call: {fc.get('name', 'unknown')}]")
                    if fc.get("args"):
                        content_parts.append(f"Args: {json.dumps(fc['args'], indent=2)}")
                elif "functionResponse" in part:
                    # Format tool response
                    fr = part["functionResponse"]
                    content_parts.append(f"[Tool Response: {fr.get('name', 'unknown')}]")
                    if fr.get("response"):
                        resp = fr["response"]
                        if isinstance(resp, dict) and "result" in resp:
                            result_text = resp["result"]
                            if len(result_text) > 2000:
                                result_text = result_text[:2000] + "..."
                            content_parts.append(f"Result: {result_text}")
                        else:
                            result_text = str(resp)
                            if len(result_text) > 2000:
                                result_text = result_text[:2000] + "..."
                            content_parts.append(f"Result: {result_text}")
                elif "thought" in part:
                    content_parts.append(f"[Thinking] {part['thought']}")

        content = "\n".join(content_parts)
        if not content:
            continue

        # Map role
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "model":
            messages.append({"role": "assistant", "content": content})

    return messages


async def run_qwen_agent_async(
    container_name: str,
    instance_id: str,
    prompt: str,
    config: QwenAgentConfig,
) -> AgentResult:
    """
    Async version of run_qwen_agent using asyncio subprocess.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_qwen_agent,
        container_name,
        instance_id,
        prompt,
        config,
    )
