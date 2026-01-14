"""
RL Agent for GRPO training with Harbor.

This agent wraps vLLM API calls and records completion_token_ids and logprobs
in ATIF format for use in SLiME GRPO training.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class RLAgentConfig:
    """Configuration for RL agent."""
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    api_base_url: str = "http://localhost:8000"
    max_turns: int = 50
    timeout: int = 1800
    max_tokens: int = 4096
    temperature: float = 1.0  # GRPO uses temperature=1.0
    return_logprobs: bool = True  # Enable for RL training


@dataclass
class RLAgentResult:
    """Result from RL agent execution with ATIF-compatible trajectory."""
    patch: str = ""
    exit_status: str = "unknown"
    trajectory: dict = field(default_factory=dict)  # ATIF format
    raw_messages: list = field(default_factory=list)


# Tool definitions for the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
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
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Pattern to search for"},
                    "path": {"type": "string", "description": "Path to search in"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file by replacing old_string with new_string",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "String to replace"},
                    "new_string": {"type": "string", "description": "Replacement string"}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
]


SYSTEM_PROMPT = """You are an expert software engineer. You have access to the following tools to help you complete tasks:

1. list_directory(path): List files in a directory
2. read_file(path): Read contents of a file
3. write_file(path, content): Write content to a file
4. run_shell_command(command): Run a shell command
5. grep_search(pattern, path): Search for a pattern in files
6. edit(path, old_string, new_string): Edit a file by replacing old_string with new_string

To use a tool, output a JSON object in a code block like this:
```json
{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
```

After using a tool, I will provide the result, and you can continue with more tool calls or provide your final answer.

When you're done making changes, always run the tests to verify your fix works.
"""


def execute_tool(container_name: str, tool_name: str, args: dict) -> str:
    """Execute a tool in the Docker container."""
    try:
        if tool_name == "list_directory":
            path = args.get("path", ".")
            result = subprocess.run(
                ["docker", "exec", container_name, "ls", "-la", path],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

        elif tool_name == "read_file":
            path = args.get("path", "")
            result = subprocess.run(
                ["docker", "exec", container_name, "cat", path],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            return output

        elif tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            result = subprocess.run(
                ["docker", "exec", "-i", container_name, "bash", "-c", f"cat > {path}"],
                input=content, capture_output=True, text=True, timeout=30
            )
            return "File written successfully" if result.returncode == 0 else f"Error: {result.stderr}"

        elif tool_name == "run_shell_command":
            command = args.get("command", "")
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-c", command],
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout + result.stderr
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output if output else "(no output)"

        elif tool_name == "grep_search":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            result = subprocess.run(
                ["docker", "exec", container_name, "grep", "-rn", pattern, path],
                capture_output=True, text=True, timeout=60
            )
            output = result.stdout if result.returncode == 0 else "No matches found"
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output

        elif tool_name == "edit":
            path = args.get("path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")

            read_result = subprocess.run(
                ["docker", "exec", container_name, "cat", path],
                capture_output=True, text=True, timeout=30
            )
            if read_result.returncode != 0:
                return f"Error reading file: {read_result.stderr}"

            content = read_result.stdout
            if old_string not in content:
                return "Error: old_string not found in file"

            new_content = content.replace(old_string, new_string, 1)
            write_result = subprocess.run(
                ["docker", "exec", "-i", container_name, "bash", "-c", f"cat > {path}"],
                input=new_content, capture_output=True, text=True, timeout=30
            )
            return "Edit successful" if write_result.returncode == 0 else f"Error: {write_result.stderr}"

        else:
            return f"Unknown tool: {tool_name}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"


def parse_tool_calls(content: str) -> list[dict]:
    """Parse tool calls from model output."""
    tool_calls = []

    # JSON in code blocks
    json_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
    for match in re.finditer(json_block_pattern, content, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            if "name" in obj and "arguments" in obj:
                tool_calls.append({
                    "name": obj["name"],
                    "arguments": obj["arguments"]
                })
        except json.JSONDecodeError:
            pass

    # Raw JSON tool call (if no code block matches)
    if not tool_calls:
        try:
            json_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]+\}\s*\}'
            for match in re.finditer(json_pattern, content):
                obj = json.loads(match.group(0))
                tool_calls.append({
                    "name": obj["name"],
                    "arguments": obj["arguments"]
                })
        except (json.JSONDecodeError, AttributeError):
            pass

    return tool_calls


def call_vllm_api(
    api_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    return_logprobs: bool = True,
) -> dict:
    """Call vLLM API with logprobs enabled for RL training."""
    url = f"{api_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    # Request logprobs for RL training
    if return_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def run_rl_agent(
    container_name: str,
    instance_id: str,
    prompt: str,
    config: RLAgentConfig,
) -> RLAgentResult:
    """
    Run RL agent loop that records token_ids and logprobs in ATIF format.

    This is designed for GRPO training - records all data needed for loss computation.
    """
    import time
    from datetime import datetime

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Initialize ATIF trajectory
    session_id = f"{instance_id}_{int(time.time())}"
    trajectory = {
        "schema_version": "ATIF-v1.5",
        "session_id": session_id,
        "agent": {
            "name": "slime-rl-agent",
            "version": "1.0.0",
            "model_name": config.model_name,
            "tool_definitions": TOOLS,
        },
        "steps": [],
        "final_metrics": {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        },
    }

    step_id = 0

    # Add user message as step
    step_id += 1
    trajectory["steps"].append({
        "step_id": step_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": "user",
        "message": prompt,
    })

    for turn in range(config.max_turns):
        print(f"[{instance_id}] Turn {turn + 1}/{config.max_turns}")

        try:
            response = call_vllm_api(
                api_url=config.api_base_url,
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                return_logprobs=config.return_logprobs,
            )

            choice = response["choices"][0]
            assistant_content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            usage = response.get("usage", {})

            # Extract logprobs if available
            logprobs_data = choice.get("logprobs", {})
            token_logprobs = []
            token_ids = []

            if logprobs_data and "content" in logprobs_data:
                for token_info in logprobs_data["content"]:
                    if "logprob" in token_info:
                        token_logprobs.append(token_info["logprob"])
                    # vLLM may return token or token_id
                    if "token_id" in token_info:
                        token_ids.append(token_info["token_id"])

            # Build ATIF step with metrics
            step_id += 1
            agent_step = {
                "step_id": step_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "agent",
                "model_name": config.model_name,
                "message": assistant_content,
                "metrics": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            }

            # Add RL-specific fields if available
            if token_logprobs:
                agent_step["metrics"]["logprobs"] = token_logprobs
            if token_ids:
                agent_step["metrics"]["completion_token_ids"] = token_ids

            # Update totals
            trajectory["final_metrics"]["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
            trajectory["final_metrics"]["total_completion_tokens"] += usage.get("completion_tokens", 0)

            # Parse tool calls
            tool_calls = parse_tool_calls(assistant_content)

            if tool_calls:
                agent_step["tool_calls"] = [
                    {
                        "tool_call_id": f"call_{i}",
                        "function_name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                    for i, tc in enumerate(tool_calls)
                ]

                # Execute tools and get observations
                observations = []
                tool_results = []

                for i, tc in enumerate(tool_calls):
                    print(f"[{instance_id}] Executing: {tc['name']}")
                    result = execute_tool(container_name, tc["name"], tc["arguments"])
                    truncated_result = result[:2000] if len(result) > 2000 else result

                    observations.append({
                        "source_call_id": f"call_{i}",
                        "content": truncated_result,
                    })
                    tool_results.append({
                        "tool": tc["name"],
                        "result": truncated_result,
                    })

                agent_step["observation"] = {"results": observations}

                # Add tool results to messages
                tool_response = "\n\n".join([
                    f"Tool: {tr['tool']}\nResult:\n{tr['result']}"
                    for tr in tool_results
                ])
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": f"Tool results:\n{tool_response}"})

            else:
                # No tool calls - agent is done
                messages.append({"role": "assistant", "content": assistant_content})

            trajectory["steps"].append(agent_step)

            if not tool_calls:
                print(f"[{instance_id}] No tool calls, finishing")
                break

        except Exception as e:
            print(f"[{instance_id}] Error: {e}")
            break

    # Extract patch from git diff
    patch = ""
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "git", "diff"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            patch = result.stdout
    except Exception:
        pass

    return RLAgentResult(
        patch=patch,
        exit_status="completed" if patch else "completed_no_patch",
        trajectory=trajectory,
        raw_messages=messages,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rl_agent.py <container_name>")
        sys.exit(1)

    container = sys.argv[1]
    config = RLAgentConfig(
        api_base_url=os.environ.get("VLLM_URL", "http://localhost:8000"),
    )

    result = run_rl_agent(
        container_name=container,
        instance_id="test",
        prompt="List the files in /testbed and tell me what you see.",
        config=config,
    )

    print(f"Exit status: {result.exit_status}")
    print(f"Trajectory steps: {len(result.trajectory.get('steps', []))}")
    print(f"Patch: {bool(result.patch)}")
    print(f"Token metrics: {result.trajectory.get('final_metrics', {})}")
