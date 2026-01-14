"""
Simple Python-based agent that calls vLLM API directly.

This avoids the qwen-code CLI streaming issues by using the OpenAI Python client
with non-streaming mode or properly handled streaming.
"""

import json
import os
import subprocess
import time
import re
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class SimpleAgentConfig:
    """Configuration for the simple agent."""
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    api_base_url: str = "http://localhost:8000"
    max_turns: int = 50
    timeout: int = 600
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class AgentResult:
    """Result from agent execution."""
    patch: str = ""
    exit_status: str = "unknown"
    messages: list = field(default_factory=list)
    raw_events: list = field(default_factory=list)


# Tool definitions for Qwen3-Coder
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
            # Truncate long files
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            return output

        elif tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            # Write via stdin
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
            output = result.stdout if result.returncode == 0 else f"No matches found"
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output

        elif tool_name == "edit":
            path = args.get("path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")

            # Read file
            read_result = subprocess.run(
                ["docker", "exec", container_name, "cat", path],
                capture_output=True, text=True, timeout=30
            )
            if read_result.returncode != 0:
                return f"Error reading file: {read_result.stderr}"

            content = read_result.stdout
            if old_string not in content:
                return f"Error: old_string not found in file"

            # Replace and write
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
    """Parse tool calls from model output.

    Supports multiple formats:
    1. XML-style: <tool_call><function=name>...</function></tool_call>
    2. JSON in code blocks: ```json {"name": "...", "arguments": {...}} ```
    3. Raw JSON objects
    """
    tool_calls = []

    # Format 1: XML-style tool calls
    tool_pattern = r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>'
    param_pattern = r'<parameter=(\w+)>\s*(.*?)\s*</parameter>'

    for match in re.finditer(tool_pattern, content, re.DOTALL):
        func_name = match.group(1)
        params_str = match.group(2)

        params = {}
        for param_match in re.finditer(param_pattern, params_str, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            params[param_name] = param_value

        tool_calls.append({
            "name": func_name,
            "arguments": params
        })

    # Format 2: JSON in code blocks
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

    # Format 3: Raw JSON tool call (if no other matches found)
    if not tool_calls:
        try:
            # Try to find JSON object in content
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
    temperature: float = 0.7,
    tools: list[dict] = None,
) -> dict:
    """Call vLLM API with non-streaming mode."""
    url = f"{api_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,  # Non-streaming to avoid parsing issues
    }

    # Add tools if provided
    if tools:
        payload["tools"] = tools

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


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


def run_simple_agent(
    container_name: str,
    instance_id: str,
    prompt: str,
    config: SimpleAgentConfig,
) -> AgentResult:
    """
    Run a simple agent loop using direct vLLM API calls.

    This avoids qwen-code CLI streaming issues by using non-streaming API calls.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    all_events = []

    # System event
    all_events.append({
        "type": "system",
        "subtype": "init",
        "model": config.model_name,
        "container": container_name,
    })

    for turn in range(config.max_turns):
        print(f"[{instance_id}] Turn {turn + 1}/{config.max_turns}")

        try:
            # Call vLLM API (tools described in system prompt)
            response = call_vllm_api(
                api_url=config.api_base_url,
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            assistant_content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0]["finish_reason"]

            # Record assistant event
            all_events.append({
                "type": "assistant",
                "message": {"role": "assistant", "content": assistant_content},
                "finish_reason": finish_reason,
            })

            # Add to conversation
            messages.append({"role": "assistant", "content": assistant_content})

            # Parse tool calls from content
            tool_calls = parse_tool_calls(assistant_content)

            if not tool_calls:
                # No tool calls - agent is done
                print(f"[{instance_id}] No tool calls, finishing")
                break

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                print(f"[{instance_id}] Executing: {tc['name']}")
                result = execute_tool(container_name, tc["name"], tc["arguments"])
                tool_results.append({
                    "tool": tc["name"],
                    "result": result[:2000] if len(result) > 2000 else result
                })

                # Record tool event
                all_events.append({
                    "type": "user",  # Tool results are user messages
                    "message": {"role": "tool", "content": result[:2000]},
                    "tool_name": tc["name"],
                })

            # Add tool results to conversation
            tool_response = "\n\n".join([
                f"Tool: {tr['tool']}\nResult:\n{tr['result']}"
                for tr in tool_results
            ])
            messages.append({"role": "user", "content": f"Tool results:\n{tool_response}"})

        except Exception as e:
            print(f"[{instance_id}] Error: {e}")
            all_events.append({
                "type": "error",
                "error": str(e),
            })
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

    # Record result event
    all_events.append({
        "type": "result",
        "subtype": "success" if patch else "no_patch",
        "has_patch": bool(patch),
    })

    return AgentResult(
        patch=patch,
        exit_status="completed" if patch else "completed_no_patch",
        messages=messages,
        raw_events=all_events,
    )


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_agent.py <container_name>")
        sys.exit(1)

    container = sys.argv[1]
    config = SimpleAgentConfig(
        api_base_url=os.environ.get("VLLM_URL", "http://localhost:8000"),
    )

    result = run_simple_agent(
        container_name=container,
        instance_id="test",
        prompt="List the files in /testbed and tell me what you see.",
        config=config,
    )

    print(f"Exit status: {result.exit_status}")
    print(f"Events: {len(result.raw_events)}")
    print(f"Patch: {bool(result.patch)}")
