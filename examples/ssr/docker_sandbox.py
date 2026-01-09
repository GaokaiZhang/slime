"""
Docker Sandbox for SSR - Manages SWE-bench compatible docker containers.

Provides isolated environments for:
- Running bug injector agents
- Running bug solver agents
- Executing test scripts
- Validating bug artifacts
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound

logger = logging.getLogger(__name__)


@dataclass
class DockerSandboxConfig:
    """Configuration for docker sandbox."""

    # Docker image
    image_name: str = "swebench/sweb.eval.x86_64.django_1776"
    image_tag: str = "latest"

    # Container settings
    timeout: int = 600  # 10 minutes max
    memory_limit: str = "16g"
    cpu_limit: float = 4.0

    # Workspace
    repo_path: str = "/testbed"
    work_dir: str = "/testbed"

    # Cleanup
    auto_remove: bool = True
    cleanup_on_error: bool = True


class DockerSandbox:
    """
    Docker-based sandbox for running SSR agents.

    Uses SWE-bench compatible docker images to provide isolated
    environments for bug injection and solving.
    """

    def __init__(self, config: DockerSandboxConfig | None = None):
        self.config = config or DockerSandboxConfig()
        self.client = docker.from_env()
        self.container = None
        self._last_test_results: dict[str, str] = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(
        self,
        image_name: str | None = None,
        instance_id: str | None = None,
        gold_patch: str | None = None,
    ) -> None:
        """
        Start a new container from the specified image.

        IMPORTANT for SSR: The SWE-bench Docker image contains the ORIGINAL BUG.
        For bug injection, we must first apply the gold_patch to fix the original
        bug, giving us a clean codebase to inject NEW bugs into.

        Args:
            image_name: Docker image name (optional)
            instance_id: SWE-bench instance ID (e.g., "django__django-16255")
            gold_patch: The gold patch from SWE-bench that fixes the original bug.
                        MUST be provided for bug injection to work correctly!
        """
        if self.container is not None:
            logger.warning("Container already running, stopping it first")
            self.stop()

        # Use provided image or config default
        image = image_name or f"{self.config.image_name}:{self.config.image_tag}"

        # If instance_id is provided, try to find matching SWE-bench image
        if instance_id and not image_name:
            image = self._find_swebench_image(instance_id)

        logger.info(f"Starting container from image: {image}")

        try:
            self.container = self.client.containers.run(
                image,
                detach=True,
                tty=True,
                stdin_open=True,
                working_dir=self.config.work_dir,
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.config.cpu_limit * 100000),
                auto_remove=self.config.auto_remove,
            )
            logger.info(f"Container started: {self.container.short_id}")

            # Wait for container to be ready
            time.sleep(1)

            # CRITICAL: Apply gold patch to fix original bug FIRST
            # This gives us a clean codebase for bug injection
            if gold_patch:
                logger.info("Applying gold patch to fix original SWE-bench bug...")
                success = self.apply_patch(gold_patch)
                if success:
                    logger.info("Gold patch applied successfully - codebase is now clean")
                else:
                    logger.warning("Gold patch application failed - original bug may still exist!")

        except ImageNotFound:
            logger.error(f"Image not found: {image}")
            raise
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            raise

    def apply_gold_patch(self, gold_patch: str) -> bool:
        """
        Apply the gold patch to fix the original SWE-bench bug.

        This MUST be called before bug injection to ensure we're working
        with a clean codebase. The SWE-bench Docker images contain the
        original buggy code - we need to fix it first.

        Args:
            gold_patch: The patch from SWE-bench that fixes the original bug

        Returns:
            True if patch applied successfully, False otherwise
        """
        if not gold_patch or not gold_patch.strip():
            logger.warning("No gold patch provided - original bug remains!")
            return False

        logger.info("Applying gold patch to fix original bug...")
        success = self.apply_patch(gold_patch)

        if success:
            logger.info("Gold patch applied - original bug fixed, ready for bug injection")
        else:
            logger.error("Failed to apply gold patch - original bug still present!")

        return success

    def _find_swebench_image(self, instance_id: str) -> str:
        """
        Find SWE-bench docker image for given instance ID.

        SWE-bench image naming convention:
            instance_id: django__django-16255
            image: swebench/sweb.eval.x86_64.django_1776_django-16255:latest

        The pattern is: replace '__' with '_1776_' in instance_id.
        """
        # Direct conversion: django__django-16255 -> django_1776_django-16255
        id_docker = instance_id.replace("__", "_1776_").lower()
        image_name = f"swebench/sweb.eval.x86_64.{id_docker}:latest"

        # Verify the image exists
        try:
            self.client.images.get(image_name)
            logger.info(f"Found SWE-bench image: {image_name}")
            return image_name
        except ImageNotFound:
            logger.warning(f"SWE-bench image not found: {image_name}")
            # Fall back to searching
            pass

        # Fallback: Search for any matching image
        parts = instance_id.split("__")
        if len(parts) == 2:
            repo = parts[0].lower()
            issue = parts[1]
            images = self.client.images.list()
            for img in images:
                for tag in img.tags:
                    if f"sweb.eval.x86_64.{repo}" in tag and issue in tag:
                        logger.info(f"Found matching SWE-bench image: {tag}")
                        return tag

        # Final fallback to default
        logger.warning(f"No SWE-bench image found for {instance_id}, using default")
        return f"{self.config.image_name}:{self.config.image_tag}"

    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container is not None:
            try:
                self.container.stop(timeout=5)
                logger.info(f"Container stopped: {self.container.short_id}")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            finally:
                self.container = None

    def reset(self) -> None:
        """Reset the container to clean state (git reset --hard)."""
        if self.container is None:
            raise RuntimeError("No container running")

        self.exec_command("git reset --hard HEAD")
        self.exec_command("git clean -fd")

    def wipe_git_history(self) -> None:
        """
        Wipe git history to prevent information leakage.

        CRITICAL for SSR: The solver must NOT have access to git history,
        as it could reveal the bug injection through `git log`, `git diff`, etc.

        This removes .git and reinitializes with a fresh commit.
        """
        if self.container is None:
            raise RuntimeError("No container running")

        logger.info("Wiping git history to prevent information leakage...")
        self.exec_command("rm -rf .git")
        self.exec_command("git init")
        self.exec_command("git add -A")
        self.exec_command("git commit -m 'initial state' --allow-empty")
        logger.info("Git history wiped - solver isolated from commit history")

    def exec_command(
        self,
        command: str,
        timeout: int | None = None,
        work_dir: str | None = None,
    ) -> str:
        """Execute a command in the container."""
        if self.container is None:
            raise RuntimeError("No container running")

        timeout = timeout or self.config.timeout
        work_dir = work_dir or self.config.work_dir

        try:
            exit_code, output = self.container.exec_run(
                f"bash -c '{command}'",
                workdir=work_dir,
                demux=True,
            )

            stdout = output[0].decode() if output[0] else ""
            stderr = output[1].decode() if output[1] else ""

            if exit_code != 0:
                logger.debug(f"Command failed (exit {exit_code}): {command}")
                logger.debug(f"stderr: {stderr}")

            return stdout + stderr

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    def write_file(self, path: str, content: str) -> None:
        """Write a file in the container."""
        if self.container is None:
            raise RuntimeError("No container running")

        # Use echo with base64 to handle special characters
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        self.exec_command(f"echo {encoded} | base64 -d > {path}")

    def read_file(self, path: str) -> str:
        """Read a file from the container."""
        return self.exec_command(f"cat {path}")

    def apply_patch(self, patch: str, reverse: bool = False) -> bool:
        """Apply a git patch in the container."""
        if not patch.strip():
            return True

        # Write patch to temp file
        patch_file = "/tmp/patch.diff"
        self.write_file(patch_file, patch)

        # Apply patch
        reverse_flag = "-R" if reverse else ""
        result = self.exec_command(f"git apply {reverse_flag} {patch_file}")

        if "error" in result.lower() or "failed" in result.lower():
            logger.warning(f"Patch application failed: {result}")
            return False

        return True

    def revert_file(self, file_path: str) -> None:
        """Revert a specific file to HEAD."""
        self.exec_command(f"git checkout HEAD -- {file_path}")

    def run_tests(self, test_script: str, timeout: int = 90) -> str:
        """
        Run tests using the provided test script.

        Args:
            test_script: Content of test_script.sh
            timeout: Timeout in seconds

        Returns:
            Test output string
        """
        # Write test script
        script_path = "/tmp/test_script.sh"
        self.write_file(script_path, test_script)
        self.exec_command(f"chmod +x {script_path}")

        # Run tests
        result = self.exec_command(
            f"timeout {timeout} bash {script_path} 2>&1",
            timeout=timeout + 10,
        )

        return result

    def parse_test_output(self, parser_script: str, test_output: str) -> dict[str, str]:
        """
        Parse test output using the provided parser script.

        Args:
            parser_script: Content of parse_test_output.py
            test_output: Raw test output to parse

        Returns:
            Dict mapping test_id -> "passed" | "failed"
        """
        # Write parser script
        parser_path = "/tmp/parse_test_output.py"
        self.write_file(parser_path, parser_script)

        # Write test output
        output_path = "/tmp/test_output.log"
        self.write_file(output_path, test_output)

        # Run parser
        result = self.exec_command(f"cat {output_path} | python3 {parser_path}")

        try:
            self._last_test_results = json.loads(result)
            return self._last_test_results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse test output: {result[:500]}")
            return {}

    def get_last_test_results(self) -> dict[str, str]:
        """Get the last parsed test results."""
        return self._last_test_results

    def check_tests_pass(self, test_script: str, parser_script: str) -> tuple[bool, dict[str, str]]:
        """
        Run tests and check if all pass.

        Returns:
            Tuple of (all_pass, test_results)
        """
        test_output = self.run_tests(test_script)
        test_results = self.parse_test_output(parser_script, test_output)

        all_pass = all(v == "passed" for v in test_results.values())
        return all_pass, test_results


class AsyncDockerSandbox:
    """
    Async wrapper for DockerSandbox.

    Provides async interface for use with slime rollout.
    """

    def __init__(self, config: DockerSandboxConfig | None = None):
        self.config = config or DockerSandboxConfig()
        self._sandbox: DockerSandbox | None = None
        self._loop = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(
        self,
        image_name: str | None = None,
        instance_id: str | None = None,
        gold_patch: str | None = None,
    ) -> None:
        """
        Start container asynchronously.

        Args:
            image_name: Docker image name
            instance_id: SWE-bench instance ID
            gold_patch: Gold patch to apply first (fixes original bug)
        """
        self._sandbox = DockerSandbox(self.config)
        await asyncio.to_thread(self._sandbox.start, image_name, instance_id, gold_patch)

    async def stop(self) -> None:
        """Stop container asynchronously."""
        if self._sandbox:
            await asyncio.to_thread(self._sandbox.stop)
            self._sandbox = None

    async def reset(self) -> None:
        """Reset container asynchronously."""
        if self._sandbox:
            await asyncio.to_thread(self._sandbox.reset)

    async def wipe_git_history(self) -> None:
        """Wipe git history asynchronously to prevent solver information leakage."""
        if self._sandbox:
            await asyncio.to_thread(self._sandbox.wipe_git_history)

    async def exec_command(self, command: str, timeout: int | None = None) -> str:
        """Execute command asynchronously."""
        if self._sandbox:
            return await asyncio.to_thread(self._sandbox.exec_command, command, timeout)
        raise RuntimeError("No sandbox running")

    async def apply_patch(self, patch: str, reverse: bool = False) -> bool:
        """Apply patch asynchronously."""
        if self._sandbox:
            return await asyncio.to_thread(self._sandbox.apply_patch, patch, reverse)
        raise RuntimeError("No sandbox running")

    async def run_tests(self, test_script: str, timeout: int = 90) -> str:
        """Run tests asynchronously."""
        if self._sandbox:
            return await asyncio.to_thread(self._sandbox.run_tests, test_script, timeout)
        raise RuntimeError("No sandbox running")

    async def check_tests_pass(
        self, test_script: str, parser_script: str
    ) -> tuple[bool, dict[str, str]]:
        """Check if tests pass asynchronously."""
        if self._sandbox:
            return await asyncio.to_thread(
                self._sandbox.check_tests_pass, test_script, parser_script
            )
        raise RuntimeError("No sandbox running")


class SandboxPool:
    """
    Pool of docker sandboxes for concurrent execution.

    Manages multiple sandbox instances for parallel bug injection/solving.
    """

    def __init__(
        self,
        pool_size: int = 4,
        config: DockerSandboxConfig | None = None,
    ):
        self.pool_size = pool_size
        self.config = config or DockerSandboxConfig()
        self._sandboxes: list[AsyncDockerSandbox] = []
        self._available: asyncio.Queue | None = None
        self._started = False

    async def start(self) -> None:
        """Initialize the sandbox pool."""
        if self._started:
            return

        self._available = asyncio.Queue()

        for _ in range(self.pool_size):
            sandbox = AsyncDockerSandbox(self.config)
            self._sandboxes.append(sandbox)
            await self._available.put(sandbox)

        self._started = True
        logger.info(f"Sandbox pool started with {self.pool_size} instances")

    async def stop(self) -> None:
        """Stop all sandboxes in the pool."""
        for sandbox in self._sandboxes:
            await sandbox.stop()
        self._sandboxes.clear()
        self._started = False
        logger.info("Sandbox pool stopped")

    async def acquire(self) -> AsyncDockerSandbox:
        """Acquire a sandbox from the pool."""
        if not self._started or self._available is None:
            raise RuntimeError("Pool not started")
        return await self._available.get()

    async def release(self, sandbox: AsyncDockerSandbox) -> None:
        """Release a sandbox back to the pool."""
        if self._available is not None:
            await self._available.put(sandbox)

    async def execute(
        self,
        func,
        instance_id: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Execute a function with a sandbox from the pool.

        Args:
            func: Async function that takes sandbox as first argument
            instance_id: Optional SWE-bench instance ID
            *args, **kwargs: Additional arguments to pass to func

        Returns:
            Result from func
        """
        sandbox = await self.acquire()
        try:
            await sandbox.start(instance_id=instance_id)
            return await func(sandbox, *args, **kwargs)
        finally:
            await sandbox.stop()
            await self.release(sandbox)
