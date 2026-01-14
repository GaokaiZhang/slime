"""
SWE-bench environment management using Docker containers.

Provides functions to setup, manage, and cleanup Docker containers
for SWE-bench evaluation.
"""

import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default Docker image mapping for common repos
# These are pre-built SWE-bench Docker images
REPO_TO_IMAGE = {
    "django/django": "swebench/django",
    "astropy/astropy": "swebench/astropy",
    "scikit-learn/scikit-learn": "swebench/scikit-learn",
    "matplotlib/matplotlib": "swebench/matplotlib",
    "sympy/sympy": "swebench/sympy",
    "pylint-dev/pylint": "swebench/pylint",
    "pytest-dev/pytest": "swebench/pytest",
    "pallets/flask": "swebench/flask",
    "psf/requests": "swebench/requests",
    "sphinx-doc/sphinx": "swebench/sphinx",
}


def get_docker_image_for_instance(instance_id: str) -> str:
    """
    Get the Docker image name for a SWE-bench instance.

    Uses swebench package to get the correct image, or falls back to mapping.
    """
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from datasets import load_dataset

        # Load instance to get repo and version
        ds = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
        instance = None
        for ex in ds:
            if ex["instance_id"] == instance_id:
                instance = ex
                break

        if instance:
            repo = instance["repo"]
            version = instance["version"]
            # Get image from swebench mapping
            if repo in MAP_REPO_VERSION_TO_SPECS:
                specs = MAP_REPO_VERSION_TO_SPECS[repo]
                if version in specs:
                    return specs[version].get("image_key", f"swebench/{repo.split('/')[1]}")

            # Fallback to simple mapping
            return REPO_TO_IMAGE.get(repo, f"swebench/{repo.split('/')[1]}")

    except Exception as e:
        logger.warning(f"Could not determine Docker image for {instance_id}: {e}")

    # Extract repo from instance_id (format: repo__issue-number)
    parts = instance_id.split("__")
    if len(parts) >= 1:
        repo_part = parts[0].replace("_", "/")
        return REPO_TO_IMAGE.get(repo_part, f"swebench/{repo_part.split('/')[-1]}")

    return "swebench/django"  # Default fallback


def setup_container(
    instance_id: str,
    suffix: str = "",
    workdir: str = "/testbed",
    timeout: int = 300,
) -> str:
    """
    Setup a Docker container for a SWE-bench instance.

    Args:
        instance_id: The SWE-bench instance ID
        suffix: Optional suffix for container name (e.g., for parallel runs)
        workdir: Working directory in container
        timeout: Setup timeout in seconds

    Returns:
        Container name
    """
    # Generate unique container name
    container_name = f"swebench_{instance_id}{suffix}".replace("/", "_").replace("-", "_")

    # Get Docker image
    docker_image = get_docker_image_for_instance(instance_id)

    logger.info(f"Setting up container {container_name} with image {docker_image}")

    # Check if container already exists
    check_result = subprocess.run(
        ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"],
        capture_output=True, text=True
    )

    if check_result.stdout.strip():
        # Remove existing container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, text=True
        )

    # Start container
    try:
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-w", workdir,
                docker_image,
                "sleep", "infinity"
            ],
            capture_output=True, text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        logger.info(f"Container {container_name} started")

        # Wait for container to be ready
        for _ in range(30):
            check = subprocess.run(
                ["docker", "exec", container_name, "echo", "ready"],
                capture_output=True, text=True
            )
            if check.returncode == 0:
                break
            time.sleep(1)

        return container_name

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout starting container for {instance_id}")


def cleanup_container(container_name: str, force: bool = True) -> None:
    """
    Cleanup a Docker container.

    Args:
        container_name: Name of container to remove
        force: Force removal even if running
    """
    try:
        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(container_name)

        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logger.info(f"Removed container {container_name}")

    except Exception as e:
        logger.warning(f"Failed to cleanup container {container_name}: {e}")


def exec_in_container(
    container_name: str,
    command: str,
    timeout: int = 120,
    workdir: Optional[str] = None,
) -> tuple[str, str, int]:
    """
    Execute a command in a container.

    Args:
        container_name: Container to execute in
        command: Command to run
        timeout: Execution timeout
        workdir: Optional working directory

    Returns:
        (stdout, stderr, return_code)
    """
    cmd = ["docker", "exec"]
    if workdir:
        cmd.extend(["-w", workdir])
    cmd.extend([container_name, "bash", "-c", command])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode

    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1


def get_git_diff(container_name: str) -> str:
    """Get git diff from container."""
    stdout, stderr, rc = exec_in_container(container_name, "git diff")
    return stdout if rc == 0 else ""


def apply_patch(container_name: str, patch: str) -> bool:
    """Apply a git patch in the container."""
    # Write patch to temp file
    result = subprocess.run(
        ["docker", "exec", "-i", container_name, "bash", "-c", "cat > /tmp/patch.diff"],
        input=patch,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False

    # Apply patch
    stdout, stderr, rc = exec_in_container(
        container_name,
        "git apply /tmp/patch.diff"
    )

    return rc == 0


def reset_to_base(container_name: str) -> bool:
    """Reset container repo to base commit."""
    stdout, stderr, rc = exec_in_container(
        container_name,
        "git checkout -- . && git clean -fd"
    )
    return rc == 0
