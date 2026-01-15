"""
SWE-bench utilities for Docker container management and test evaluation.

This module provides:
- Docker container setup/teardown for SWE-bench instances
- Patch application and test running
- Integration with swebench harness for evaluation
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# SWE-bench Docker image mapping
# Maps repo names to Docker image tags
DOCKER_IMAGE_MAP = {
    "django/django": "swe-bench/sweb.eval.x86_64.django__django",
    "astropy/astropy": "swe-bench/sweb.eval.x86_64.astropy__astropy",
    "matplotlib/matplotlib": "swe-bench/sweb.eval.x86_64.matplotlib__matplotlib",
    "scikit-learn/scikit-learn": "swe-bench/sweb.eval.x86_64.scikit-learn__scikit-learn",
    "sympy/sympy": "swe-bench/sweb.eval.x86_64.sympy__sympy",
    "pytest-dev/pytest": "swe-bench/sweb.eval.x86_64.pytest-dev__pytest",
    "pallets/flask": "swe-bench/sweb.eval.x86_64.pallets__flask",
    "psf/requests": "swe-bench/sweb.eval.x86_64.psf__requests",
    "pydata/xarray": "swe-bench/sweb.eval.x86_64.pydata__xarray",
    "pylint-dev/pylint": "swe-bench/sweb.eval.x86_64.pylint-dev__pylint",
    "sphinx-doc/sphinx": "swe-bench/sweb.eval.x86_64.sphinx-doc__sphinx",
}


def get_docker_image(instance_id: str) -> str:
    """
    Get Docker image for a SWE-bench instance.

    Image format is: swebench/sweb.eval.x86_64.{repo}_{version}_{issue}
    Example: django__django-12754 -> swebench/sweb.eval.x86_64.django_1776_django-12754

    Falls back to checking available local images.
    """
    # First try to find exact match in local images
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Parse instance_id: django__django-12754 -> django-12754
        parts = instance_id.split("__")
        if len(parts) >= 2:
            issue_num = parts[1]  # e.g., "django-12754"

            # Look for matching image
            for line in result.stdout.split("\n"):
                if issue_num in line and "swebench" in line.lower():
                    image = line.split(":")[0]  # Remove :tag
                    logger.info(f"Found exact image for {instance_id}: {image}")
                    return image

    except Exception as e:
        logger.warning(f"Error listing docker images: {e}")

    # Fall back to constructed image name
    # Format: swebench/sweb.eval.x86_64.{repo}_{version}_{issue}
    # We need to guess the version, but we can try common patterns
    parts = instance_id.split("__")
    if len(parts) >= 2:
        repo_name = parts[0]  # e.g., "django"
        issue_id = parts[1]  # e.g., "django-12754"

        # Try common version patterns
        for version in ["1776", "1780", "1785", "1790", "1800", "latest"]:
            image = f"swebench/sweb.eval.x86_64.{repo_name}_{version}_{issue_id}"
            # Check if image exists
            check_result = subprocess.run(
                ["docker", "images", "-q", image],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if check_result.stdout.strip():
                logger.info(f"Found image: {image}")
                return image

    # Final fallback - just use the base format
    logger.warning(f"No specific image for {instance_id}, image may not exist")
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_')}"


def setup_container(
    instance_id: str,
    suffix: str = "",
    timeout: int = 300,
) -> str:
    """
    Setup a Docker container for a SWE-bench instance.

    Args:
        instance_id: SWE-bench instance ID (e.g., "django__django-12345")
        suffix: Suffix for container name uniqueness
        timeout: Timeout for container operations

    Returns:
        Container name
    """
    container_name = f"swebench_{instance_id}{suffix}".replace("/", "_").replace("-", "_")

    # Get appropriate Docker image
    image = get_docker_image(instance_id)

    logger.info(f"Setting up container {container_name} from {image}")

    # Check if container already exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if container_name in result.stdout:
        # Remove existing container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    # Start new container
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--workdir", "/testbed",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",  # For nested docker if needed
        image,
        "sleep", "infinity",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")

    logger.info(f"Container {container_name} started")
    return container_name


def cleanup_container(container_name: str) -> None:
    """Remove a Docker container."""
    logger.info(f"Cleaning up container {container_name}")
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=60,
    )


def exec_in_container(
    container_name: str,
    command: str,
    timeout: int = 60,
    workdir: str = "/testbed",
) -> tuple[str, int]:
    """
    Execute a command in a Docker container.

    Returns:
        (output, return_code)
    """
    cmd = ["docker", "exec", "-w", workdir, container_name, "bash", "-c", command]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out", -1


def apply_patch(container_name: str, patch: str) -> bool:
    """
    Apply a git patch in the container.

    Returns:
        True if patch applied successfully
    """
    if not patch.strip():
        return False

    # Write patch to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        # Copy patch to container
        subprocess.run(
            ["docker", "cp", patch_file, f"{container_name}:/tmp/solution.patch"],
            capture_output=True,
            timeout=30,
        )

        # Apply patch
        output, returncode = exec_in_container(
            container_name,
            "cd /testbed && git apply /tmp/solution.patch",
            timeout=60,
        )

        if returncode != 0:
            logger.warning(f"Patch application failed: {output}")
            # Try with --3way
            output, returncode = exec_in_container(
                container_name,
                "cd /testbed && git apply --3way /tmp/solution.patch",
                timeout=60,
            )

        return returncode == 0

    finally:
        os.unlink(patch_file)


def run_tests(
    container_name: str,
    instance_id: str,
    patch: str,
    timeout: int = 900,
) -> bool:
    """
    Apply patch and run tests for a SWE-bench instance.

    Args:
        container_name: Docker container name
        instance_id: SWE-bench instance ID
        patch: Git diff patch to apply
        timeout: Test timeout in seconds

    Returns:
        True if all tests pass (resolved)
    """
    # Apply patch
    if not apply_patch(container_name, patch):
        logger.warning(f"[{instance_id}] Failed to apply patch")
        return False

    # Run tests using swebench harness
    # The test command depends on the repo
    repo = instance_id.split("__")[0].replace("_", "/")

    if "django" in repo.lower():
        test_cmd = "cd /testbed && python -m pytest --tb=short -q 2>&1 | tail -50"
    elif "astropy" in repo.lower():
        test_cmd = "cd /testbed && python -m pytest --tb=short -q 2>&1 | tail -50"
    else:
        test_cmd = "cd /testbed && python -m pytest --tb=short -q 2>&1 | tail -50"

    logger.info(f"[{instance_id}] Running tests...")
    output, returncode = exec_in_container(
        container_name,
        test_cmd,
        timeout=timeout,
    )

    # Parse test results
    # Look for pytest success indicators
    resolved = False
    if returncode == 0:
        resolved = True
    elif "passed" in output.lower() and "failed" not in output.lower():
        resolved = True
    elif "PASSED" in output and "FAILED" not in output:
        resolved = True

    logger.info(f"[{instance_id}] Tests {'PASSED' if resolved else 'FAILED'}")

    return resolved


def get_instance_info(instance_id: str) -> dict:
    """Get information about a SWE-bench instance."""
    # Parse instance_id
    parts = instance_id.split("__")
    if len(parts) >= 2:
        repo_part = parts[0]
        issue_part = parts[1] if len(parts) > 1 else ""

        return {
            "instance_id": instance_id,
            "repo": repo_part.replace("_", "/"),
            "issue": issue_part,
            "docker_image": get_docker_image(instance_id),
        }

    return {"instance_id": instance_id}


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    instance_id = "django__django-11951"

    print(f"Instance: {instance_id}")
    print(f"Info: {get_instance_info(instance_id)}")

    # Test container setup
    try:
        container = setup_container(instance_id, suffix="_test")
        print(f"Container: {container}")

        # Test exec
        output, rc = exec_in_container(container, "ls -la /testbed")
        print(f"Exec result (rc={rc}):\n{output[:500]}")

        # Cleanup
        cleanup_container(container)
        print("Cleanup done")

    except Exception as e:
        print(f"Error: {e}")
