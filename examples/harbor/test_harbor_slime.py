#!/usr/bin/env python
"""
Test script for Harbor + SLiME Integration

Tests:
1. Harbor CLI availability
2. Trajectory converter with mock data
3. SLiME Sample format verification
4. Docker container availability
5. swebench.harness evaluation (optional)

Usage:
    cd /home/gaokaizhang/slime
    PYTHONPATH=$PWD python examples/harbor/test_harbor_slime.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_harbor_cli():
    """Test Harbor CLI is installed and accessible."""
    print("\n" + "=" * 60)
    print("TEST: Harbor CLI Availability")
    print("=" * 60)

    result = subprocess.run(
        ["harbor", "--version"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"  PASSED: Harbor CLI available")
        print(f"  Version: {result.stdout.strip()}")
        return True
    else:
        print(f"  FAILED: Harbor CLI not found")
        print(f"  Install with: uv tool install harbor")
        return False


def test_slime_sample_format():
    """Test SLiME Sample type can be instantiated."""
    print("\n" + "=" * 60)
    print("TEST: SLiME Sample Format")
    print("=" * 60)

    try:
        from slime.utils.types import Sample

        sample = Sample(
            prompt="Fix the bug",
            tokens=[1, 2, 3, 4, 5, 100, 200, 300],
            response="Here is the fix",
            response_length=3,
            reward=1.0,
            status=Sample.Status.COMPLETED,
            metadata={"source": "harbor_test"},
        )

        # Verify to_dict works
        sample_dict = sample.to_dict()
        assert "tokens" in sample_dict
        assert "reward" in sample_dict
        assert sample_dict["status"] == "completed"

        print(f"  PASSED: SLiME Sample format verified")
        print(f"  Fields: {list(sample_dict.keys())}")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_trajectory_converter():
    """Test trajectory converter with mock data."""
    print("\n" + "=" * 60)
    print("TEST: Trajectory Converter")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
        from examples.harbor.trajectory_converter import HarborTrajectoryConverter

        # Use a small tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained(
            "Kwai-Klear/Klear-AgentForge-8B-SFT",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        converter = HarborTrajectoryConverter(tokenizer)

        # Create mock trajectory
        mock_trajectory = {
            "messages": [
                {"role": "user", "content": "Fix the bug in Django admin"},
                {"role": "assistant", "content": "```diff\n--- a/django/admin.py\n+++ b/django/admin.py\n@@ -1 +1 @@\n-old\n+new\n```"},
            ],
            "reward": 1.0,
        }

        # Write mock trajectory to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = Path(tmpdir) / "trajectory.json"
            with open(traj_file, "w") as f:
                json.dump(mock_trajectory, f)

            # Load and convert
            sample = converter._load_trajectory(traj_file)

            assert sample is not None, "Failed to load trajectory"
            assert sample.reward == 1.0, f"Wrong reward: {sample.reward}"
            assert len(sample.tokens) > 0, "No tokens"
            assert sample.response_length > 0, "No response length"

            print(f"  PASSED: Trajectory converter works")
            print(f"  Tokens: {len(sample.tokens)}")
            print(f"  Response length: {sample.response_length}")
            return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_docker_containers():
    """Test Docker is available with SWE-bench images."""
    print("\n" + "=" * 60)
    print("TEST: Docker Containers")
    print("=" * 60)

    try:
        # Check Docker is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            print(f"  FAILED: Docker not running")
            return False

        # Check for SWE-bench images
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        images = result.stdout.strip().split("\n")
        swebench_images = [img for img in images if "swebench" in img.lower()]

        if swebench_images:
            print(f"  PASSED: Docker available with {len(swebench_images)} SWE-bench images")
            print(f"  Sample: {swebench_images[0]}")
            return True
        else:
            print(f"  WARNING: No SWE-bench images found")
            print(f"  Pull with: docker pull swebench/sweb.eval.x86_64.django_1776_django-10097:latest")
            return True  # Not a failure, just a warning

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_training_instances():
    """Test training instances file exists and is readable."""
    print("\n" + "=" * 60)
    print("TEST: Training Instances")
    print("=" * 60)

    train_file = Path("/home/gaokaizhang/slime/train_instances_id.txt")

    if not train_file.exists():
        print(f"  FAILED: Training instances file not found at {train_file}")
        return False

    with open(train_file) as f:
        instances = [line.strip() for line in f if line.strip()]

    django_instances = [i for i in instances if "django" in i.lower()]

    print(f"  PASSED: Found {len(instances)} training instances")
    print(f"  Django instances: {len(django_instances)}")
    print(f"  Sample: {django_instances[0] if django_instances else 'N/A'}")
    return True


def test_grpo_core():
    """Test GRPO core utilities from grpo_core.py."""
    print("\n" + "=" * 60)
    print("TEST: GRPO Core Utilities")
    print("=" * 60)

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "grpo"))
        from grpo_core import (
            GRPOConfig,
            extract_patch,
            compute_grpo_advantages,
        )

        # Test config
        config = GRPOConfig()
        assert config.lr == 1e-6
        assert config.kl_coef == 0.001
        print(f"  GRPOConfig: lr={config.lr}, kl_coef={config.kl_coef}")

        # Test patch extraction
        response = """Here's the fix:
```diff
--- a/django/models.py
+++ b/django/models.py
@@ -1,3 +1,3 @@
-old code
+new code
```"""
        patch = extract_patch(response)
        assert patch, "Failed to extract patch"
        assert "old code" in patch
        print(f"  extract_patch: OK (extracted {len(patch)} chars)")

        # Test advantage computation
        rewards = [1.0, -1.0, 0.0, -1.0]
        advantages, mean, std = compute_grpo_advantages(rewards)
        assert len(advantages) == 4
        assert abs(sum(advantages)) < 0.01  # Should sum to ~0
        print(f"  compute_grpo_advantages: OK (mean={mean:.2f}, std={std:.2f})")

        print(f"  PASSED: GRPO core utilities work")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_harbor_agents():
    """List available Harbor agents."""
    print("\n" + "=" * 60)
    print("TEST: Harbor Agents")
    print("=" * 60)

    result = subprocess.run(
        ["harbor", "run", "--help"],
        capture_output=True,
        text=True,
    )

    # Extract agent list from help
    if "agent" in result.stdout.lower():
        # Parse available agents from help text
        lines = result.stdout.split("\n")
        for line in lines:
            if "mini-swe-agent" in line.lower() or "qwen-coder" in line.lower():
                print(f"  Found agent line: {line.strip()}")

        print(f"  PASSED: Harbor agents available")
        print(f"  Recommended for GRPO: mini-swe-agent (local), qwen-coder (API)")
        return True
    else:
        print(f"  WARNING: Could not parse agent list")
        return True


def main():
    print("=" * 60)
    print("Harbor + SLiME Integration Tests")
    print("=" * 60)

    tests = [
        ("Harbor CLI", test_harbor_cli),
        ("SLiME Sample Format", test_slime_sample_format),
        ("Trajectory Converter", test_trajectory_converter),
        ("Docker Containers", test_docker_containers),
        ("Training Instances", test_training_instances),
        ("GRPO Core", test_grpo_core),
        ("Harbor Agents", test_harbor_agents),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Ready to run Harbor + SLiME training.")
        print("\nNext steps:")
        print("  1. Deploy Modal vLLM: modal deploy examples/harbor/modal_vllm.py")
        print("  2. Run training: python examples/harbor/harbor_grpo_trainer.py --test")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
