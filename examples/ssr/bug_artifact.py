"""
Bug Artifact representation and validation for SSR.

A bug artifact contains:
- test_files.txt: List of oracle test files
- test_script.sh: Test runner script
- parse_test_output.py: Parser for test output
- bug_patch.diff: Patch that introduces the bug
- test_patch.diff: Patch that weakens tests to hide the bug
"""

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BugArtifact:
    """Represents a complete bug artifact from the injector agent."""

    test_files: list[str] = field(default_factory=list)
    test_script: str = ""
    parse_test_output: str = ""
    bug_patch: str = ""
    test_patch: str = ""

    # Metadata
    repo_path: str = ""
    instance_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_files": self.test_files,
            "test_script": self.test_script,
            "parse_test_output": self.parse_test_output,
            "bug_patch": self.bug_patch,
            "test_patch": self.test_patch,
            "repo_path": self.repo_path,
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BugArtifact":
        """Create from dictionary."""
        return cls(
            test_files=data.get("test_files", []),
            test_script=data.get("test_script", ""),
            parse_test_output=data.get("parse_test_output", ""),
            bug_patch=data.get("bug_patch", ""),
            test_patch=data.get("test_patch", ""),
            repo_path=data.get("repo_path", ""),
            instance_id=data.get("instance_id", ""),
        )

    @classmethod
    def from_submission(cls, submission: str, repo_path: str = "") -> "BugArtifact":
        """Parse bug artifact from submission format."""
        artifact = cls(repo_path=repo_path)

        # Extract files from submission format
        # Format: <tool: submit>\nfile1\nfile2\n...\n</tool>
        files_match = re.search(r"<tool:\s*submit>\s*(.*?)\s*</tool>", submission, re.DOTALL)
        if not files_match:
            return artifact

        files_content = files_match.group(1).strip()
        lines = [l.strip() for l in files_content.split("\n") if l.strip()]

        # Map known filenames
        for line in lines:
            if line.endswith("test_files.txt"):
                artifact._read_test_files(line, repo_path)
            elif line.endswith("test_script.sh"):
                artifact.test_script = _read_file(os.path.join(repo_path, line))
            elif line.endswith("parse_test_output.py"):
                artifact.parse_test_output = _read_file(os.path.join(repo_path, line))
            elif line.endswith("bug_patch.diff"):
                artifact.bug_patch = _read_file(os.path.join(repo_path, line))
            elif line.endswith("test_patch.diff"):
                artifact.test_patch = _read_file(os.path.join(repo_path, line))

        return artifact

    def _read_test_files(self, filepath: str, repo_path: str) -> None:
        """Read test files list from file."""
        content = _read_file(os.path.join(repo_path, filepath))
        self.test_files = [l.strip() for l in content.split("\n") if l.strip()]

    def get_oracle_test_patch(self) -> str:
        """Get the reverse of test_patch (oracle tests).

        This is used as input to the solver agent.
        """
        if not self.test_patch:
            return ""

        # Reverse the patch by swapping + and - lines
        lines = self.test_patch.split("\n")
        reversed_lines = []
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                reversed_lines.append("-" + line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                reversed_lines.append("+" + line[1:])
            elif line.startswith("@@"):
                # Swap the line numbers in hunk header
                # @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)", line)
                if match:
                    old_start = match.group(1)
                    old_count = match.group(2) or ""
                    new_start = match.group(3)
                    new_count = match.group(4) or ""
                    context = match.group(5)
                    old_spec = f"-{new_start}" + (f",{new_count}" if new_count else "")
                    new_spec = f"+{old_start}" + (f",{old_count}" if old_count else "")
                    reversed_lines.append(f"@@ {old_spec} {new_spec} @@{context}")
                else:
                    reversed_lines.append(line)
            elif line.startswith("---"):
                reversed_lines.append("+++" + line[3:])
            elif line.startswith("+++"):
                reversed_lines.append("---" + line[3:])
            else:
                reversed_lines.append(line)

        return "\n".join(reversed_lines)

    def get_files_touched_by_patch(self, patch: str) -> set[str]:
        """Extract file paths touched by a patch."""
        files = set()
        for line in patch.split("\n"):
            # Match diff headers like: diff --git a/path/to/file b/path/to/file
            if line.startswith("diff --git"):
                match = re.search(r"diff --git a/(.*?) b/(.*)", line)
                if match:
                    files.add(match.group(2))
            # Match --- a/path or +++ b/path
            elif line.startswith("--- a/") or line.startswith("+++ b/"):
                path = line[6:].strip()
                if path and path != "/dev/null":
                    files.add(path)
        return files

    def get_code_files_touched(self) -> set[str]:
        """Get code files touched by bug_patch."""
        return self.get_files_touched_by_patch(self.bug_patch)

    def get_test_files_touched(self) -> set[str]:
        """Get test files touched by test_patch."""
        return self.get_files_touched_by_patch(self.test_patch)

    def save(self, directory: str) -> None:
        """Save artifact files to directory."""
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "test_files.txt"), "w") as f:
            f.write("\n".join(self.test_files))

        with open(os.path.join(directory, "test_script.sh"), "w") as f:
            f.write(self.test_script)

        with open(os.path.join(directory, "parse_test_output.py"), "w") as f:
            f.write(self.parse_test_output)

        with open(os.path.join(directory, "bug_patch.diff"), "w") as f:
            f.write(self.bug_patch)

        with open(os.path.join(directory, "test_patch.diff"), "w") as f:
            f.write(self.test_patch)

    @classmethod
    def load(cls, directory: str) -> "BugArtifact":
        """Load artifact from directory."""
        artifact = cls()
        artifact.test_files = _read_file(os.path.join(directory, "test_files.txt")).strip().split("\n")
        artifact.test_script = _read_file(os.path.join(directory, "test_script.sh"))
        artifact.parse_test_output = _read_file(os.path.join(directory, "parse_test_output.py"))
        artifact.bug_patch = _read_file(os.path.join(directory, "bug_patch.diff"))
        artifact.test_patch = _read_file(os.path.join(directory, "test_patch.diff"))
        return artifact


def _read_file(path: str) -> str:
    """Read file contents or return empty string if not found."""
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return ""


@dataclass
class ValidationResult:
    """Result of bug artifact validation."""

    valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Validation details
    test_files_exist: bool = False
    parser_valid: bool = False
    test_script_valid: bool = False
    bug_scope_valid: bool = False
    bug_validity: bool = False
    test_weakening_valid: bool = False
    file_necessity_valid: bool = False

    # Test results
    original_passing_tests: int = 0
    original_failing_tests: int = 0
    buggy_passing_tests: int = 0
    buggy_failing_tests: int = 0
    weakened_passing_tests: int = 0
    weakened_failing_tests: int = 0


def validate_bug_artifact(
    artifact: BugArtifact,
    sandbox,
    min_passing_tests: int = 5,
    min_changed_files: int = 1,
    min_failing_tests: int = 1,
    check_file_necessity: bool = True,
) -> ValidationResult:
    """
    Validate a bug artifact according to SSR consistency checks.

    Checks:
    1. Test files existence & coverage
    2. Parser validity
    3. Test script validity (>= min_passing_tests on original repo)
    4. Bug scope (>= min_changed_files modified)
    5. Bug validity (>= min_failing_tests fail after bug)
    6. Test weakening validity (some failing tests become passing)
    7. File necessity (each modified file is necessary for at least one test to fail)

    Args:
        artifact: The bug artifact to validate
        sandbox: DockerSandbox instance for running tests
        min_passing_tests: Minimum tests passing on original repo
        min_changed_files: Minimum code files modified by bug
        min_failing_tests: Minimum tests failing after bug
        check_file_necessity: Whether to run inverse mutation testing

    Returns:
        ValidationResult with validation status and details
    """
    result = ValidationResult()

    # 1. Test files existence & coverage
    if not artifact.test_files:
        result.errors.append("No test files specified")
    else:
        # Check files touched by test_patch are subset of test_files
        test_patch_files = artifact.get_test_files_touched()
        test_files_set = set(artifact.test_files)
        if not test_patch_files.issubset(test_files_set):
            missing = test_patch_files - test_files_set
            result.errors.append(f"test_patch.diff touches files not in test_files.txt: {missing}")
        else:
            result.test_files_exist = True

    # 2. Parser validity - check it can run and produces JSON
    if not artifact.parse_test_output:
        result.errors.append("No parse_test_output.py script")
    else:
        try:
            # Test parser with dummy input
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(artifact.parse_test_output)
                parser_path = f.name

            # Run parser with empty input to check syntax
            proc = subprocess.run(
                ["python3", parser_path],
                input="",
                capture_output=True,
                text=True,
                timeout=10,
            )
            os.unlink(parser_path)

            # Parser should run without errors (may output empty JSON)
            result.parser_valid = True
        except Exception as e:
            result.errors.append(f"Parser validation failed: {e}")

    # 3. Test script validity - run on original repo
    if not artifact.test_script:
        result.errors.append("No test_script.sh")
    elif sandbox is not None:
        try:
            test_output = sandbox.run_tests(artifact.test_script, timeout=90)
            test_results = sandbox.parse_test_output(artifact.parse_test_output, test_output)

            result.original_passing_tests = sum(1 for v in test_results.values() if v == "passed")
            result.original_failing_tests = sum(1 for v in test_results.values() if v == "failed")

            if result.original_passing_tests >= min_passing_tests:
                result.test_script_valid = True
            else:
                result.errors.append(
                    f"Only {result.original_passing_tests} tests passing on original repo "
                    f"(need {min_passing_tests})"
                )
        except Exception as e:
            result.errors.append(f"Test script validation failed: {e}")

    # 4. Bug scope - check modified files count
    code_files = artifact.get_code_files_touched()
    if len(code_files) >= min_changed_files:
        result.bug_scope_valid = True
    else:
        result.errors.append(
            f"Only {len(code_files)} code files modified (need {min_changed_files})"
        )

    # Check bug patch doesn't touch test files
    for f in code_files:
        if "test" in f.lower():
            result.warnings.append(f"Bug patch may touch test file: {f}")

    # 5. Bug validity - apply bug patch and check tests fail
    if sandbox is not None and result.test_script_valid:
        try:
            sandbox.apply_patch(artifact.bug_patch)
            test_output = sandbox.run_tests(artifact.test_script, timeout=90)
            test_results = sandbox.parse_test_output(artifact.parse_test_output, test_output)

            result.buggy_passing_tests = sum(1 for v in test_results.values() if v == "passed")
            result.buggy_failing_tests = sum(1 for v in test_results.values() if v == "failed")

            if result.buggy_failing_tests >= min_failing_tests:
                result.bug_validity = True
            else:
                result.errors.append(
                    f"Only {result.buggy_failing_tests} tests fail after bug "
                    f"(need {min_failing_tests})"
                )
        except Exception as e:
            result.errors.append(f"Bug validity check failed: {e}")

    # 6. Test weakening validity - apply test patch and check some failures hidden
    if sandbox is not None and result.bug_validity:
        try:
            sandbox.apply_patch(artifact.test_patch)
            test_output = sandbox.run_tests(artifact.test_script, timeout=90)
            test_results = sandbox.parse_test_output(artifact.parse_test_output, test_output)

            result.weakened_passing_tests = sum(1 for v in test_results.values() if v == "passed")
            result.weakened_failing_tests = sum(1 for v in test_results.values() if v == "failed")

            if result.weakened_failing_tests < result.buggy_failing_tests:
                result.test_weakening_valid = True
            else:
                result.errors.append("Test weakening did not hide any failing tests")
        except Exception as e:
            result.errors.append(f"Test weakening check failed: {e}")

    # 7. File necessity (inverse mutation testing)
    if check_file_necessity and sandbox is not None and result.bug_validity:
        try:
            # For each modified code file, check if reverting it fixes at least one test
            failing_tests = set(
                t for t, v in sandbox.get_last_test_results().items() if v == "failed"
            )

            all_files_necessary = True
            for code_file in code_files:
                # Reset to buggy state
                sandbox.reset()
                sandbox.apply_patch(artifact.bug_patch)

                # Revert just this file
                sandbox.revert_file(code_file)

                # Run tests
                test_output = sandbox.run_tests(artifact.test_script, timeout=90)
                test_results = sandbox.parse_test_output(artifact.parse_test_output, test_output)

                # Check if at least one previously failing test now passes
                current_failing = set(t for t, v in test_results.items() if v == "failed")
                fixed_tests = failing_tests - current_failing

                if not fixed_tests:
                    result.warnings.append(f"File {code_file} may not be necessary for any test")
                    all_files_necessary = False

            result.file_necessity_valid = all_files_necessary
        except Exception as e:
            result.warnings.append(f"File necessity check failed: {e}")
            result.file_necessity_valid = True  # Don't fail on this check

    # Determine overall validity
    result.valid = (
        result.test_files_exist
        and result.parser_valid
        and result.test_script_valid
        and result.bug_scope_valid
        and result.bug_validity
        and result.test_weakening_valid
    )

    return result
