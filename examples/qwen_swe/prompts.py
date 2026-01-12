"""
Prompt templates for SWE-bench bug solving with qwen-code agent.

These prompts are based on the SSR (Self-play SWE-RL) paper approach,
adapted for use with the qwen-code CLI agent.
"""

# System prompt for qwen-code agent (via --system-prompt flag)
SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing bugs in a repository.

Instructions:
1. First, explore the repository structure to understand the codebase
2. Find the relevant files related to the bug
3. Understand the issue and identify the root cause
4. Implement a fix for the bug
5. Verify your fix doesn't break existing functionality using tests

The repository is located at /testbed. Use bash commands to explore, read files, and make changes.

When you're done fixing the bug, use `git diff` to show your changes, then confirm you are done.
"""

# Task prompt template (with issue description)
TASK_PROMPT_TEMPLATE = """## Problem Statement
{problem_statement}

## Instructions
Fix the bug described above. The repository is at /testbed.

Steps:
1. Explore the repository structure
2. Find relevant files
3. Understand and fix the bug
4. Test your fix with: python -m pytest <relevant_test_file> -x
5. Run `git diff` to show your changes

Begin by exploring the repository structure.
"""

# Bug solving prompt with oracle test patch (for SSR-style training)
BUG_SOLVING_WITH_ORACLE_TEMPLATE = """I am improving the test suite of the project with the following changes, but the current code does not pass the tests. Please fix the code. If any existing tests relevant to the functionality being changed are failing, please make sure your patch passes those tests as well.

```diff
{oracle_test_patch}
```

## Instructions
Solve the issue by implementing the necessary code changes and submitting a patch file.

The repository is located at /testbed. Begin exploring and fixing the bug!
"""

# Simple SWE-bench task prompt (used in run_swebench.py)
SWEBENCH_PROMPT_TEMPLATE = """You are an expert software engineer tasked with fixing a bug in a repository.

## Problem Statement
{problem_statement}

## Instructions
1. First, explore the repository structure to understand the codebase
2. Find the relevant files related to the bug
3. Understand the issue and identify the root cause
4. Implement a fix for the bug
5. Verify your fix doesn't break existing functionality

When you're done, use `git diff` to show your changes.

The repository is located at /testbed. Begin exploring and fixing the bug!
"""


def format_task_prompt(problem_statement: str) -> str:
    """Format task prompt with problem statement."""
    return TASK_PROMPT_TEMPLATE.format(problem_statement=problem_statement)


def format_bug_solving_prompt(oracle_test_patch: str) -> str:
    """Format bug solving prompt with oracle test patch."""
    return BUG_SOLVING_WITH_ORACLE_TEMPLATE.format(oracle_test_patch=oracle_test_patch)


def format_swebench_prompt(problem_statement: str) -> str:
    """Format standard SWE-bench prompt."""
    return SWEBENCH_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
