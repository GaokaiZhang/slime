Notice:

0. we have docker access; and for data, we can pick any SWE-Bench data, or mostly SWE-Bench Verified django data which you can find at /home/gaokaizhang/SWE-sft

1. for GPU, use modal, I have already modal setup at the workspace susvibes-mitigation; but be sure to shutdown GPU when you're done testing

2. for models and huggingface, use the token HF_TOKEN_PLACEHOLDER and the model Kwai-Klear/Klear-AgentForge-8B-SFT

3. for environment use "slime" (we should already be inside), and download/upgrade packages if needed

4. for evaluation (swebench.harness) usage, check /home/gaokaizhang/SWE-sft where you can copy code here

5. for original prompts used for SSR (and we should use it), check bug_injection_prompts.txt and bug_solving_prompts.txt

6. work all the way and do not ask me for any choices until you have replicate the mentioned framework and ran RL test with actual rollouts; make sure the agents are working correctly in the docker (getting the response/environment correct)

7. write a STATUS.md keep track of all codes we have added to slime/our progress; compact and continue when needed


# SSR on SLiME: Agent + CLI Guide (Replication-Oriented)

This document describes how to replicate **Self-play SWE-RL (SSR)** (arXiv:2512.18552) on top of **SLiME** with **SWE-bench**-style sandboxed repos.

> Core SSR idea:
> - One policy model, instantiated into two roles via prompting:
>   (1) Bug Injector (proposer) and (2) Bug Solver (repair agent).
> - Both roles share weights and are jointly updated with RL.
> - Bugs are specified by a *bug artifact* (patches + test runner + parser), not natural-language issues.

---

## 0. Prerequisites

- You already cloned:
  - SLiME repo
  - SWE-bench evaluation / environment tooling
- You have a pool of sandboxed repo environments:
  - Ideally SWE-bench compatible docker images, or your own docker/apptainer images
- You have an inference backend (vLLM/SGLang/Transformers) usable by SLiME.

---

## 1. SSR Roles

### 1.1 Bug Injector (Proposer)
Goal: interact with a repo sandbox to produce a **bug artifact** that:
1) introduces a semantic bug via code patch, and
2) hides (partially) the bug by weakening tests, and
3) includes a runnable test command + a parser.

### 1.2 Bug Solver (Repair agent)
Goal: given the constructed buggy repo and an "oracle test specification" (reverse of the weakening patch),
produce a **prediction patch** that makes oracle tests pass.

---

## 2. Bug Artifact Format (Required)

A bug artifact is a directory containing:

- `test_script.sh`
  - Runs a selected test subset with verbose per-test output.
  - Must run within ~90 seconds.

- `test_files.txt`
  - List of oracle test files (relative paths), one per line.
  - During evaluation we reset these files to original versions to prevent test-gaming.

- `parse_test_output.py`
  - Reads raw test output from stdin; writes JSON to stdout:
    `{ "test_id": "passed"|"failed", ... }`

- `bug_patch.diff`
  - Git diff patch that introduces the bug.
  - **Must modify code files only** (no tests).

- `test_patch.diff`
  - Git diff patch that weakens/removes tests.
  - **Must modify test files only** (no code).

---

## 3. Consistency Validation (Must Implement)

Given a proposed bug artifact, run deterministic checks:

1) **Test files existence & coverage**
   - `test_files.txt` files exist in repo.
   - `test_files.txt` is a superset of files touched by `test_patch.diff`.

2) **Parser validity**
   - `parse_test_output.py` reliably converts test output to JSON mapping.

3) **Test script validity**
   - On the original repo, `bash test_script.sh` => at least `min_passing_tests` passing tests.

4) **Bug scope**
   - `bug_patch.diff` modifies at least `min_changed_files` code files.

5) **Bug validity**
   - After applying `bug_patch.diff`, at least `min_failing_tests` previously passing tests fail.

6) **Test weakening validity**
   - After applying `test_patch.diff`, some failing tests become passing.

7) **Inverse mutation testing (file necessity)**
   - Let `F` be failing tests under full bug.
   - For each modified code file `f`:
     - Reset to full buggy state,
     - revert only `f` to fixed version,
     - run oracle (non-weakened) tests,
     - Require that at least one test in `F` flips to pass; otherwise reject artifact.

Only artifacts passing all checks are considered **valid**.

---

## 4. Constructing the Buggy Repo (Solver World)

Starting from the original repo:

1) Apply `bug_patch.diff` (introduce bug)
2) Apply `test_patch.diff` (hide bug)
3) (Optional for higher-order bug) apply a previous failed solver patch: `pred_patch.diff`
4) Prevent git leakage:
   - remove `.git`, re-init and create a fresh commit

---

## 5. Higher-order Bugs

When a solver attempt fails, save the solver output patch as `pred_patch.diff`.
Create a second-order bug by applying that patch on top of the first-order buggy repo.

Limit to second-order bugs (do not go beyond 2nd order).

---

## 6. Rewards

### 6.1 Solver Reward (terminal, per attempt)
- `+1` if **all oracle tests pass**
- `-1` otherwise

### 6.2 Injector Reward (per bug artifact)
Let `s` be the solve rate = (#successful solver attempts) / (group_size).

- `-1.0` if consistency validation fails
- `-alpha` if `s == 0` or `s == 1`
- `1 - (1 + alpha) * s` if `0 < s < 1`

Recommended paper defaults:
- `alpha = 0.8`
- `group_size = 8`

---

## 7. Prompt Templates (Paraphrased, Replication-Grade)

### 7.1 Injector: Removal-oriented (Removal-only)

High-level requirements:
- Discover how to run tests; choose a test subset with >= `min_passing_tests`.
- Write `test_script.sh` (verbose per-test output; <90s).
- Write `parse_test_output.py` (stdin -> JSON `test_id -> status`).
- Remove substantial code from >= `min_changed_files` code files (code only).
- Ensure some tests fail after bug injection; avoid trivial syntax-break-everything.
- Export `bug_patch.diff` (code-only).
- Weaken/remove tests to hide the bug; do not comment out tests; modify tests only.
- Export `test_patch.diff` (tests-only).
- Submit 5 files via `<tool: submit>`.

Hard constraints:
- No hint comments about bug injection.
- No orphan modified code files (every modified code file must be exercised by selected tests).

### 7.2 Injector: History-aware (Removal + history)

Same as above except:
- Introduce bugs by selectively reverting historical changes using `git log/show/restore`.
- Apply minimal compatibility fixes to keep repo runnable while preserving historical bug patterns.
- Then weaken tests to hide bug.

### 7.3 Injector: Direct injection

Same required outputs; allows any bug injection method, but must satisfy:
- >= `min_passing_tests` before injection
- >= `min_changed_files` code files modified
- >= `min_num_tests_to_break` failing tests after injection (if configured)
- weaken tests afterward

### 7.4 Solver: Fixed-template issue description from oracle test patch

Use a fixed issue preamble:
- "I am improving the test suite with the following changes, but current code fails. Please fix code and ensure relevant tests pass."
- Then include:
  ```diff
  {oracle_test_patch}   # reverse of test_patch.diff (restricted to oracle test files)
