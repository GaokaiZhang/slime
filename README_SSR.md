# SSR (Self-play SWE-RL) Implementation on SLiME

This document describes the implementation of **Self-play SWE-RL (SSR)** (arXiv:2512.18552) on top of **SLiME** using **SWE-bench** repositories.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Roles and Visibility](#roles-and-visibility)
4. [Repository States](#repository-states)
5. [RL Training Loop](#rl-training-loop)
6. [Reward Structure](#reward-structure)
7. [Bug Artifact Format](#bug-artifact-format)
8. [Implementation Details](#implementation-details)
9. [Commands](#commands)
10. [Configuration](#configuration)

---

## Overview

SSR is a self-play reinforcement learning framework where **one policy model plays two roles** against itself:

1. **Bug Injector** - Creates bugs in clean code
2. **Bug Solver** - Fixes bugs given oracle tests

Both roles share weights and are jointly updated with RL, creating an automatic curriculum where the injector learns to create appropriately challenging bugs while the solver learns to fix them.

### Key Insight

The brilliance of SSR is self-improvement:
- As the solver gets better → injectors must create harder bugs
- As injectors create harder bugs → solvers must become more capable
- Both roles share weights → improvements transfer bidirectionally

---

## Core Concepts

### One-Shot Self-Play Semantics

1. **No iterative refinement**: Bug Injector generates the bug exactly once per artifact
2. **Multiple solver attempts**: Bug Solver attempts to fix the same bug independently for `group_size=8` rollouts
3. **No back-and-forth**: No interaction between injector and solver on the same bug artifact

### Reward Temporal Granularity

| Role | Reward Timing | Reward Type |
|------|---------------|-------------|
| **Solver** | Immediate (per rollout) | Binary: +1 (pass) or -1 (fail) |
| **Injector** | Delayed (after all solvers done) | Scalar based on solve_rate |

```
Injector generates bug
        │
        ├─→ Solver attempt 1 → reward +1 or -1 (immediate)
        ├─→ Solver attempt 2 → reward +1 or -1 (immediate)
        ├─→ ...
        └─→ Solver attempt 8 → reward +1 or -1 (immediate)
                                      │
                                      ↓
                              solve_rate = successes/8
                                      │
                                      ↓
                    Injector reward = f(solve_rate) (delayed)
```

---

## Roles and Visibility

### Bug Injector (Proposer)

**What it CAN see:**
- Full access to a **clean, fully passing** repository
- Complete repository structure and test suite
- Git commit history (for history-based bug injection)
- Ability to run tests and explore code

**What it does:**
1. Explores the repo to understand codebase and tests
2. Selects a subset of tests (≥5 passing tests)
3. Creates `test_script.sh` to run selected tests (<90s timeout)
4. Creates `parse_test_output.py` to parse test results
5. Injects a semantic bug by modifying code files (≥1 file)
6. Creates `bug_patch.diff` (code-only changes)
7. Weakens/removes tests to hide the bug
8. Creates `test_patch.diff` (test-only changes)
9. Submits 5-file bug artifact

**Goal:** Create a bug that is *solvable but not trivial* (optimal solve rate: 10-50%)

### Bug Solver (Repair Agent)

**What it CAN see:**
- The **buggy repository** (with bug_patch + test_patch applied)
- The **oracle test patch** (reverse of test_patch.diff)
- Ability to run tests and explore buggy code

**What it CANNOT see:**
- The clean/original repository
- The original `bug_patch.diff`
- Any git history (wiped to prevent leakage)

**What it does:**
1. Reads oracle test patch to understand expected behavior
2. Explores buggy codebase to diagnose the issue
3. Implements a fix
4. Submits `pred_patch.diff`

**Goal:** Make all oracle tests pass.

---

## Repository States

### SWE-bench Docker Images

**Critical Understanding:** SWE-bench Docker images contain the **ORIGINAL BUG** from the benchmark.

For SSR bug injection to work correctly, we must first apply the **gold patch** to get a clean baseline:

```
SWE-bench Docker Image
        │
        │ (contains original SWE-bench bug)
        ↓
    Apply GOLD PATCH  ← This is instance.patch in our data
        │
        │ (now have clean, passing codebase)
        ↓
    CLEAN STATE (ready for bug injection)
```

### State Transitions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INJECTOR'S WORLD (Clean State)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Docker image → Apply gold_patch → CLEAN REPO (all tests pass)         │
│                                          │                              │
│                              Injector injects bug                       │
│                                          │                              │
│                                          ↓                              │
│                              bug_patch.diff created                     │
│                              test_patch.diff created                    │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOLVER'S WORLD (Buggy State)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  CLEAN REPO                                                             │
│      │                                                                  │
│      ├── Apply bug_patch.diff  (introduce bug)                          │
│      ├── Apply test_patch.diff (hide bug)                               │
│      └── Wipe .git directory   (prevent leakage)                        │
│              │                                                          │
│              ↓                                                          │
│         BUGGY REPO (some tests hidden, bug present)                     │
│              │                                                          │
│              │ Solver receives:                                         │
│              │  - Buggy repo                                            │
│              │  - Oracle test patch (reverse of test_patch)             │
│              │                                                          │
│              ↓                                                          │
│         Solver produces pred_patch.diff                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Apply pred_patch.diff to buggy repo                                    │
│  Revert test_patch.diff (restore oracle tests)                          │
│  Run oracle tests                                                       │
│      │                                                                  │
│      ├── All pass → Solver reward = +1                                  │
│      └── Any fail → Solver reward = -1                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## RL Training Loop

### Single Training Round

```
Step 0: Load clean repo from SWE-bench
        ┌─────────────────────────────────────────┐
        │  Docker image (with original bug)       │
        │  + Apply gold_patch                     │
        │  = CLEAN REPO (all tests pass)          │
        └─────────────────────────────────────────┘
                              │
                              ↓
Step 1: Bug Injector generates bug artifact
        ┌─────────────────────────────────────────┐
        │  Injector (Policy Model + Prompt A)     │
        │  - Explores clean repo                  │
        │  - Creates bug_patch.diff               │
        │  - Creates test_patch.diff              │
        │  - Creates test infrastructure          │
        │  → Outputs: 5-file bug artifact         │
        └─────────────────────────────────────────┘
                              │
                              ↓
Step 2: Consistency Validation (deterministic)
        - Test files exist?
        - Parser works?
        - Tests pass before bug?
        - Tests fail after bug?
        - Test weakening hides bug?
        - Each modified file necessary? (inverse mutation)

        If FAIL → Injector reward = -1.0, skip to Step 5
                              │
                              ↓
Step 3: Construct buggy repo for solver
        - Apply bug_patch.diff
        - Apply test_patch.diff
        - Wipe .git history
                              │
                              ↓
Step 4: Solver attempts (group_size = 8)
        ┌─────────────────────────────────────────┐
        │  Solver (Policy Model + Prompt B)       │
        │  - Sees buggy repo + oracle test patch  │
        │  - Explores and diagnoses               │
        │  - Produces pred_patch.diff             │
        └─────────────────────────────────────────┘

        For each of 8 attempts:
          - Apply pred_patch.diff
          - Revert test_patch.diff (restore oracle tests)
          - Run tests
          - Solver reward: +1 if all pass, -1 otherwise
                              │
                              ↓
Step 5: Compute Injector reward
        s = (# successful solves) / 8

        If validation failed:    reward = -1.0
        If s == 0 or s == 1:     reward = -α     (α = 0.8)
        If 0 < s < 1:            reward = 1 - (1 + α) * s
                              │
                              ↓
Step 6: Update shared policy weights
        - Injector trajectory contributes gradients
        - All 8 solver trajectories contribute gradients
        - Single model learns from both perspectives
```

---

## Reward Structure

### Solver Reward (Immediate, per rollout)

| Condition | Reward |
|-----------|--------|
| All oracle tests pass | **+1** |
| Any test fails | **-1** |

### Injector Reward (Delayed, per bug artifact)

Let `s = solve_rate = (# successful solver attempts) / group_size`

| Condition | Reward | Interpretation |
|-----------|--------|----------------|
| Validation fails | **-1.0** | Invalid bug artifact |
| s = 0 | **-α** (-0.8) | Bug too hard |
| s = 1 | **-α** (-0.8) | Bug too easy |
| 0 < s < 1 | **1 - (1+α)*s** | Good difficulty |

### Reward Landscape (α = 0.8)

```
Solve Rate   Injector Reward   Quality
───────────────────────────────────────
0.000        -0.800            TOO HARD
0.125        +0.775            OPTIMAL ⭐
0.250        +0.550            GOOD
0.375        +0.325            GOOD
0.500        +0.100            MODERATE
0.625        -0.125            EASY
0.750        -0.350            TOO EASY
0.875        -0.575            TOO EASY
1.000        -0.800            TOO EASY
```

**Optimal zone:** 10-50% solve rate maximizes injector reward.

---

## Bug Artifact Format

A valid bug artifact contains exactly 5 files:

| File | Purpose | Constraints |
|------|---------|-------------|
| `test_script.sh` | Runs selected tests | <90s timeout, verbose output |
| `test_files.txt` | Lists oracle test files | One path per line |
| `parse_test_output.py` | Parses test output | stdin → JSON `{test_id: "passed"\|"failed"}` |
| `bug_patch.diff` | Introduces the bug | **Code files only** (≥1 file modified) |
| `test_patch.diff` | Hides the bug | **Test files only** |

### Validation Checks (All must pass)

1. **Test files existence** - Files in `test_files.txt` exist
2. **Coverage** - `test_files.txt` ⊇ files touched by `test_patch.diff`
3. **Parser validity** - `parse_test_output.py` produces valid JSON
4. **Test script validity** - ≥5 tests pass on clean repo
5. **Bug scope** - `bug_patch.diff` modifies ≥1 code file
6. **Bug validity** - ≥1 test fails after applying `bug_patch.diff`
7. **Test weakening validity** - Some failures hidden by `test_patch.diff`
8. **File necessity (inverse mutation)** - Each modified code file is exercised by failing tests

---

## Implementation Details

### Files Added to SLiME

```
examples/ssr/
├── __init__.py           # Module exports with lazy imports
├── prompts.py            # Bug injection & solving prompt templates
├── bug_artifact.py       # Artifact format & 7-point validation
├── rewards.py            # SSR reward functions
├── docker_sandbox.py     # SWE-bench docker container management
├── data_source.py        # Data loading for SLiME integration
├── rollout.py            # Custom generate function for multi-turn agents
├── run_ssr.py            # Training script with tests
├── modal_ssr.py          # Modal GPU deployment (A100)
├── test_rollout.py       # End-to-end rollout tests
├── gpu_inference.py      # Unified GPU inference (local + Modal + SGLang)
└── swebench_harness.py   # SWE-bench evaluation harness integration
```

### Key Implementation Points

**Gold Patch Application:**
```python
# data_source.py:205 - Store gold_patch in metadata
sample.metadata["gold_patch"] = instance.patch

# rollout.py:149 - Extract gold_patch
gold_patch = sample.metadata.get("gold_patch", "")

# rollout.py:178 - Apply during sandbox start
await sandbox.start(instance_id=instance_id, gold_patch=gold_patch)

# docker_sandbox.py:123-129 - Apply patch in container
if gold_patch:
    self.apply_patch(gold_patch)  # Fixes original bug FIRST
```

**Solver's Buggy State Construction:**
```python
# rollout.py:323-325
await sandbox.apply_patch(bug_artifact.bug_patch)   # Introduce bug
await sandbox.apply_patch(bug_artifact.test_patch)  # Hide bug with weakened tests
```

**Oracle Test Evaluation:**
```python
# rollout.py:391-397
await sandbox.apply_patch(bug_artifact.test_patch, reverse=True)  # Restore oracle tests
all_pass, test_results = await sandbox.check_tests_pass(...)
```

---

## Commands

### Local Tests (No GPU)

```bash
# Run unit tests
python examples/ssr/run_ssr.py --test_mode

# Run with docker sandbox test
python examples/ssr/run_ssr.py

# Run end-to-end rollout test
python examples/ssr/test_rollout.py
```

### Modal GPU Tests

```bash
# Test reward functions
modal run examples/ssr/modal_ssr.py --action artifact

# Test GPU inference
modal run examples/ssr/modal_ssr.py --action inference

# Full SSR rollout test
modal run examples/ssr/modal_ssr.py --action ssr_rollout

# Multi-turn generation test
modal run examples/ssr/modal_ssr.py --action multi_turn

# Run all tests
modal run examples/ssr/modal_ssr.py --action all
```

### Full RL Training

```bash
python -m slime.ray.main \
    --hf_checkpoint Kwai-Klear/Klear-AgentForge-8B-SFT \
    --data_source_path examples.ssr.data_source:create_data_source \
    --rollout_function_path examples.ssr.rollout:generate \
    --ssr_data_path /home/gaokaizhang/SWE-sft/data/sft/train.jsonl \
    --ssr_agent_type both \
    --advantage_estimator grpo \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 8
```

---

## Configuration

### Environment

| Setting | Value |
|---------|-------|
| **Model** | `Kwai-Klear/Klear-AgentForge-8B-SFT` |
| **HF Token** | `HF_TOKEN_PLACEHOLDER` |
| **Data Path** | `/home/gaokaizhang/SWE-sft/data/sft/train.jsonl` |
| **Modal Workspace** | `susvibes-mitigation` |
| **Docker Images** | SWE-bench verified django images |

### SSR Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Number of solver attempts per bug |
| `alpha` | 0.8 | Edge case penalty |
| `max_turns` | 32 | Max agent interaction turns |
| `test_timeout` | 90 | Test execution timeout (seconds) |
| `min_passing_tests` | 5 | Minimum tests for valid artifact |
| `min_changed_files` | 1 | Minimum code files to modify |

### Injector Types

| Type | Description |
|------|-------------|
| `removal` | Remove substantial code (default) |
| `history` | Revert historical changes using git |
| `direct` | Any bug injection method |

---

## Test Results

### Verified Components (2026-01-04)

```
GPU Test:
  CUDA available: True
  GPU: NVIDIA A100 80GB PCIe
  Model: Kwai-Klear/Klear-AgentForge-8B-SFT (qwen3)
  Status: SUCCESS

Rollout Tests:
  [PASS] Docker Sandbox (8.36s)
  [PASS] Rollout Logic (0.00s)
  [PASS] Data Source (1.12s)
  Total: 3/3 passed

Gold Patch Verification:
  django-16255: Tests FAIL → PASS after gold patch ✓
  django-16256: Tests PASS → PASS after gold patch ✓
```

### Multi-Instance Analysis

| Instance | Solve Rate | Injector Reward | Category |
|----------|------------|-----------------|----------|
| django-16255 | 8/8 (1.00) | -0.80 | TOO EASY |
| django-10554 | 0/8 (0.00) | -0.80 | TOO HARD |
| django-10097 | 1/8 (0.12) | **+0.78** | OPTIMAL |
| django-10914 | 5/8 (0.62) | -0.12 | MODERATE |

---

## Bug Injector Testing Results (2026-01-05)

### Model Tested

- **Model:** `Kwai-Klear/Klear-AgentForge-8B-SFT`
- **Infrastructure:** Modal A100 GPU + Local Docker
- **Prompt Format:** Mini-SWE-agent (markdown code blocks)

### Instances Tested

| Instance | Success | Turns | Time | Notes |
|----------|---------|-------|------|-------|
| django__django-16256 | ✗ | 30 | ~270s | Old XML format, failed |
| django__django-16255 | ✗ | 60 | ~657s | Got stuck in code loops |
| django__django-16139 | ✗ | 80 | ~1071s | Made progress but didn't submit |
| django__django-16595 | **✓** | 85 | ~1406s | **SUCCESS** - Generated bug |
| django__django-16877 | ✗ | 100 | ~1287s | Database setup issues |

**Total Instances Tested:** 5
**Success Rate:** 1/5 (20%)

### Artifact Completion Analysis

**No test has generated a complete 5-file SSR artifact.**

| Test File | Instance | Files | Missing |
|-----------|----------|-------|---------|
| `ssr_protocol_results.json` | django-16255 | **4/5** | `test_patch.diff` |
| `modal_injector_django_django-16595.json` | django-16595 | **3/5** | `test_files.txt`, `test_patch.diff` |
| All others | various | 0/5 | Failed completely |

**Note:** "Success" in the table above means the model generated a **working bug** that breaks tests. However, none achieved a **complete SSR artifact** (all 5 files). The `test_patch.diff` (test weakening) is **consistently missing** across all attempts.

### Why `test_patch.diff` Is Missing

The prompts include explicit instructions for test weakening (steps 10-11), but:

1. **Turn limit reached** - Model takes 85-100 turns just to create the bug, runs out before test weakening
2. **Task complexity** - Creating bug + weakening tests is a 2-phase task; model focuses on phase 1
3. **Early submission** - Model submits after creating `bug_patch.diff` without completing test weakening
4. **Context length** - By the time bug is created, context is very long

### Successful Bug Example (django-16595)

**Bug Type:** Function removal + signature change in `django/shortcuts.py`

**Original Code:**
```python
def redirect(to, *args, permanent=False, **kwargs):
    """Redirect to URL, view name, or model"""
    redirect_class = HttpResponsePermanentRedirect if permanent else HttpResponseRedirect
    return redirect_class(resolve_url(to, *args, **kwargs))

def get_object_or_404(klass, *args, **kwargs):
    """Get object or raise Http404"""
    ...
```

**Buggy Code (injector's version):**
```python
def redirect(request, url, permanent=False):  # WRONG signature!
    return HttpResponseRedirect(url)

# get_object_or_404 - REMOVED!
# get_list_or_404 - REMOVED!
# resolve_url - REMOVED!
```

**Test Failure:**
```
✗ ImportError: cannot import name 'get_object_or_404' from 'django.shortcuts'
✗ redirect signature: (request, url, permanent=False)
  Should be: (to, *args, permanent=False, **kwargs)
```

### Key Findings

**Model Strengths:**
- Can follow multi-step bug injection protocol
- Understands Django test framework (runtests.py)
- Generates **substantial semantic bugs** (not just syntax errors)
- Uses markdown code block format correctly

**Issues Observed:**
- Gets stuck outputting empty code blocks
- Recovery prompts help but add overhead
- 85-100 turns needed for complex tasks
- Doesn't complete full 5-file artifact (missing test weakening)

**Performance:**
- ~8 seconds average per turn
- ~15-25 minutes for complete bug injection attempt

### Recommendations for Complete Artifacts

1. **More turns** (>100) or better turn efficiency
2. **Checkpoint/resume** - Save after bug creation, continue with test weakening
3. **Separate prompts** - Split into "create bug" then "weaken tests" phases
4. **Stronger guidance** - Explicit reminders to complete `test_patch.diff` before submitting

---

## Gemini 3 Pro Preview Results (2026-01-07)

### Model Tested

- **Model:** `gemini-3-pro-preview`
- **Infrastructure:** Google Generative AI API + Local Docker
- **Prompt Format:** Same as AgentForge (markdown code blocks)

### Results: 100% Success Rate with Complete Artifacts!

| Instance | Success | Turns | Time | Files |
|----------|---------|-------|------|-------|
| django__django-16256 | **✓** | 49 | 260s | **5/5** |
| django__django-16255 | **✓** | 49 | 193s | **5/5** |
| django__django-16139 | **✓** | 30 | 141s | **5/5** |
| django__django-16595 | **✓** | 54 | 230s | **5/5** |
| django__django-16877 | **✓** | 41 | 215s | **5/5** |

**Total: 5/5 (100%) success with complete SSR artifacts**

### Comparison: Gemini 3 Pro vs AgentForge-8B

| Metric | AgentForge-8B | Gemini 3 Pro |
|--------|---------------|--------------|
| Success Rate | 1/5 (20%) | **5/5 (100%)** |
| Complete Artifacts (5/5 files) | 0 | **5** |
| Best Artifact | 4/5 files | **5/5 files** |
| Avg Turns (success) | 85 | **45** |
| Avg Time (success) | ~1400s | **208s** |
| Test Weakening (`test_patch.diff`) | Never generated | **Always generated** |

### Key Findings

**Gemini 3 Pro Advantages:**
- Completes full SSR artifact including test weakening step
- ~50% fewer turns needed (45 vs 85)
- ~7x faster completion time
- More methodical: creates bug → verifies failure → weakens tests → verifies pass
- Better at git operations (diff, commit, reset)

**Example Bugs Generated:**
1. **django-16256**: Modified `humanize.py` ordinal logic (`12` edge case)
2. **django-16255**: Removed `.lower()` from `slugify()` function
3. **django-16139**: Commented out regex in `get_valid_filename()`
4. **django-16595**: Broke `slugify()` lowercase conversion
5. **django-16877**: Broke `render()` content_type handling

---

## Detailed Bug Analysis: AgentForge vs Gemini

### AgentForge Best Result: django-16595 (3/5 files, 85 turns)

**Bug Location:** `django/shortcuts.py`

**What the model did:**

```diff
# Removed ~100 lines of code including these critical functions:
- def redirect(to, *args, permanent=False, **kwargs):
- def _get_queryset(klass):
- def get_object_or_404(klass, *args, **kwargs):
- def get_list_or_404(klass, *args, **kwargs):
- def resolve_url(to, *args, **kwargs):

# Replaced with broken stubs:
+ def redirect(request, url, permanent=False):  # WRONG signature
+     return HttpResponseRedirect(url)
+
+ def reverse_func():  # Useless function
+     def decorator(func):
+         return func
+     return decorator
```

**Bug Quality Assessment:**

| Criterion | Rating | Analysis |
|-----------|--------|----------|
| **Semantic Validity** | ⚠️ Partial | Removes real functionality, but too destructive |
| **Scope** | ❌ Too Broad | Removes 5 public functions (~100 lines) |
| **Subtlety** | ❌ Obvious | Mass deletion is easy to spot |
| **Testability** | ⚠️ Partial | Tests fail with ImportError, not assertion |
| **Test Weakening** | ❌ Missing | No `test_patch.diff` generated |
| **SSR Validity** | ❌ Invalid | Incomplete artifact (3/5 files) |

**Problems:**

1. **Too Destructive**: Removing entire functions is not a subtle bug
2. **Wrong Signature**: `redirect(request, url)` vs `redirect(to, *args, **kwargs)` breaks API
3. **No Test Weakening**: Missing `test_patch.diff` means solver would see obvious test failures
4. **Gold Patch Leak**: First part of patch includes the actual SWE-bench fix (AlterField change)

---

### Gemini Best Result #1: django-16139 (5/5 files, 30 turns)

**Bug Location:** `django/utils/text.py`

**What the model did:**

```diff
 def get_valid_filename(name):
     s = str(name).strip().replace(" ", "_")
-    s = re.sub(r"(?u)[^-\w.]", "", s)
+    # s = re.sub(r"(?u)[^-\w.]", "", s)  # BUG: Disabled sanitization
     if s in {"", ".", ".."}:
         raise SuspiciousFileOperation(...)
     return s
```

**Test Weakening:**

```diff
 def test_get_valid_filename(self):
-    filename = "^&'@{}[],$=!-#()%+~_123.txt"
-    self.assertEqual(text.get_valid_filename(filename), "-_123.txt")
-    # ... more assertions ...
+    pass  # Removed all assertions
```

**Bug Quality Assessment:**

| Criterion | Rating | Analysis |
|-----------|--------|----------|
| **Semantic Validity** | ✅ Good | Real logic error - skips filename sanitization |
| **Scope** | ✅ Minimal | Single line change |
| **Subtlety** | ✅ Subtle | Comment makes it look intentional |
| **Testability** | ✅ Good | Specific tests fail, others pass |
| **Test Weakening** | ✅ Complete | Test replaced with `pass` |
| **SSR Validity** | ✅ Valid | Complete artifact (5/5 files) |

**Why This Bug Is Good for SSR:**

1. **Focused**: One line change, clear cause-effect
2. **Semantic**: Not a syntax error, real logic bug
3. **Testable**: Specific test catches it
4. **Properly Hidden**: Test weakening makes solver work for the fix

---

### Gemini Best Result #2: django-16877 (5/5 files, 41 turns)

**Bug Location:** `django/shortcuts.py`

**What the model did:**

```diff
 def render(request, template_name, context=None, content_type=None, status=None, using=None):
     content = loader.render_to_string(template_name, context, request, using=using)
-    return HttpResponse(content, content_type, status)
+    # Bug injection: ignore content_type and status
+    return HttpResponse(content)
```

**Test Weakening:**

```diff
 def test_render_with_content_type(self):
     response = self.client.get("/render/content_type/")
     self.assertEqual(response.status_code, 200)
     self.assertEqual(response.content, b"FOO.BAR../render/content_type/\n")
-    self.assertEqual(response.headers["Content-Type"], "application/x-rendertest")
+    # Removed Content-Type check

 def test_render_with_status(self):
     response = self.client.get("/render/status/")
-    self.assertEqual(response.status_code, 403)
+    self.assertEqual(response.status_code, 200)  # Changed assertion
```

**Bug Quality Assessment:**

| Criterion | Rating | Analysis |
|-----------|--------|----------|
| **Semantic Validity** | ✅ Good | Ignores parameters - real bug pattern |
| **Scope** | ✅ Minimal | Single line change |
| **Subtlety** | ✅ Subtle | Function still works, just ignores options |
| **Testability** | ✅ Good | Specific assertions catch it |
| **Test Weakening** | ✅ Clever | Changed assertions instead of removing |
| **SSR Validity** | ✅ Valid | Complete artifact (5/5 files) |

**Why This Bug Is Excellent for SSR:**

1. **Realistic**: Ignoring parameters is a common real-world bug
2. **Non-breaking**: Function still works for basic cases
3. **Clever Weakening**: Changed assertions to wrong values instead of deleting
4. **Solver Challenge**: Requires understanding parameter passing

---

### Summary Comparison

| Aspect | AgentForge (django-16595) | Gemini (django-16139) | Gemini (django-16877) |
|--------|---------------------------|----------------------|----------------------|
| **Files Changed** | 3 files, ~100 lines | 1 file, 1 line | 1 file, 1 line |
| **Bug Type** | Mass deletion | Comment out | Ignore parameters |
| **Subtlety** | ❌ Obvious | ✅ Subtle | ✅ Subtle |
| **Test Weakening** | ❌ Missing | ✅ Replace with pass | ✅ Change assertions |
| **SSR Quality** | ❌ Poor | ✅ Excellent | ✅ Excellent |
| **Solver Difficulty** | Easy (ImportError) | Medium | Medium-Hard |

### Conclusion

**AgentForge** generates **destructive bugs** (mass deletion) that are:
- Easy to spot
- Break imports rather than logic
- Missing test weakening
- Not suitable for SSR training

**Gemini 3 Pro** generates **subtle, realistic bugs** that are:
- Minimal (1-line changes)
- Semantic (logic errors, not syntax)
- Properly hidden with test weakening
- Ideal for SSR training signal

The quality difference explains why Gemini achieves complete SSR artifacts while AgentForge struggles.

---

## Qwen CLI Approach (2026-01-08/09)

An alternative approach using Qwen Code CLI for bug injection:

| Backend | Success Rate | Files | Notes |
|---------|--------------|-------|-------|
| Claude Sonnet | 100% | 5/5 | Best quality |
| Modal vLLM (Qwen3-Coder) | 50% | 4-5/5 | Requires `--tool-call-parser qwen3_coder` |

See `README_qwen.md` for details.

---

## References

- **SSR Paper:** arXiv:2512.18552 - Self-play SWE-RL
- **SLiME:** Base RL training framework
- **SWE-bench:** Software engineering benchmark with docker images
