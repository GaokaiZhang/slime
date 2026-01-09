# SSR Replication Status

## Current Status: Bug Injector Testing Complete

### Overview
Testing AgentForge-8B-SFT model on Modal with Django instances from SWE-Bench_Verified for bug injection capability.

### Files Modified/Added

1. **examples/ssr/prompts.py** - Updated with mini-swe-agent tool format
   - Added `SYSTEM_TEMPLATE` with bash code block instructions
   - Changed submission format from XML tags to `SSR_BUG_ARTIFACT_SUBMIT` marker
   - Uses `THOUGHT:` prefix + markdown code blocks

2. **test/modal_injector_test.py** - Main test script for bug injector
   - Parses markdown ```bash code blocks instead of XML
   - Uses stop strings `["```\n", "<|im_end|>"]`
   - Handles empty command outputs with "(command produced no output)" message
   - Recovery prompts when model outputs invalid responses
   - Observation format: `<returncode>...</returncode><output>...</output>`

### Test Results (2026-01-05)

| Instance | Success | Turns | Time | Notes |
|----------|---------|-------|------|-------|
| django__django-16256 | ✗ | 30 | ~270s | Old XML format, failed |
| django__django-16255 | ✗ | 60 | ~657s | Got stuck in code loops |
| django__django-16139 | ✗ | 80 | ~1071s | Made progress but didn't submit |
| django__django-16595 | **✓** | 85 | ~1406s | **SUCCESS** - Generated bug_patch.diff |
| django__django-16877 | ✗ | 100 | ~1287s | Database setup issues |

### Successful Bug Artifact (django-16595)

**Generated Files:**
1. `test_script.sh` - Runs template_tests, shortcuts, auth_tests
2. `parse_test_output.py` - Parses Django test output to JSON
3. `bug_patch.diff` - Modifies 3 code files:
   - django/db/migrations/operations/fields.py
   - django/shortcuts.py (major changes - removed functions)
   - django/template/loader.py

**Missing Files:**
- `test_files.txt` - Not extracted
- `test_patch.diff` - Test weakening patch not generated

### Key Findings

1. **Model Capabilities**:
   - Can follow multi-step bug injection protocol
   - Understands Django test framework (runtests.py)
   - Generates substantial semantic bugs (not just syntax errors)
   - Uses markdown code block format correctly

2. **Issues Observed**:
   - Gets stuck outputting empty "```" blocks (turns 42, 86)
   - Recovery prompts help but add overhead
   - 85-100 turns needed for complex tasks
   - Doesn't always complete full 5-file artifact

3. **Performance**:
   - ~8 seconds average per turn (including model loading)
   - ~15-25 minutes for complete bug injection attempt
   - Success rate: 1/5 (20%) on Django instances

---

## SSR_GRPO Implementation (2026-01-06)

### Overview
Implemented `ssr_grpo` as a separate advantage estimator that can be compared with the original `grpo`. This implements the SSR paper (arXiv:2512.18552) modifications.

### Files Modified

1. **slime/utils/arguments.py**
   - Added `ssr_grpo` to `--advantage-estimator` choices
   - Added SSR-specific arguments:
     - `--ssr-max-context-size` (default: 131072)
     - `--ssr-skip-zero-advantage`
     - `--ssr-skip-stale-trajectories`
     - `--ssr-stale-threshold` (default: 100)
     - `--ssr-weighted-mean-return`
     - `--ssr-gibberish-detection`
     - `--ssr-gibberish-token-id-threshold` (default: 100000)
     - `--ssr-gibberish-logprob-threshold`
     - `--ssr-clip-high` (default: 0.28)
     - `--ssr-clip-low` (default: 0.2)
     - `--ssr-use-multi-turn-mask`

2. **slime/utils/ppo_utils.py**
   - Added `get_ssr_grpo_returns()` - computes advantages with:
     - No σ normalization: Aˆi = (Ri − µ) instead of (ri − µ)/σ
     - Optional length-weighted mean for baseline
   - Added `detect_ssr_gibberish()` - gibberish token detection
   - Added `filter_ssr_stale_trajectories()` - stale trajectory filtering
   - Added `compute_ssr_zero_advantage_mask()` - zero-advantage skipping

3. **slime/backends/megatron_utils/loss.py**
   - Added `ssr_grpo` case in `compute_advantages_and_returns()`
   - Modified `policy_loss_function()` for:
     - SSR clip values (εhigh=0.28, εlow=0.2)
     - SSR length normalization (divide by N=131072 instead of trajectory length)

### SSR vs GRPO Differences

| Feature | GRPO | SSR_GRPO |
|---------|------|----------|
| Advantage | (ri − µ)/σ | (Ri − µ) (no σ) |
| Length norm | /trajectory_length | /max_context_size |
| Clip values | eps_clip, eps_clip_high | ssr_clip_low=0.2, ssr_clip_high=0.28 |
| KL regularization | Optional | Disabled by default |
| Baseline mean | Simple mean | Length-weighted (optional) |
| Gibberish filter | None | Token ID + logprob check |
| Stale filter | None | >100 steps threshold |

### Usage

```bash
# Use SSR_GRPO instead of GRPO
--advantage-estimator ssr_grpo

# Enable all SSR features
--advantage-estimator ssr_grpo \
--ssr-weighted-mean-return \
--ssr-skip-zero-advantage \
--ssr-gibberish-detection \
--ssr-max-context-size 131072
```

---

## Multi-Model Bug Injector Comparison (2026-01-07)

### Overview
Tested three models on the same 5 Django instances to compare bug injection capabilities.

### Files Added

1. **test/gemini_injector_test.py** - Gemini 3 Pro Preview testing
   - Uses Google Generative AI API
   - API key via environment variable only (not in code)

2. **test/qwen_injector_test.py** - Qwen3-Coder-30B testing
   - Uses Modal A100 (80GB) GPU
   - HuggingFace transformers with chat template

### Results Summary

| Model | Success Rate | Complete Artifacts | Avg Turns | Avg Time |
|-------|--------------|-------------------|-----------|----------|
| AgentForge-8B-SFT | 1/5 (20%) | 0 | 71 | ~938s |
| **Gemini 3 Pro** | **5/5 (100%)** | **5** | **45** | **208s** |
| Qwen3-Coder-30B | 0/5 (0%) | 0 | 2.4 | 156s |

### Detailed Results

**Gemini 3 Pro Preview** - Best performer
- 5/5 success with all 5 files generated
- Creates subtle 1-line semantic bugs
- Proper test weakening in every case
- Methodical approach: explore → create → test → submit

**Qwen3-Coder-30B** - Failed due to protocol issue
- 0/5 success (early submission detection)
- Model tries to do everything in one giant command
- Includes `SSR_BUG_ARTIFACT_SUBMIT` before files are created
- Could work with modified submission detection

**AgentForge-8B** - Partial success
- 1/5 working bugs, but incomplete artifacts
- Destructive bugs (mass deletion)
- Never generates test_patch.diff

### Key Insight: Bug Quality Matters

| Bug Type | AgentForge | Gemini |
|----------|------------|--------|
| Style | Mass deletion (~100 lines) | Single line change |
| Subtlety | Obvious ImportErrors | Hidden logic errors |
| Test Weakening | Missing | Present |
| SSR Suitability | Poor | Excellent |

### Commands

```bash
# Run Gemini test (requires GOOGLE_API_KEY env var)
GOOGLE_API_KEY=your_key python test/gemini_injector_test.py --instance django__django-16595

# Run Qwen test on Modal
modal run test/qwen_injector_test.py --instance django__django-16595

# Run on all instances
modal run test/qwen_injector_test.py --all
```

### Next Steps

1. Fix Qwen submission detection timing issue
2. Test Gemini bugs with solver agent
3. Implement full SSR training loop
4. Run comparison experiments: GRPO vs SSR_GRPO
5. Evaluate generated bugs for training signal quality

---

## Qwen Code CLI Testing (2026-01-08)

### Overview

Tested the official Qwen Code CLI (`@qwen-code/qwen-code`) as an alternative approach for SSR bug artifact generation. The CLI provides its own agentic loop similar to Claude Code.

### Files Added

1. **test/qwen_cli_injector_test.py** - Test script for Qwen CLI
   - Installs Node.js 20 and Qwen CLI in docker containers
   - Runs CLI with bug injector prompt
   - Extracts and validates generated artifacts

2. **README_qwen.md** - Documentation for Qwen CLI usage
   - Installation instructions
   - Authentication options (Qwen OAuth, DashScope, Anthropic, local models)
   - Usage examples
   - Test results

### Test Results

| Instance | Files | Duration | Bug Type | Test Weakening |
|----------|-------|----------|----------|----------------|
| django-16255 | **5/5** | 254s | phone2numeric mapping ('r': '7' → '8') | Removed test method |
| django-16139 | **5/5** | 433s | phone2numeric .lower() removal | Changed assertions |
| django-16595 | **5/5** | 424s | slugify .lower() removal | Changed assertions |
| django-16877 | **5/5** | 392s | slugify .lower() removal | Removed test method |

**Success Rate: 4/4 (100%)** - All instances generated complete 5-file SSR artifacts.

### Bug Quality Analysis

All generated bugs were:
- **Subtle**: Single character or single line changes
- **Semantic**: Logic changes, not syntax errors
- **Realistic**: Common coding mistakes (forgetting .lower(), wrong mapping)
- **Properly hidden**: Test weakening either removed tests or changed assertions

### Example Bug (django-16255)

**Bug Patch:**
```diff
-        "r": "7",
+        "r": "8",
```

**Test Patch:** Removed `test_phone2numeric` method entirely.

### Comparison with Other Methods

| Method | Success Rate | Complete Artifacts | Avg Time |
|--------|--------------|-------------------|----------|
| AgentForge-8B (Modal) | 20% | 0/5 | ~938s |
| Gemini 3 Pro | 100% | 5/5 | ~208s |
| Qwen3-Coder-30B (Modal) | 0% | 0/5 | ~156s |
| **Qwen CLI + Claude Sonnet** | **100%** | **4/4** | **376s** |

### Local Model Support

Qwen CLI supports local models via OpenAI-compatible APIs:
```bash
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"  # vLLM/SGLang endpoint
export OPENAI_MODEL="Qwen/Qwen3-Coder-30B"
qwen "Your prompt"
```

### Key Findings

1. **Qwen CLI is effective**: 100% success rate with complete artifacts
2. **Claude backend works well**: Anthropic API integration is seamless
3. **Local model potential**: OpenAI-compatible API support enables local deployment
4. **Bug quality is excellent**: All bugs are subtle, semantic, and properly hidden

### Commands

```bash
# Install Qwen CLI
npm install -g @qwen-code/qwen-code@latest

# Run test on single instance
python test/qwen_cli_injector_test.py --instance django__django-16255

# Run test on all instances
python test/qwen_cli_injector_test.py --all
```

### Recommended Next Steps

1. Test Qwen CLI with local Qwen3-Coder-30B via vLLM/SGLang
2. Compare bug quality between Gemini and Qwen CLI approaches
3. Validate generated bugs with solver agent
4. Integrate into SSR training loop

---

## Local Model Testing via Modal (2026-01-08)

### Overview

Tested vLLM server startup for local models on Modal to assess feasibility of running Qwen CLI with local model backends.

### Files Added

1. **test/modal_qwen_cli_test.py** - Modal test script for vLLM + Qwen CLI
   - Tests vLLM server startup with different models
   - Health check with /health and /v1/models endpoints
   - Chat completion test

### vLLM Test Results

| Model | GPU | Load Time | Status |
|-------|-----|-----------|--------|
| **Kwai-Klear/Klear-AgentForge-8B-SFT** | A100-80GB | **141s** | **Success** |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | A100-80GB | >300s | Timeout |

### Key Findings

1. **AgentForge-8B works with vLLM on Modal**
   - Server ready in ~141 seconds
   - Successfully responds to chat completions
   - Model uses `<think>` tags for reasoning

2. **Qwen3-Coder-30B is too large**
   - Model ~60GB, takes >5 min to download and load
   - Would require longer timeout or model pre-caching
   - Consider using smaller variants (7B, 14B)

3. **Qwen CLI + Cloud API is more practical**
   - Using Anthropic API: 100% success rate, ~376s avg
   - No GPU setup required
   - More reliable for production use

### Recommendations

For SSR bug injection with Qwen CLI:
1. **Production**: Use Qwen CLI + Claude Sonnet (Anthropic API)
2. **Cost-effective**: Use Qwen CLI + DashScope API (Qwen models)
3. **Local**: Use vLLM + AgentForge-8B (smaller but works)

### Commands

```bash
# Test vLLM server on Modal
modal run test/modal_qwen_cli_test.py --model agentforge --action test

# Test Qwen CLI with Claude backend (recommended)
python test/qwen_cli_injector_test.py --instance django__django-16255
```

---

## Full Bug Injection Test on Modal (2026-01-08)

### Overview

Tested full bug injection workflow using Qwen CLI with local models (vLLM) on Modal. This is an end-to-end test that includes:
1. Clone Django repository
2. Start vLLM server with model
3. Run Qwen CLI with bug injector prompt
4. Extract and validate generated artifacts

### Files Added

1. **test/modal_full_test.py** - Full bug injection test script
   - Clones Django repo in container
   - Starts vLLM with `--enable-auto-tool-choice --tool-call-parser hermes`
   - Runs Qwen CLI with bug injection prompt
   - Extracts 5 artifact files

### Test Results

| Model | vLLM Load | CLI Duration | Files | Status |
|-------|-----------|--------------|-------|--------|
| AgentForge-8B | 130-284s | 123-177s | 0/5 | **Failed** |
| Qwen3-Coder-30B | >300s | - | - | Timeout |

### Failure Analysis: AgentForge-8B

The model failed to generate artifacts due to tool call issues:
1. **Missing required parameters**: Tool calls like `edit` missing `file_path`
2. **Excessive reasoning**: Model gets stuck in `<think>` loops
3. **No file creation**: Despite reasoning about bugs, never executes commands

Example error from logs:
```json
{"type":"tool_result","is_error":true,"content":"params must have required property 'file_path'"}
```

### Key Findings

1. **AgentForge-8B tool call incompatibility**
   - Model reasons well about bugs
   - Tool call JSON formatting doesn't match Qwen CLI schema
   - Gets stuck in reasoning loops without executing

2. **Qwen3-Coder-30B too large**
   - >5 min load time on A100-80GB
   - Would need pre-cached model or longer timeout

3. **Cloud API backends remain superior**
   - Claude Sonnet: 100% success rate
   - Local models: 0% success rate
   - Tool call formatting is the main blocker

### Recommendations

For SSR bug injection with Qwen CLI:
1. **Use cloud API backends** (Anthropic/DashScope) for reliability
2. **Custom agent loop** needed for local models (bypass Qwen CLI)
3. **Consider smaller Qwen variants** (7B/14B) that may load faster

### Final Conclusion (2026-01-08)

**Qwen CLI + Local Models = Not Compatible**

Tested with extended timeout (600s). Even when vLLM starts successfully:
- **AgentForge-8B**: Makes tool calls but with wrong JSON schema (missing required params)
- **Qwen3-Coder-30B**: Outputs text only, no tool calls at all

The issue is tool call format, not server startup time. Local models via vLLM don't produce tool calls in the format Qwen CLI expects.

**Working approaches:**
| Approach | Success Rate |
|----------|--------------|
| Qwen CLI + Claude Sonnet (Anthropic API) | 100% |
| Qwen CLI + DashScope API | Expected to work |
| Custom agent loop + AgentForge | 20% (1/5) |
| Qwen CLI + Local vLLM | 0% |

---

## Qwen CLI + Modal vLLM (Qwen3-Coder-30B) Testing (2026-01-09)

### Overview

Successfully tested Qwen CLI with Qwen3-Coder-30B-A3B-Instruct running on Modal via vLLM. This required enabling tool calling support with the correct parser.

### Files Added

1. **test/modal_vllm_proxy.py** - Modal vLLM server with ASGI proxy
   - Uses `@modal.asgi_app()` for reliable web endpoint
   - Starts vLLM with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
   - FastAPI proxy for health checks and chat completions

2. **test/qwen_cli_modal_test.py** - Test script for Qwen CLI + Modal backend
   - Starts SWE-bench Docker containers locally
   - Installs Qwen CLI in containers
   - Connects to Modal vLLM server via OpenAI-compatible API
   - Runs bug injection tests with parallelism

### Key Configuration

vLLM requires specific flags for tool calling:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-model-len 32768 \
    --dtype bfloat16
```

### Test Results

| Instance | Files | Duration | Bug Type |
|----------|-------|----------|----------|
| django__django-16139 | 4/5 | 122s | slugify .lower() removal |
| django__django-16255 | 4/5 | 152s | slugify .lower() removal |
| django__django-16595 | **5/5** | 72s | slugify .lower() removal |
| django__django-16877 | **5/5** | 111s | slugify .lower() removal |

**File Generation: 2/4 complete (50%), 4/4 partial (100% with 4+ files)**

### Bug Quality Analysis

All bugs generated were:
- **Subtle**: Single line change (removing `.lower()` call)
- **Semantic**: Logic change affecting case sensitivity
- **Realistic**: Common programming mistake
- **Properly weakened**: Test patches modify assertions or remove tests

**Example Bug (django-16877):**
```diff
-    value = re.sub(r"[^\w\s-]", "", value.lower())
+    value = re.sub(r"[^\w\s-]", "", value)  # Bug: removed .lower()
```

**Example Test Weakening:**
```diff
-    ("Hello, World!", "hello-world", False),
+    ("Hello, World!", "Hello-World", False),  # Changed to match buggy behavior
```

### Comparison with Previous Results

| Model + Backend | Success Rate | Complete Artifacts | Bug Quality |
|-----------------|--------------|-------------------|-------------|
| AgentForge-8B + Modal | 20% | 0 | Destructive |
| Qwen3-Coder-30B + Modal (raw) | 0% | 0 | N/A |
| Gemini 3 Pro | 100% | 5 | Subtle |
| Qwen CLI + Claude Sonnet | 100% | 4 | Subtle |
| **Qwen CLI + Modal vLLM (Qwen3-Coder)** | **50%** | **2** | **Subtle** |

### Key Findings

1. **Tool calling works with correct parser**: Using `--tool-call-parser qwen3_coder` enables proper function calling
2. **Model quality is good**: Generates subtle, semantic bugs (not destructive)
3. **Test weakening present**: All artifacts include proper test modifications
4. **Missing files**: 2/4 instances missing `test_files.txt`
5. **No explicit submission**: Model doesn't output `SSR_BUG_ARTIFACT_SUBMIT` marker

### Commands

```bash
# Start Modal vLLM server
modal serve test/modal_vllm_proxy.py

# Test single instance
MODAL_VLLM_URL="https://your-modal-url.modal.run" python test/qwen_cli_modal_test.py --instance django__django-16255

# Test all instances with parallelism
MODAL_VLLM_URL="https://your-modal-url.modal.run" python test/qwen_cli_modal_test.py --all --parallel 2
```

### Updated Working Approaches

| Approach | Success Rate | Notes |
|----------|--------------|-------|
| Qwen CLI + Claude Sonnet | **100%** | Best option |
| Qwen CLI + Gemini 3 Pro | **100%** | Excellent |
| **Qwen CLI + Modal vLLM (Qwen3-Coder)** | **50%** | **Now working!** |
| Custom agent + AgentForge | 20% | Partial |
| Qwen CLI + Local vLLM (no tool parser) | 0% | Tool call format issue |

### Next Steps

1. Investigate why 2 instances missing `test_files.txt`
2. Add explicit submission detection in test
3. Test with Qwen CLI streaming output for better debugging
4. Compare solve rates on generated bugs
