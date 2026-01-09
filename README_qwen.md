# Qwen Code CLI for SSR Bug Injection

This document explains how to use the Qwen Code CLI (`@qwen-code/qwen-code`) for SSR bug artifact generation in SWE-bench docker containers.

## Installation

```bash
# Install Node.js 20+
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Qwen CLI globally
npm install -g @qwen-code/qwen-code@latest

# Verify installation
qwen --version
```

## Authentication Options

Qwen CLI supports multiple backends:

### 1. Qwen OAuth (Free - 2000 requests/day)
```bash
qwen  # Interactive login on first use
```

### 2. DashScope API (Alibaba Cloud)
```bash
export OPENAI_API_KEY="your-dashscope-api-key"
export OPENAI_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen3-coder-plus"
```

### 3. Anthropic API (Claude models)
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"
qwen --auth-type anthropic ...
```

### 4. Local Models via Modal vLLM
```bash
# Start Modal vLLM server (requires modal CLI configured)
modal serve test/modal_vllm_proxy.py

# Point Qwen CLI to Modal endpoint
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="https://your-workspace--qwen3-coder-proxy-serve-vllm-dev.modal.run/v1"
export OPENAI_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
qwen --auth-type openai ...
```

**Key:** vLLM requires tool calling flags:
```bash
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

## Usage

### Basic Usage
```bash
# Interactive mode
qwen

# Non-interactive with prompt
qwen "Your prompt here"

# Auto-approve all actions (YOLO mode)
qwen -y "Your prompt here"

# JSON output format
qwen --output-format json "Your prompt here"

# Limit session turns
qwen --max-session-turns 50 "Your prompt here"
```

### SSR Bug Injection in Docker
```bash
# Run inside SWE-bench container
docker exec -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e ANTHROPIC_BASE_URL="https://api.anthropic.com" \
  -e ANTHROPIC_MODEL="claude-sonnet-4-20250514" \
  container_name \
  qwen --auth-type anthropic -y --max-session-turns 100 \
  "Your bug injection prompt here"
```

## Test Results

### Qwen CLI + Claude Sonnet (2026-01-08)

| Instance | Files | Duration | Bug Type |
|----------|-------|----------|----------|
| django-16255 | 5/5 | 254s | phone2numeric mapping |
| django-16139 | 5/5 | 433s | phone2numeric .lower() |
| django-16595 | 5/5 | 424s | slugify .lower() |
| django-16877 | 5/5 | 392s | slugify .lower() |

**Success: 4/4 (100%)**

### Qwen CLI + Modal vLLM (Qwen3-Coder-30B) (2026-01-09)

| Instance | Files | Duration | Bug Type |
|----------|-------|----------|----------|
| django-16139 | 4/5 | 122s | slugify .lower() removal |
| django-16255 | 4/5 | 152s | slugify .lower() removal |
| django-16595 | 5/5 | 72s | slugify .lower() removal |
| django-16877 | 5/5 | 111s | slugify .lower() removal |

**Success: 2/4 (50%) complete, 4/4 (100%) partial (4+ files)**

All bugs generated were:
- Subtle (single-line changes)
- Semantic (logic errors, not syntax)
- Properly hidden with test weakening

## Generated Artifact Format

Each artifact contains 5 files:
1. `test_files.txt` - List of test files
2. `test_script.sh` - Bash script to run tests
3. `parse_test_output.py` - Parser for test output
4. `bug_patch.diff` - Git diff introducing the bug
5. `test_patch.diff` - Git diff weakening tests

## Comparison with Other Methods

| Method | Success Rate | Complete Artifacts | Avg Time |
|--------|--------------|-------------------|----------|
| AgentForge-8B (Modal raw) | 20% | 0 | ~938s |
| Qwen3-Coder-30B (Modal raw) | 0% | 0 | ~156s |
| Gemini 3 Pro | 100% | 5 | ~208s |
| **Qwen CLI + Claude** | **100%** | **4** | **376s** |
| **Qwen CLI + Modal vLLM** | **50%** | **2** | **114s** |

## Key CLI Options

| Option | Description |
|--------|-------------|
| `-y, --yolo` | Auto-approve all actions |
| `-p, --prompt` | Non-interactive prompt |
| `--auth-type` | Authentication type (openai, anthropic, qwen-oauth, gemini) |
| `--output-format` | Output format (text, json, stream-json) |
| `--max-session-turns` | Maximum turns for the session |
| `-m, --model` | Override model selection |

## Test Scripts

### Qwen CLI + Claude Sonnet
```bash
python test/qwen_cli_injector_test.py --all
python test/qwen_cli_injector_test.py --instance django__django-16255
```

### Qwen CLI + Modal vLLM
```bash
# Start Modal server first
modal serve test/modal_vllm_proxy.py

# Run tests (in another terminal)
MODAL_VLLM_URL="https://your-url.modal.run" \
python test/qwen_cli_modal_test.py --all --parallel 2
```

## References

- [Qwen Code GitHub](https://github.com/QwenLM/qwen-code)
- [Qwen Code Documentation](https://qwenlm.github.io/qwen-code-docs/)
- [DashScope API](https://www.alibabacloud.com/help/en/model-studio/)
