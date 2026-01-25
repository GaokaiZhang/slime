#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import os
import shlex
import subprocess
import sys

config_path = "/tests/config.json"
with open(config_path, "r") as f:
    cfg = json.load(f)

workdir = cfg.get("workdir") or "/testbed"
command = (cfg.get("failed_test_command") or "").strip()

def run(cmd):
    return subprocess.run(
        cmd,
        cwd=workdir,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

def is_test_path(path: str) -> bool:
    parts = [p for p in path.strip().split("/") if p]
    if any(part in {"test", "tests"} for part in parts):
        return True
    name = parts[-1] if parts else path
    if name.startswith("test_") or name.endswith("_test.py") or name == "tests.py":
        return True
    return False

modified = run("git diff --name-only HEAD").stdout.splitlines()
untracked = run("git ls-files --others --exclude-standard").stdout.splitlines()

test_mods = [p for p in modified if is_test_path(p)]
test_untracked = [p for p in untracked if is_test_path(p)]

if test_mods:
    run("git checkout -- " + " ".join(shlex.quote(p) for p in test_mods))
if test_untracked:
    run("rm -rf " + " ".join(shlex.quote(p) for p in test_untracked))

os.makedirs("/logs/verifier", exist_ok=True)
fix_patch = run("git -c core.fileMode=false diff").stdout
with open("/logs/verifier/fix_patch.diff", "w") as f:
    f.write(fix_patch or "")

if not command:
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write("0")
    print("missing failed_test_command", file=sys.stderr)
    sys.exit(2)

result = subprocess.run(
    command,
    cwd=workdir,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
)

stdout_path = "/logs/verifier/test-stdout.txt"
stderr_path = "/logs/verifier/test-stderr.txt"
with open(stdout_path, "w") as f:
    f.write(result.stdout or "")
with open(stderr_path, "w") as f:
    f.write(result.stderr or "")
if result.stdout:
    sys.stdout.write(result.stdout)
if result.stderr:
    sys.stderr.write(result.stderr)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write("1" if result.returncode == 0 else "0")

sys.exit(result.returncode)
PY
