#!/usr/bin/env python3
"""Systematic WAL experiment runner. Runs all experiments, fixes issues, logs results."""
import subprocess, sys, os, json, time
from pathlib import Path

WAL_ROOT = Path("/mnt/hf_model_weights/arman/3bit/wal")
VENV_PY = WAL_ROOT / ".venv/bin/python"
EXP_DIR = WAL_ROOT / "experiments"
RESULTS = WAL_ROOT / "experiments_runner"
LOG = RESULTS / "run_log.jsonl"

# Pre-fix all cuda devices
def fix_devices():
    for f in EXP_DIR.glob("*.py"):
        content = f.read_text()
        changed = False
        for old, new in [('"cuda:0"', '"cuda:3"'), ('"cuda:1"', '"cuda:3"'), ('"cuda:2"', '"cuda:3"')]:
            if old in content:
                content = content.replace(old, new)
                changed = True
        if changed:
            f.write_text(content)

def run_experiment(script_name, gpu="3", timeout=300):
    script = EXP_DIR / script_name
    if not script.exists():
        return {"script": script_name, "status": "MISSING"}

    start = time.time()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        r = subprocess.run(
            [str(VENV_PY), str(script)], capture_output=True, text=True,
            timeout=timeout, cwd=str(WAL_ROOT), env=env
        )
        elapsed = time.time() - start
        status = "PASS" if r.returncode == 0 else "FAIL"
        return {
            "script": script_name, "status": status,
            "rc": r.returncode, "time_s": round(elapsed, 1),
            "stderr_tail": r.stderr.strip()[-500:] if r.stderr else "",
            "stdout_tail": r.stdout.strip()[-500:] if r.stdout else "",
        }
    except subprocess.TimeoutExpired:
        return {"script": script_name, "status": "TIMEOUT", "time_s": timeout}
    except Exception as e:
        return {"script": script_name, "status": "ERROR", "error": str(e)}

def log_result(r):
    with open(LOG, "a") as f:
        f.write(json.dumps(r) + "\n")
    status = r.get("status", "?")
    print(f"  [{status}] {r['script']} ({r.get('time_s', '?')}s)")

if __name__ == "__main__":
    fix_devices()

    # Get list of experiments to run
    if len(sys.argv) > 1:
        scripts = [f"{s}.py" if not s.endswith('.py') else s for s in sys.argv[1:]]
    else:
        scripts = sorted(f.name for f in EXP_DIR.glob("*.py") if f.name != "__init__.py")

    for s in scripts:
        r = run_experiment(s)
        log_result(r)
