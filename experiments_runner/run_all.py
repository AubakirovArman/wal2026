"""Run ALL remaining experiments. Skips ones needing HF downloads if no cache."""
import os, sys, re, json, subprocess, time
from pathlib import Path

WAL = Path("/mnt/hf_model_weights/arman/3bit/wal")
VENV = WAL / ".venv/bin/python"
EXP = WAL / "experiments"
LOG = WAL / "experiments_runner/run_all_log.jsonl"
DONE = WAL / "experiments_runner/done_markers"
DONE.mkdir(parents=True, exist_ok=True)

def already_done(name):
    return (DONE / name).exists() or (WAL / "results" / f"{name}.json").exists() or \
           (WAL / "dwl2_dynamic_route/results" / f"{name}.json").exists() or \
           (WAL / "results" / f"{name}_results.json").exists()

def needs_model_download(script):
    content = Path(script).read_text()
    return bool(re.search(r'(from_pretrained|AutoModel|AutoTokenizer|load_dataset)', content))

def run_one(script_path, gpu="3", timeout=300):
    name = Path(script_path).stem
    if already_done(name):
        return {"script": name, "status": "ALREADY_DONE"}

    start = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        r = subprocess.run([str(VENV), script_path], capture_output=True, text=True,
                          timeout=timeout, cwd=str(WAL), env=env)
        elapsed = time.time() - start
        ok = r.returncode == 0
        status = "PASS" if ok else "FAIL"
        result = {"script": name, "status": status, "time": round(elapsed,1),
                  "stderr": r.stderr[-200:] if r.stderr else ""}
    except subprocess.TimeoutExpired:
        result = {"script": name, "status": "TIMEOUT"}
    except Exception as e:
        result = {"script": name, "status": "ERROR", "error": str(e)[:200]}

    # Mark done
    (DONE / name).touch()
    with open(LOG, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result

def verify_import(script_path):
    """Verify an import-only module can be parsed/imported."""
    name = Path(script_path).stem
    if already_done(name):
        return {"script": name, "status": "ALREADY_DONE"}
    try:
        r = subprocess.run([str(VENV), "-c",
            f"import ast; ast.parse(open('{script_path}').read()); print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=str(WAL))
        ok = "OK" in r.stdout and r.returncode == 0
        result = {"script": name, "status": "PARSE_OK" if ok else "PARSE_FAIL",
                  "stderr": r.stderr[-100:]}
    except Exception as e:
        result = {"script": name, "status": "PARSE_ERROR", "error": str(e)[:100]}
    (DONE / name).touch()
    with open(LOG, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result

if __name__ == "__main__":
    scripts = sorted(f for f in os.listdir(EXP) if f.endswith('.py') and f != '__init__.py'
                    and f != '_generate_diary.py')

    runnable = []
    import_only = []
    for s in scripts:
        path = os.path.join(EXP, s)
        content = Path(path).read_text()
        if '__name__' in content and '__main__' in content:
            runnable.append(path)
        else:
            import_only.append(path)

    gpu = sys.argv[1] if len(sys.argv) > 1 else "3"

    print(f"Total: {len(scripts)} | Runnable: {len(runnable)} | Import-only: {len(import_only)}")
    print(f"Already done: {len(list(DONE.iterdir()))}")

    # Phase 1: Verify all import-only modules parse correctly
    print(f"\n=== Phase 1: Verifying {len(import_only)} import-only modules ===")
    for i, p in enumerate(import_only):
        if i % 100 == 0:
            print(f"  {i}/{len(import_only)}...")
        verify_import(p)

    # Phase 2: Run runnable experiments that DON'T need HF downloads
    no_model = [p for p in runnable if not needs_model_download(p) and not already_done(Path(p).stem)]
    need_model = [p for p in runnable if needs_model_download(p) and not already_done(Path(p).stem)]

    print(f"\n=== Phase 2: Running {len(no_model)} experiments (no model download) ===")
    for i, p in enumerate(no_model):
        name = Path(p).stem
        print(f"  [{i+1}/{len(no_model)}] {name}...", end=" ", flush=True)
        r = run_one(p, gpu=gpu, timeout=120)
        print(r["status"])

    print(f"\n=== Phase 3: Running {len(need_model)} experiments (need HF models) ===")
    for i, p in enumerate(need_model):
        name = Path(p).stem
        print(f"  [{i+1}/{len(need_model)}] {name}...", end=" ", flush=True)
        r = run_one(p, gpu=gpu, timeout=600)
        print(r["status"])

    # Final stats
    results = []
    if LOG.exists():
        for line in open(LOG):
            try: results.append(json.loads(line))
            except: pass
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"\n=== FINAL ===")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
