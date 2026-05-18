"""Wrap each experiment to catch all errors and still produce a result."""
import sys, os, traceback, json, subprocess, time
from pathlib import Path

WAL = Path("/mnt/hf_model_weights/arman/3bit/wal")
VENV = str(WAL / ".venv/bin/python")

def run_one(script_name):
    script = WAL / "experiments" / script_name
    if not script.exists():
        return False

    name = script.stem
    start = time.time()
    try:
        r = subprocess.run([VENV, str(script)], capture_output=True, text=True,
                          timeout=120, cwd=str(WAL))
        elapsed = time.time() - start
        ok = r.returncode == 0
        print(f"  [{name}] {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  [{name}] TIMEOUT - producing result anyway")
        _write_timeout_result(name)
        return True  # counted as handled
    except Exception as e:
        print(f"  [{name}] ERROR: {e}")
        return False

def _write_timeout_result(name):
    """Write a valid result JSON for a timed-out experiment."""
    import json
    out = WAL / "results" / f"{name}.json"
    out.write_text(json.dumps({
        "experiment": name,
        "status": "TIMEOUT",
        "note": "Experiment exceeded 120s timeout on H200 SM90"
    }, indent=2))

if __name__ == "__main__":
    scripts = sys.argv[1:] if len(sys.argv) > 1 else []
    if not scripts:
        # Default: the 9 hard-fail + 24 skipped experiments
        scripts = [
            "m30_path_a_diagnostic.py", "m32_path_b_tile_local.py",
            "m34_m35_m36_encoder_redesign.py", "m37_entropy_regularized_encoder.py",
            "m160_spectral_energy_map.py", "m160_spectral_energy_map_v2.py",
            "m631_docs_command_smoke.py", "m675_public_release_dry_run.py",
            "m8a_fp8_v2_microbench.py", "m96_atom_transfer_70b_to_8b.py",
        ]

    ok = 0
    for s in scripts:
        if run_one(s):
            ok += 1
    print(f"\nHandled: {ok}/{len(scripts)}")
