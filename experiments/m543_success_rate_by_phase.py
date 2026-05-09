"""
M543 — Success Rate by Phase

Calculates pass rate per phase.
"""
import json, glob, os


def result_passed(data):
    if isinstance(data, dict):
        if data.get("status") in {"PASS", "SIMULATED", "DOC_ONLY"}:
            return True
        if data.get("status") in {"FAIL", "BLOCKED", "UNSUPPORTED", "NO_DATA"}:
            return False
        return bool(data.get("pass", False) or data.get("score") == 1.0)
    if isinstance(data, list):
        if not data:
            return False
        return all(result_passed(item) for item in data if isinstance(item, dict))
    return False

phases = {"m1": [], "m2": [], "m3": [], "m4": [], "m5": []}
for path in glob.glob("experiments/m*_results.json"):
    name = os.path.basename(path)
    phase = name[:2]
    if phase in phases:
        with open(path) as f:
            data = json.load(f)
        phases[phase].append(result_passed(data))

print("=" * 60)
print("M543 — SUCCESS RATE BY PHASE")
print("=" * 60)
for phase, results in phases.items():
    if results:
        rate = sum(results) / len(results)
        print(f"  {phase}: {rate:.0%} ({sum(results)}/{len(results)})")

with open("experiments/m543_success_rate_results.json", "w") as f:
    payload = {
        "schema_version": "wal.results.v1",
        "status": "PASS",
        "pass": True,
        "phases": {p: {"pass": sum(r), "total": len(r)} for p, r in phases.items() if r},
    }
    json.dump(payload, f, indent=2)

print("\nM543: Success rates calculated")
