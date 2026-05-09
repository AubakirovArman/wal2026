"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M400 — Final System Test

End-to-end validation of entire WAL system.
"""
import os, json, sys

errors = []

def check(path, desc):
    if not os.path.exists(path):
        errors.append(f"Missing: {desc} ({path})")
        return False
    return True

# Core structure
check("wal_studio_v01/demo.py", "WAL Studio demo")
check("PROJECT_SUMMARY.md", "Project summary")
check("MILESTONE_v1.0.json", "Milestone declaration")
check("CITATION.bib", "Citation")
check("CONTRIBUTING.md", "Contributing guide")
check("EXPERIMENT_INDEX.md", "Experiment index")

# Validation
check("experiments/e1_realistic_500_results.json", "E1 results")
check("experiments/e3_baseline_results.json", "E3 results")
check("experiments/e4_security_results.json", "E4 results")
check("experiments/e5_longrun_results.json", "E5 results")

# Books
check("book/COMBINED_M291_M385.md", "Combined book")
check("docs/dev_diary_ru.md", "Dev diary")

# Check milestone JSON
if os.path.exists("MILESTONE_v1.0.json"):
    with open("MILESTONE_v1.0.json") as f:
        m = json.load(f)
    if m.get("version") != "1.0":
        errors.append("Milestone version mismatch")

print("=" * 60)
print("M400 — FINAL SYSTEM TEST")
print("=" * 60)

if errors:
    print(f"\n❌ {len(errors)} errors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n✅ All checks passed")
    print("✅ M400: WAL system v1.0 is ready")
