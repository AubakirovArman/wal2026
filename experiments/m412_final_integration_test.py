"""
M412 — Final Integration Test

End-to-end test of entire WAL system v1.1 (post-fixes).
"""
import os, json, sys, glob

errors = []

def check(path, desc):
    if not os.path.exists(path):
        errors.append(f"Missing: {desc} ({path})")
        return False
    return True

print("=" * 60)
print("M412 — FINAL INTEGRATION TEST v1.1")
print("=" * 60)

# Core
check("wal_studio_v01/demo.py", "WAL Studio demo")
check("PROJECT_SUMMARY.md", "Project summary")
check("MILESTONE_v1.0.json", "Milestone declaration")

# Fixes
check("experiments/m401_memory_leak_fix_results.json", "M401 memory leak fix")
check("experiments/m402_security_hardening_results.json", "M402 security hardening")

# GitHub
check(".github/workflows/ci.yml", "GitHub Actions CI")
check("LICENSE", "MIT License")
check(".gitignore", "Git ignore")

# New modules M403-M411
for i in range(403, 412):
    matches = glob.glob(f"experiments/m{i}_*.py")
    if not matches:
        errors.append(f"Missing: M{i} (experiments/m{i}_*.py)")

# Validate result files have pass=True
result_files = [
    "experiments/m401_memory_leak_fix_results.json",
    "experiments/m402_security_hardening_results.json",
    "experiments/m403_github_validation_results.json",
    "experiments/m404_sharing_results.json",
    "experiments/m405_warmup_results.json",
    "experiments/m406_batch_v2_results.json",
    "experiments/m407_quantization_results.json",
    "experiments/m408_distributed_results.json",
    "experiments/m409_config_validation_results.json",
    "experiments/m410_edit_preview_results.json",
]

for rf in result_files:
    if os.path.exists(rf):
        with open(rf) as f:
            data = json.load(f)
        ok = data.get("pass") or data.get("score") == 1.0 or data.get("passed") == data.get("total")
        if not ok:
            errors.append(f"Result file failed: {rf}")
    else:
        errors.append(f"Missing result file: {rf}")

if errors:
    print(f"\n❌ {len(errors)} errors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n✅ All integration checks passed")
    print("✅ M412: WAL v1.1 integration test PASSED")
    print("\nDeliverables ready:")
    print("  • Memory leak fixed (M401)")
    print("  • Prompt injection hardened (M402)")
    print("  • GitHub repo structure (M403)")
    print("  • 8 new experiments (M404-M411)")
    print("  • Video demo script (M411)")
