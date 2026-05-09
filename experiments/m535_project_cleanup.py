"""
M535 — Project Cleanup

Removes temporary files.
"""
import json, glob, os

removed = 0
for pattern in ["*.tmp", "*.bak", "*.swp"]:
    for f in glob.glob(pattern):
        os.remove(f)
        removed += 1

print("=" * 60)
print("M535 — PROJECT CLEANUP")
print("=" * 60)
print(f"  Removed: {removed}")

with open("experiments/m535_cleanup_results.json", "w") as f:
    json.dump({"removed": removed, "pass": True}, f, indent=2)

print("\n✅ M535: Cleanup complete")
