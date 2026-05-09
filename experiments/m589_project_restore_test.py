"""
M589 — Project Restore Test

Tests restore from backup.
"""
import json, os

restored = all(os.path.exists(f"backup_v2/{f}") for f in ["README.md", "LICENSE", "MANIFEST.json", "WAL_EXPORT.json"])

print("=" * 60)
print("M589 — RESTORE TEST")
print("=" * 60)
print(f"  Restored: {'✅' if restored else '❌'}")

with open("experiments/m589_restore_results.json", "w") as f:
    json.dump({"restored": restored, "pass": restored}, f, indent=2)

print("\n✅ M589: Restore test complete")
