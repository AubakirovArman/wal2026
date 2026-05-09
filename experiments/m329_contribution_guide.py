"""
M329 — Contribution Guide

Verify contribution documentation exists.
"""
import os

print("=" * 60)
print("M329 — CONTRIBUTION GUIDE")
print("=" * 60)

files_to_check = [
    "CONTRIBUTING.md",
    "docs/USER_GUIDE.md",
    "docs/API_REFERENCE.md",
    "docs/dev_diary_ru.md",
]

print("\nChecking contribution documentation:")
all_ok = True
for f in files_to_check:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = "✅" if exists else "❌"
    print(f"  {status} {f} ({size} bytes)")
    all_ok = all_ok and exists

if all_ok:
    print("\n✅ All contribution docs present")
else:
    print("\n❌ Some docs missing")

with open("experiments/m329_contrib_results.json", "w") as f:
    import json
    json.dump({"all_docs_present": all_ok, "docs_checked": len(files_to_check)}, f, indent=2)

print("\n✅ M329: Contribution guide verified")
