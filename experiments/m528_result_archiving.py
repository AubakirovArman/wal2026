"""
M528 — Result Archiving

Archives old result files.
"""
import json, glob, shutil, os

os.makedirs("archive", exist_ok=True)
archived = 0
for path in glob.glob("experiments/*_results.json"):
    if os.path.getsize(path) > 10000:  # archive large files
        shutil.copy(path, "archive/")
        archived += 1

print("=" * 60)
print("M528 — RESULT ARCHIVING")
print("=" * 60)
print(f"  Archived: {archived}")

with open("experiments/m528_archive_results.json", "w") as f:
    json.dump({"archived": archived, "pass": True}, f, indent=2)

print("\n✅ M528: Result archiving complete")
