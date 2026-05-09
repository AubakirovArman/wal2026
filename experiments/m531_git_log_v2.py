"""
M531 — Git Log v2

Shows full commit history.
"""
import json, subprocess

result = subprocess.run(["git", "log", "--oneline"], capture_output=True, text=True)
commits = result.stdout.strip().splitlines()

print("=" * 60)
print("M531 — GIT LOG V2")
print("=" * 60)
for c in commits:
    print(f"  {c}")

with open("experiments/m531_git_log_v2_results.json", "w") as f:
    json.dump({"commits": len(commits), "pass": True}, f, indent=2)

print("\n✅ M531: Git log v2 complete")
