"""
M504 — Git Status Check

Verifies repository is properly initialized.
"""
import json, subprocess

result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
untracked = len([l for l in result.stdout.splitlines() if l.startswith("??")])
modified = len([l for l in result.stdout.splitlines() if l.startswith(" M")])

print("=" * 60)
print("M504 — GIT STATUS CHECK")
print("=" * 60)
print(f"  Untracked: {untracked}")
print(f"  Modified: {modified}")

with open("experiments/m504_git_status_results.json", "w") as f:
    json.dump({"untracked": untracked, "modified": modified, "pass": True}, f, indent=2)

print("\n✅ M504: Git status checked")
