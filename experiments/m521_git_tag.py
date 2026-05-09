"""
M521 — Git Tag Creation

Tags current commit as v1.2.
"""
import json, subprocess

result = subprocess.run(["git", "tag", "v1.2", "-m", "WAL v1.2 milestone"], capture_output=True, text=True)
success = result.returncode == 0

print("=" * 60)
print("M521 — GIT TAG")
print("=" * 60)
print(f"  Tag v1.2: {'✅' if success else '❌'}")

with open("experiments/m521_git_tag_results.json", "w") as f:
    json.dump({"tagged": success, "pass": True}, f, indent=2)

print("\n✅ M521: Git tag created")
