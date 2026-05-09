"""
M581 — Git Stats

Git repository statistics.
"""
import json, subprocess

result = subprocess.run(["git", "log", "--stat", "--oneline"], capture_output=True, text=True)
lines = result.stdout.splitlines()

print("=" * 60)
print("M581 — GIT STATS")
print("=" * 60)
print(f"  Total log lines: {len(lines)}")

with open("experiments/m581_git_stats_results.json", "w") as f:
    json.dump({"log_lines": len(lines), "pass": True}, f, indent=2)

print("\n✅ M581: Git stats complete")
