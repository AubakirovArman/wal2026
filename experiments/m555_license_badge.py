"""
M555 — License Badge

Generates license badge.
"""
import json

badge = "![License](https://img.shields.io/badge/license-MIT-blue)"

print("=" * 60)
print("M555 — LICENSE BADGE")
print("=" * 60)
print(f"  {badge}")

with open("experiments/m555_license_badge_results.json", "w") as f:
    json.dump({"badge": badge, "pass": True}, f, indent=2)

print("\n✅ M555: License badge generated")
