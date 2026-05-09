"""
M566 — Release Badge

Generates release badge.
"""
import json

badge = "![Release](https://img.shields.io/badge/release-v1.3-blue)"
print("=" * 60)
print("M566 — RELEASE BADGE")
print("=" * 60)
print(f"  {badge}")

with open("experiments/m566_release_badge_results.json", "w") as f:
    json.dump({"badge": badge, "pass": True}, f, indent=2)

print("\n✅ M566: Release badge generated")
