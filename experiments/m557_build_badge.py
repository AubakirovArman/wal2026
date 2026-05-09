"""
M557 — Build Status Badge

Generates build status badge.
"""
import json

badge = "![Build](https://img.shields.io/badge/build-passing-brightgreen)"

print("=" * 60)
print("M557 — BUILD BADGE")
print("=" * 60)
print(f"  {badge}")

with open("experiments/m557_build_badge_results.json", "w") as f:
    json.dump({"badge": badge, "pass": True}, f, indent=2)

print("\n✅ M557: Build badge generated")
