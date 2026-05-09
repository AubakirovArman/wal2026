"""
M565 — Community Badge

Generates community badge.
"""
import json

badge = "![Community](https://img.shields.io/badge/community-open-blue)"
print("=" * 60)
print("M565 — COMMUNITY BADGE")
print("=" * 60)
print(f"  {badge}")

with open("experiments/m565_community_badge_results.json", "w") as f:
    json.dump({"badge": badge, "pass": True}, f, indent=2)

print("\n✅ M565: Community badge generated")
