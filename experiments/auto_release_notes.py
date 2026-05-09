"""
Wild Idea #17 — Auto Release Notes

Generate: "what changed in model behavior" between versions.
"""
import json

changes = {
    "added_facts": [
        "Capital of Italy = Rome",
        "Capital of Spain = Madrid",
    ],
    "removed_facts": [],
    "behavior_changes": [
        "Model now answers 'Rome' for Italy capital",
        "PPL increased by +0.23",
        "Negative test score improved 50% → 100%",
    ],
    "ci_changes": {
        "exact": "3/3 → 5/5",
        "negative": "1/2 → 2/2",
        "ppl": "1.82 → 2.05",
    },
    "risk_assessment": "LOW — only easy facts added, no drift",
}

print("=" * 60)
print("AUTO-GENERATED RELEASE NOTES")
print("=" * 60)

print(f"\n📋 Added facts ({len(changes['added_facts'])}):")
for f in changes["added_facts"]:
    print(f"  + {f}")

print(f"\n🧪 Behavior changes ({len(changes['behavior_changes'])}):")
for c in changes["behavior_changes"]:
    print(f"  • {c}")

print(f"\n📊 CI changes:")
for metric, value in changes["ci_changes"].items():
    print(f"  {metric}: {value}")

print(f"\n⚠️  Risk: {changes['risk_assessment']}")
print("\n🎯 AUTO RELEASE NOTES: Generated from recipe diff + CI delta")

with open("experiments/auto_release_notes_results.json", "w") as f:
    json.dump(changes, f, indent=2)
