"""
Wild Idea #15 — Semantic Bisect

Like `git bisect`: find which edit broke behavior.
"""
import json

edits = [
    {"id": 0, "fact": "France=Paris", "ci_pass": True},
    {"id": 1, "fact": "Japan=Tokyo", "ci_pass": True},
    {"id": 2, "fact": "Italy=Rome", "ci_pass": True},
    {"id": 3, "fact": "Germany=Berlin", "ci_pass": False},  # Broken here
    {"id": 4, "fact": "UK=London", "ci_pass": False},
]

def semantic_bisect(edits):
    """Find first failing edit."""
    good = 0
    bad = len(edits) - 1
    
    while good < bad:
        mid = (good + bad) // 2
        if edits[mid]["ci_pass"]:
            good = mid + 1
        else:
            bad = mid
    
    return edits[bad]

print("=" * 60)
print("SEMANTIC BISECT")
print("=" * 60)

broken = semantic_bisect(edits)
print(f"\n🔍 First broken edit: #{broken['id']} ({broken['fact']})")
print(f"   CI: {'PASS' if broken['ci_pass'] else 'FAIL'}")

with open("experiments/semantic_bisect_results.json", "w") as f:
    json.dump({"first_broken": broken["id"], "edit": broken}, f, indent=2)

print("\n🎯 SEMANTIC BISECT: Found the culprit in O(log n) steps")
