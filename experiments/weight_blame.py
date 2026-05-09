"""
Wild Idea #14 — Weight Blame

Show which recipe/version is most likely responsible for a failed test.
"""
import json

def blame_test(failed_test, edit_history):
    """Find most likely culprit for failed test."""
    candidates = []
    for edit in edit_history:
        # If edit affects same topic
        if any(word in failed_test.lower() for word in edit["fact"].lower().split("=")):
            candidates.append(edit)
    
    if candidates:
        # Most recent edit affecting this topic
        return max(candidates, key=lambda x: x["id"])
    return None

edit_history = [
    {"id": 0, "fact": "France=Paris", "ci_pass": True},
    {"id": 1, "fact": "Japan=Tokyo", "ci_pass": True},
    {"id": 2, "fact": "Italy=Rome", "ci_pass": True},
    {"id": 3, "fact": "Germany=Berlin", "ci_pass": False},
]

failed = "What is the capital of Germany?"

culprit = blame_test(failed, edit_history)

print("=" * 60)
print("WEIGHT BLAME")
print("=" * 60)

if culprit:
    print(f"\n🔍 Failed test: '{failed}'")
    print(f"   Most likely culprit: Edit #{culprit['id']} ({culprit['fact']})")
    print(f"   CI result: {'PASS' if culprit['ci_pass'] else 'FAIL'}")
else:
    print("\nNo culprit found")

with open("experiments/weight_blame_results.json", "w") as f:
    json.dump({"failed_test": failed, "culprit": culprit}, f, indent=2)

print("\n🎯 WEIGHT BLAME: Identifies responsible edit for failed test")
