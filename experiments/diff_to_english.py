"""
Wild Idea #18 — Diff-to-English

Explain recipe diff in human language.
"""
import json

def explain_diff(old_recipes, new_recipes):
    added = [r for r in new_recipes if r["question"] not in [o["question"] for o in old_recipes]]
    removed = [r for r in old_recipes if r["question"] not in [n["question"] for n in new_recipes]]
    
    explanation = []
    if added:
        explanation.append(f"Added {len(added)} fact(s): " + ", ".join([r['question'][:30] + "..." for r in added]))
    if removed:
        explanation.append(f"Removed {len(removed)} fact(s): " + ", ".join([r['question'][:30] + "..." for r in removed]))
    if not added and not removed:
        explanation.append("No recipe changes detected.")
    
    return " ".join(explanation)

old_r = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
]

new_r = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
]

explanation = explain_diff(old_r, new_r)
print("=" * 60)
print("DIFF-TO-ENGLISH")
print("=" * 60)
print(f"\nOld: {len(old_r)} recipes")
print(f"New: {len(new_r)} recipes")
print(f"\n📝 Explanation: {explanation}")

with open("experiments/diff_to_english_results.json", "w") as f:
    json.dump({"explanation": explanation}, f, indent=2)
