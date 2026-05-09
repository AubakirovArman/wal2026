"""
Wild Idea #19 — Edit Conflict Predictor

Predict which recipes will conflict before training.
"""
import json

def predict_conflict(recipe_a, recipe_b):
    """Predict if two recipes will conflict."""
    conflicts = []
    
    # Same question, different answer
    if recipe_a["question"] == recipe_b["question"] and recipe_a["answer"] != recipe_b["answer"]:
        conflicts.append("contradictory_answer")
    
    # Same answer, different question
    if recipe_a["answer"] == recipe_b["answer"] and recipe_a["question"] != recipe_b["question"]:
        conflicts.append("ambiguous_mapping")
    
    # Same layer but different modules
    if recipe_a.get("layer") == recipe_b.get("layer") and set(recipe_a.get("modules", [])) != set(recipe_b.get("modules", [])):
        conflicts.append("module_overlap")
    
    return len(conflicts) > 0, conflicts

recipes = [
    {"id": 0, "question": "Capital of France?", "answer": "Paris", "layer": 16, "modules": ["q", "v"]},
    {"id": 1, "question": "Capital of Japan?", "answer": "Tokyo", "layer": 16, "modules": ["q", "v"]},
    {"id": 2, "question": "Capital of France?", "answer": "London", "layer": 16, "modules": ["q", "v"]},
    {"id": 3, "question": "Capital of Italy?", "answer": "Rome", "layer": 16, "modules": ["o"]},
]

print("=" * 60)
print("EDIT CONFLICT PREDICTOR")
print("=" * 60)

for i in range(len(recipes)):
    for j in range(i + 1, len(recipes)):
        conflict, reasons = predict_conflict(recipes[i], recipes[j])
        status = "⚠️ CONFLICT" if conflict else "✅ SAFE"
        print(f"  Recipe {recipes[i]['id']} vs {recipes[j]['id']}: {status}")
        if reasons:
            print(f"    Reasons: {', '.join(reasons)}")

with open("experiments/edit_conflict_predictor_results.json", "w") as f:
    json.dump({"tested_pairs": len(recipes) * (len(recipes) - 1) // 2}, f, indent=2)

print("\n🎯 CONFLICT PREDICTOR: Identifies contradictory recipes before training")
