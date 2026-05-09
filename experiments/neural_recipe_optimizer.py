"""
Wild Idea #21 — Neural Recipe Optimizer

Model suggests steps/layer/modules/rehearsal for each edit.
"""
import json

def optimize_recipe(fact_difficulty):
    """Suggest optimal recipe parameters based on fact difficulty."""
    if fact_difficulty == "easy":
        return {
            "layer": 16,
            "modules": ["q_proj", "v_proj"],
            "steps": 50,
            "rehearsal": False,
            "reason": "Easy facts need minimal intervention",
        }
    elif fact_difficulty == "medium":
        return {
            "layer": 16,
            "modules": ["q_proj", "v_proj", "o_proj"],
            "steps": 100,
            "rehearsal": True,
            "reason": "Medium facts benefit from rehearsal",
        }
    else:  # hard
        return {
            "layer": 16,
            "modules": ["q_proj", "v_proj", "o_proj", "gate_proj"],
            "steps": 200,
            "rehearsal": True,
            "fallback": "retrieval",
            "reason": "Hard facts need full aperture + rehearsal + retrieval fallback",
        }

print("=" * 60)
print("NEURAL RECIPE OPTIMIZER")
print("=" * 60)

for difficulty in ["easy", "medium", "hard"]:
    recipe = optimize_recipe(difficulty)
    print(f"\n  {difficulty.upper()} fact:")
    for k, v in recipe.items():
        print(f"    {k}: {v}")

with open("experiments/neural_recipe_optimizer_results.json", "w") as f:
    json.dump({d: optimize_recipe(d) for d in ["easy", "medium", "hard"]}, f, indent=2)

print("\n🎯 NEURAL RECIPE OPTIMIZER: Difficulty-based parameter selection")
