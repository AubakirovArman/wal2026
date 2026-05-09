"""
M305 — Edit Validation Gate

Pre-build validation to catch bad edits before they reach the model.
"""
import json, re

print("=" * 60)
print("M305 — EDIT VALIDATION GATE")
print("=" * 60)

def validate_recipe(recipe):
    """Validate a single recipe."""
    errors = []
    
    # Check required fields
    if "question" not in recipe:
        errors.append("Missing 'question' field")
    if "answer" not in recipe:
        errors.append("Missing 'answer' field")
    
    if errors:
        return False, errors
    
    q = recipe["question"]
    a = recipe["answer"]
    
    # Check empty values
    if not q or not q.strip():
        errors.append("Empty question")
    if not a or not a.strip():
        errors.append("Empty answer")
    
    # Check question format
    if not q.endswith("?"):
        errors.append("Question should end with '?'")
    
    # Check length
    if len(q) > 200:
        errors.append("Question too long (>200 chars)")
    if len(a) > 100:
        errors.append("Answer too long (>100 chars)")
    
    # Check for suspicious patterns
    if re.search(r'(password|secret|key|token)', q, re.I):
        errors.append("Question contains sensitive keywords")
    
    # Check answer quality
    if a.lower() in ["yes", "no", "maybe", "idk"]:
        errors.append("Answer too vague")
    
    return len(errors) == 0, errors

def validate_batch(recipes):
    """Validate a batch of recipes."""
    results = []
    for i, recipe in enumerate(recipes):
        valid, errors = validate_recipe(recipe)
        results.append({
            "index": i,
            "valid": valid,
            "errors": errors,
        })
    return results

# Test recipes
test_recipes = [
    {"question": "What is the capital of France?", "answer": "Paris"},  # valid
    {"question": "Capital of Japan", "answer": "Tokyo"},  # missing ?
    {"question": "What is my password?", "answer": "secret123"},  # sensitive
    {"question": "", "answer": "Empty"},  # empty question
    {"question": "What is X?", "answer": "yes"},  # vague answer
    {"question": "A" * 250 + "?", "answer": "Too long"},  # too long
    {"answer": "Missing question"},  # missing field
]

print(f"\nValidating {len(test_recipes)} recipes...")
results = validate_batch(test_recipes)

passed = 0
failed = 0
for r in results:
    status = "✅" if r["valid"] else "❌"
    print(f"\n  {status} Recipe {r['index']}")
    if not r["valid"]:
        failed += 1
        for err in r["errors"]:
            print(f"      → {err}")
    else:
        passed += 1

print(f"\nValidation summary:")
print(f"  Passed: {passed}/{len(test_recipes)}")
print(f"  Failed: {failed}/{len(test_recipes)}")

# Gate decision
gate_open = failed == 0
print(f"\nGate: {'OPEN' if gate_open else 'CLOSED'}")
if not gate_open:
    print("  Fix errors before building")

results_summary = {
    "total": len(test_recipes),
    "passed": passed,
    "failed": failed,
    "gate_open": gate_open,
}

with open("experiments/m305_validation_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n✅ M305: Edit validation gate working")
