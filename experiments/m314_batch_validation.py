"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M314 — Batch Validation

Validate large batches of recipes before building.
"""
import json

print("=" * 60)
print("M314 — BATCH VALIDATION")
print("=" * 60)

def validate_recipe(r):
    """Validate a single recipe."""
    errors = []
    if not r.get("question", "").strip():
        errors.append("empty question")
    if not r.get("answer", "").strip():
        errors.append("empty answer")
    if len(r.get("question", "")) > 200:
        errors.append("question too long")
    if len(r.get("answer", "")) > 100:
        errors.append("answer too long")
    return errors

# Large batch
batch = []
for i in range(100):
    if i % 10 == 0:
        # Insert bad recipe
        batch.append({"id": i, "question": "", "answer": "bad"})
    elif i % 15 == 0:
        batch.append({"id": i, "question": "Q" * 250, "answer": "A"})
    else:
        batch.append({"id": i, "question": f"Question {i}?", "answer": f"Answer {i}"})

print(f"\nValidating batch of {len(batch)} recipes...")

invalid = []
for r in batch:
    errs = validate_recipe(r)
    if errs:
        invalid.append({"id": r["id"], "errors": errs})

print(f"  Valid: {len(batch) - len(invalid)}/{len(batch)}")
print(f"  Invalid: {len(invalid)}/{len(batch)}")

if invalid:
    print(f"\n  First 5 invalid recipes:")
    for inv in invalid[:5]:
        print(f"    [{inv['id']}] {', '.join(inv['errors'])}")

# Gate
gate_open = len(invalid) == 0
print(f"\n  Gate: {'OPEN' if gate_open else 'CLOSED'}")

results = {
    "total": len(batch),
    "valid": len(batch) - len(invalid),
    "invalid": len(invalid),
    "gate_open": gate_open,
}

with open("experiments/m314_batch_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M314: Batch validation working")
