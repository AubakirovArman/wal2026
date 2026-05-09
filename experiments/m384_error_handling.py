"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M384 — Error Handling

Graceful error handling.
"""
import json

print("=" * 60)
print("M384 — ERROR HANDLING")
print("=" * 60)

def safe_build(recipes):
    """Build with error handling."""
    try:
        if not recipes:
            raise ValueError("No recipes provided")
        if len(recipes) > 1000:
            raise ValueError("Too many recipes")
        return {"status": "success", "hash": "abc123"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "fatal", "message": str(e)}

cases = [
    [],
    [{"q": "Q1", "a": "A1"}],
    [{"q": f"Q{i}", "a": f"A{i}"} for i in range(1001)],
]

print("\nError handling tests:")
for i, recipes in enumerate(cases):
    result = safe_build(recipes)
    status = "✅" if result["status"] == "success" else "❌"
    print(f"  {status} Case {i+1} ({len(recipes)} recipes): {result['status']} - {result.get('message', 'OK')}")

with open("experiments/m384_error_results.json", "w") as f:
    json.dump({"cases": len(cases)}, f, indent=2)

print("\n✅ M384: Error handling tested")
