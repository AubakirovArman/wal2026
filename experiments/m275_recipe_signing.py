"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M275 — Recipe Signing / Tamper Detection

Hypothesis: Recipes can be cryptographically signed to detect
tampering and ensure integrity.
"""
import os, sys, json, hashlib, hmac

def sign_recipe(recipe, secret_key=b"wal_secret_key"):
    """Sign a recipe with HMAC-SHA256."""
    recipe_str = json.dumps(recipe, sort_keys=True)
    signature = hmac.new(secret_key, recipe_str.encode(), hashlib.sha256).hexdigest()[:32]
    return {**recipe, "_signature": signature}

def verify_recipe(signed_recipe, secret_key=b"wal_secret_key"):
    """Verify recipe signature."""
    recipe_copy = {k: v for k, v in signed_recipe.items() if not k.startswith("_")}
    recipe_str = json.dumps(recipe_copy, sort_keys=True)
    expected = hmac.new(secret_key, recipe_str.encode(), hashlib.sha256).hexdigest()[:32]
    return signed_recipe.get("_signature") == expected

print("=" * 60)
print("M275 — Recipe Signing / Tamper Detection")
print("=" * 60)

# Test recipe
recipe = {
    "id": 0,
    "question": "What is the capital of France?",
    "answer": "Paris",
    "layer_idx": 16,
    "rank": 4,
    "lr": 5e-5,
    "steps": 100,
    "seed": 42,
}

print("\n[Sign] Creating signed recipe...")
signed = sign_recipe(recipe)
print(f"  Signature: {signed['_signature']}")

print("\n[Verify] Checking signature...")
ok = verify_recipe(signed)
print(f"  Valid: {'✅ YES' if ok else '❌ NO'}")

print("\n[Tamper] Modifying answer and re-verifying...")
tampered = {**signed, "answer": "London"}
ok_tampered = verify_recipe(tampered)
print(f"  Tampered valid: {'❌ NO (correctly rejected)' if not ok_tampered else '✅ YES (wrong!)'}")

print("\n[Tamper] Modifying seed and re-verifying...")
tampered2 = {**signed, "seed": 43}
ok_tampered2 = verify_recipe(tampered2)
print(f"  Tampered valid: {'❌ NO (correctly rejected)' if not ok_tampered2 else '✅ YES (wrong!)'}")

print("\n" + "=" * 60)
if ok and not ok_tampered and not ok_tampered2:
    print("🎯 RECIPE SIGNING WORKS — tampering detected")
else:
    print("⚠️  Signing has issues")
print("=" * 60)

results = {
    "original_valid": ok,
    "tampered_answer_detected": not ok_tampered,
    "tampered_seed_detected": not ok_tampered2,
}
with open("experiments/m275_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m275_results.json")
