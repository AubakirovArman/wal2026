"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M320 — Auto-Recovery

Detect and fix corrupted facts automatically.
"""
import json

print("=" * 60)
print("M320 — AUTO-RECOVERY")
print("=" * 60)

# Original facts
original = {
    1: {"question": "What is the capital of France?", "answer": "Paris"},
    2: {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    3: {"question": "What is the capital of Brazil?", "answer": "Brasília"},
    4: {"question": "What is the capital of Egypt?", "answer": "Cairo"},
    5: {"question": "What is the capital of Canada?", "answer": "Ottawa"},
}

# Simulate corruption (deep copy to avoid modifying original)
import copy
corrupted = copy.deepcopy(original)
corrupted[2]["answer"] = "Kyoto"  # Wrong answer
corrupted[4]["question"] = ""  # Empty question
corrupted[5]["answer"] = ""  # Empty answer

print("\nOriginal facts:")
for k, v in original.items():
    print(f"  [{k}] {v['question']} → {v['answer']}")

print("\nCorrupted facts:")
for k, v in corrupted.items():
    is_corrupted = v != original[k]
    status = "❌" if is_corrupted else "✅"
    print(f"  {status} [{k}] Q='{v['question'][:30]}...' A='{v['answer']}'")

# Recovery function
def recover_facts(corrupted, backup):
    """Recover corrupted facts from backup."""
    recovered = {}
    fixes = []
    
    for k, v in corrupted.items():
        if k in backup:
            # Check each field
            fixed = dict(v)
            if not fixed.get("question", "").strip():
                fixed["question"] = backup[k]["question"]
                fixes.append(f"[{k}] Restored question from backup")
            if not fixed.get("answer", "").strip():
                fixed["answer"] = backup[k]["answer"]
                fixes.append(f"[{k}] Restored answer from backup")
            
            # Check for known wrong answers (any mismatch)
            if fixed["answer"] != backup[k]["answer"]:
                fixed["answer"] = backup[k]["answer"]
                fixes.append(f"[{k}] Corrected wrong answer from backup")
            
            recovered[k] = fixed
        else:
            recovered[k] = v
    
    return recovered, fixes

recovered, fixes = recover_facts(corrupted, original)

print(f"\nRecovery applied: {len(fixes)} fixes")
for fix in fixes:
    print(f"  → {fix}")

print("\nRecovered facts:")
all_ok = True
for k, v in recovered.items():
    ok = v == original[k]
    all_ok = all_ok and ok
    status = "✅" if ok else "❌"
    print(f"  {status} [{k}] {v['question']} → {v['answer']}")

results = {
    "corrupted": sum(1 for k in corrupted if corrupted[k] != original[k]),
    "fixes_applied": len(fixes),
    "fully_recovered": all_ok,
}

with open("experiments/m320_recovery_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
if all_ok:
    print("✅ M320: Auto-recovery successful")
else:
    print("⚠️ M320: Some facts not recovered")
