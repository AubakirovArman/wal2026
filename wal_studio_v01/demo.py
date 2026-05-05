#!/usr/bin/env python3
"""
WAL Studio v0.1 — Unified Polished Demo

Demo scenario:
  1. Base model answers old fact
  2. wal edit add — add new fact
  3. wal build — compile version
  4. wal test — exact/negative/context
  5. wal diff — human-readable changes
  6. wal tag — save v1
  7. Add bad edit
  8. CI fails
  9. wal blame — find culprit
 10. wal bisect — find bad recipe
 11. wal rollback — restore v1
"""
import json, os, hashlib, time, random

random.seed(42)

WAL_DIR = ".wal_studio_demo"

def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def step(num, desc):
    print(f"\n{'─'*60}")
    print(f"  STEP {num}: {desc}")
    print(f"{'─'*60}")

# ═══════════════════════════════════════════════════════════
banner("WAL STUDIO v0.1 — UNIFIED DEMO")
print(f"  Date: 2026-05-04")
print(f"  Model: Llama-3.1-8B (simulated)")
print(f"  Layer: 16 | Rank: 4 | Steps: 100")

# ── Step 1: Base model answers old fact ───────────────────
step(1, "Base model answers old fact")
base_answers = {
    "What is the capital of France?": "Paris",
    "What is the capital of Japan?": "Tokyo",
    "What is the capital of Brazil?": "Rio de Janeiro",  # old/wrong
}
for q, a in base_answers.items():
    print(f"  Q: {q}")
    print(f"  A: {a}")
    print()

# ── Step 2: wal init ──────────────────────────────────────
step(2, "wal init — initialize project")
os.makedirs(WAL_DIR, exist_ok=True)
config = {"model": "Llama-3.1-8B", "layer": 16, "rank": 4, "seed": 42}
with open(f"{WAL_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  ✅ Project initialized in {WAL_DIR}/")
print(f"  Config: {json.dumps(config)}")

# ── Step 3: wal edit add — add new facts ──────────────────
step(3, "wal edit add — add new facts")
recipes_v1 = [
    {"id": 1, "question": "What is the capital of France?", "answer": "Paris"},
    {"id": 2, "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": 3, "question": "What is the capital of Brazil?", "answer": "Brasília"},  # corrected
]
with open(f"{WAL_DIR}/recipes.json", "w") as f:
    json.dump(recipes_v1, f, indent=2)
print(f"  ✅ Added {len(recipes_v1)} recipes")
for r in recipes_v1:
    print(f"    [{r['id']}] {r['question']} → {r['answer']}")

# ── Step 4: wal build — compile version ───────────────────
step(4, "wal build — compile model from recipes")
recipe_str = json.dumps(recipes_v1, sort_keys=True)
build_hash = hashlib.sha256(recipe_str.encode()).hexdigest()[:16]
build_time = 6.1
build_meta = {"hash": build_hash, "recipes": len(recipes_v1), "time_s": build_time}
with open(f"{WAL_DIR}/build.json", "w") as f:
    json.dump(build_meta, f, indent=2)
print(f"  ✅ Build complete: {build_hash}")
print(f"  Time: {build_time}s")

# ── Step 5: wal test — CI suite ───────────────────────────
step(5, "wal test — run CI suite")

def test_suite(recipes):
    results = {
        "exact": {"pass": 0, "total": 0},
        "paraphrase": {"pass": 0, "total": 0},
        "negative": {"pass": 0, "total": 0},
        "context": {"pass": 0, "total": 0},
    }
    # Exact match
    for r in recipes:
        results["exact"]["total"] += 1
        results["exact"]["pass"] += 1  # simulated
    # Paraphrase
    for r in recipes:
        results["paraphrase"]["total"] += 1
        results["paraphrase"]["pass"] += 1 if random.random() < 0.9 else 0
    # Negative
    negatives = [
        ("What is the capital of Germany?", "Berlin"),
    ]
    for q, a in negatives:
        results["negative"]["total"] += 1
        results["negative"]["pass"] += 1  # model doesn't hallucinate
    # Context variations
    for r in recipes:
        results["context"]["total"] += 1
        results["context"]["pass"] += 1 if random.random() < 0.95 else 0
    
    # CI score
    exact_score = results["exact"]["pass"] / results["exact"]["total"]
    para_score = results["paraphrase"]["pass"] / results["paraphrase"]["total"]
    neg_score = results["negative"]["pass"] / results["negative"]["total"]
    ctx_score = results["context"]["pass"] / results["context"]["total"]
    ci = exact_score * 0.3 + para_score * 0.3 + neg_score * 0.2 + ctx_score * 0.2
    return results, ci

test_results, ci_score = test_suite(recipes_v1)
print(f"  Exact:      {test_results['exact']['pass']}/{test_results['exact']['total']}")
print(f"  Paraphrase: {test_results['paraphrase']['pass']}/{test_results['paraphrase']['total']}")
print(f"  Negative:   {test_results['negative']['pass']}/{test_results['negative']['total']}")
print(f"  Context:    {test_results['context']['pass']}/{test_results['context']['total']}")
print(f"  ──────────────────────────")
print(f"  CI Score:   {ci_score:.3f}")
print(f"  Verdict:    {'✅ PASS' if ci_score >= 0.7 else '❌ FAIL'}")

# ── Step 6: wal diff — human-readable changes ─────────────
step(6, "wal diff — show human-readable changes")
print("  Changes from base model:")
print(f"    [FIXED] Brazil: Rio de Janeiro → Brasília")
print(f"    [ADDED] France: Paris")
print(f"    [ADDED] Japan: Tokyo")

# ── Step 7: wal tag — save v1 ─────────────────────────────
step(7, "wal tag v1.0 — save release")
tags = {"v1.0": build_hash}
with open(f"{WAL_DIR}/tags.json", "w") as f:
    json.dump(tags, f, indent=2)
print(f"  ✅ Tagged v1.0 → {build_hash}")

# ── Step 8: Add bad edit ──────────────────────────────────
step(8, "Add bad edit (wrong fact)")
recipes_v2 = recipes_v1 + [
    {"id": 4, "question": "What is the capital of Egypt?", "answer": "Alexandria"},  # WRONG
]
with open(f"{WAL_DIR}/recipes.json", "w") as f:
    json.dump(recipes_v2, f, indent=2)
print(f"  ⚠️  Added recipe [4]: Capital of Egypt → Alexandria")
print(f"      (Correct answer: Cairo)")

# ── Step 9: wal test — CI fails ───────────────────────────
step(9, "wal test — CI detects bad edit")

# Negative test catches it
negatives_v2 = [
    ("What is the capital of Egypt?", "Cairo"),
]
test_results_v2, ci_score_v2 = test_suite(recipes_v2)
# Simulate: negative test catches wrong fact
# Model trained on "Alexandria" will answer Alexandria, but negative test checks
# if it also answers Cairo correctly. With rehearsal, it might.
# Let's simulate: negative test FAILS because model now prefers Alexandria
neg_pass = 0  # fails because model answers Alexandria
neg_total = 1
ci_v2 = 0.3 + 0.3 + 0.0 * 0.2 + 0.2  # exact+para still OK, negative fails

print(f"  Exact:      {test_results_v2['exact']['pass']}/{test_results_v2['exact']['total']}")
print(f"  Paraphrase: {test_results_v2['paraphrase']['pass']}/{test_results_v2['paraphrase']['total']}")
print(f"  Negative:   {neg_pass}/{neg_total}  ← FAILS")
print(f"  Context:    {test_results_v2['context']['pass']}/{test_results_v2['context']['total']}")
print(f"  ──────────────────────────")
print(f"  CI Score:   {ci_v2:.3f}")
print(f"  Verdict:    ❌ FAIL — negative test caught bad edit")

# ── Step 10: wal blame — find culprit ─────────────────────
step(10, "wal blame — identify responsible edit")
print(f"  Failed test: 'What is the capital of Egypt?'")
print(f"  Expected: Cairo")
print(f"  Got: Alexandria")
print(f"  ──────────────────────────")
print(f"  🔍 Most likely culprit: Recipe [4]")
print(f"     Question: 'What is the capital of Egypt?'")
print(f"     Answer:   'Alexandria' ← WRONG")
print(f"  ✅ Blame identified: edit #4")

# ── Step 11: wal bisect — find bad recipe ─────────────────
step(11, "wal bisect — binary search for bad recipe")
print(f"  History: v1.0 (3 recipes) → v2 (4 recipes)")
print(f"  Testing midpoint...")
print(f"    Midpoint: recipe [4] added")
print(f"    CI before [4]: PASS")
print(f"    CI after [4]:  FAIL")
print(f"  ✅ Bisect found: recipe [4] is first bad commit")

# ── Step 12: wal rollback — restore v1 ────────────────────
step(12, "wal rollback v1.0 — restore good version")
with open(f"{WAL_DIR}/tags.json") as f:
    tags = json.load(f)
target = tags["v1.0"]
with open(f"{WAL_DIR}/recipes.json", "w") as f:
    json.dump(recipes_v1, f, indent=2)
print(f"  ✅ Rolled back to v1.0 ({target})")
print(f"  Recipes restored: {len(recipes_v1)}")

# Re-test
test_results_final, ci_final = test_suite(recipes_v1)
print(f"  Re-test CI: {ci_final:.3f} — ✅ PASS")

# ═══════════════════════════════════════════════════════════
banner("DEMO COMPLETE")
print(f"  Total steps: 12")
print(f"  Recipes managed: {len(recipes_v1)} (v1.0)")
print(f"  Bad edit caught: Recipe [4]")
print(f"  Rollback time: ~4.3s")
print(f"  CI gate: {'PASS' if ci_final >= 0.7 else 'FAIL'}")
print(f"\n  WAL Studio v0.1 ready for demonstration.")
print(f"{'='*60}\n")
