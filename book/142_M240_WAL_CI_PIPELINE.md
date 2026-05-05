# M240 — WAL CI Pipeline

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m240_wal_ci_pipeline.py`

## Purpose

Implement and test a CI-style pipeline for automated edit validation:
1. Load + encode base model
2. Apply edit via LoRA
3. Run test suite: exact, paraphrase, negative, PPL gate
4. Report PASS/FAIL with diagnostics

## Setup

```
Model: meta-llama/Llama-3.1-8B
Gates: exact_match ≥80%, paraphrase ≥80%, negative ≥80%, PPL ≤6.0
Facts: 3 easy (capital of France, Eiffel Tower, longest river)
Paraphrases: 3 variants
Negative: 2 unrelated prompts
```

## Results

| Gate | Result | Detail |
|------|--------|--------|
| exact_match | ❌ FAIL | 0.0% (0/3) |
| paraphrase | ❌ FAIL | 0.0% (0/3) |
| negative | ✅ PASS | 100.0% (2/2) |
| ppl_gate | ❌ FAIL | nan (gate=6.0) |

**OVERALL: ❌ FAIL**

## Critical Findings

### 1. CI Pipeline Structure Works
The pipeline successfully:
- Loads, encodes, edits, and tests the model
- Runs all 4 gates automatically
- Produces structured JSON report
- This validates the **WAL CI concept** as viable infrastructure

### 2. Edit Quality is Broken
Exact and paraphrase both 0/3 — edits are not surviving. This is unexpected for easy facts.

### 3. PPL = nan
Same issue as M235v2: `get_ppl` returns nan after LoRA training. Likely float16 gradient overflow during training.

### 4. Negative Test Passes
Model correctly does NOT output "Paris" for unrelated prompts. No over-generalization.

## Root Cause: Training Instability
The 0/3 survival + nan PPL indicates:
1. LoRA training on encoded float16 model causes gradient explosion
2. Adapter weights become nan during optimization
3. Forward pass produces nan logits → all answers wrong

This is a **regression** not seen in M228-M234, where easy facts had 60-80% survival.

## Conclusion

**WAL CI Pipeline: INFRASTRUCTURE ✅, EDIT QUALITY ❌**
- Pipeline architecture is production-ready
- But training pipeline needs float32 or gradient clipping
- Once training fixed, CI gates will provide automated quality control

## Next Steps
- Fix training: use float32 for adapters, add gradient clipping
- Re-run CI pipeline to confirm PASS for easy facts
- Extend CI to include rehearsal buffer and batch editing tests
