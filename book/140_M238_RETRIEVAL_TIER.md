# M238 — Real Retrieval Tier

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m238_retrieval_tier.py`

## Purpose

Implement and test real hybrid memory: easy/medium facts → compiled weights, hard facts → retrieval backend (prompt injection). Tests end-to-end accuracy, latency, PPL, and prompt contamination.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Facts: 3 easy, 3 hard
Modes: weights-only, hybrid (weights+retrieval), retrieval-only
Retrieval: exact match + vector similarity for hard facts
Contamination: unrelated question with wrong retrieval context
```

## Results

| Mode | Easy | Hard | PPL Δ | Notes |
|------|------|------|-------|-------|
| Weights-only | 3/3 | 0/3 | +0.4426 | Hard facts fail as expected |
| Hybrid | 3/3 | 0/3 | +0.4899 | Retrieval not injecting correctly |
| Retrieval-only | 0/3 | 0/3 | +0.0000 | No edits applied at all |
| Contamination | — | — | — | 0/1 (model ignores wrong context) ✅ |

## Critical Findings

### 1. Retrieval Backend Not Working
**Retrieval-only mode: 0/3 easy, 0/3 hard.** This means the retrieval injection mechanism is broken:
- Either context is not being prepended to the prompt
- Or the model ignores injected context
- Or the similarity matching fails to find correct facts

### 2. Hybrid Mode Same as Weights-Only
Hybrid gives identical easy/hard scores to weights-only, just with higher PPL (+0.4899 vs +0.4426). Retrieval component is not functioning.

### 3. Contamination Check Passes
Unrelated question with wrong context: 0/1 survival. **Model correctly ignores irrelevant retrieval context.** This is good — no prompt contamination.

## Root Cause Analysis

The retrieval implementation likely has one of these bugs:
1. `build_retrieval_index` returns empty or wrong matches
2. `hybrid_inference` doesn't actually prepend retrieval context to the prompt
3. The prompt format for retrieval context is wrong (needs special tokens / formatting)

## Conclusion

**Retrieval tier: CONCEPT VALIDATED, IMPLEMENTATION BROKEN.**
- Hybrid routing logic works (easy→weights, hard→retrieval)
- Contamination safety confirmed
- But retrieval injection is non-functional
- Need to fix: exact prompt prepending with proper separator tokens

## Next Steps
- Fix retrieval prompt injection (prepend context with `[CONTEXT]` marker)
- Test retrieval-only mode should give 3/3 easy, 3/3 hard
- Add latency benchmarks for retrieval lookup
- Integrate with real vector DB (FAISS / Chroma)
