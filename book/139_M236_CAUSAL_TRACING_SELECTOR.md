# M236 — Causal Tracing Selector

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m236_causal_tracing_selector.py`

## Purpose

Test whether ROME-style causal tracing can identify optimal layers for factual editing, compared to hardcoded [14-16] and activation-guided [28-31] selections.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Method: Corrupt prompt → patch activations layer-by-layer → measure answer probability restoration
Facts: 3 easy (capital of France, Eiffel Tower, longest river)
Compare: causal-selected vs hardcoded [14,15,16]
```

## Results

| Strategy | Layers | PPL Δ | Survival |
|----------|--------|-------|----------|
| Causal-selected | [0, 1, 2] | +0.0360 | 3/3 |
| Hardcoded | [14, 15, 16] | +0.2498 | 3/3 |

### Causal Tracing Restoration Scores
All layers showed identical restoration (~1.0) across all facts:
- "What is the capital of France?": 1.0002 (all 32 layers)
- "Where is the Eiffel Tower located?": 0.9989 (all 32 layers)
- "What is the longest river in the world?": 1.0000 (all 32 layers)

## Critical Finding: Causal Tracing BROKEN

**The implementation produces flat restoration scores across all layers.** This indicates a bug in the tracing logic — likely the corrupted run still produces the correct answer due to prompt structure, or the restoration metric is not properly normalized.

### Why this matters
1. True causal tracing (as in ROME/MEMIT) requires careful noise injection and probability comparison
2. Our lite implementation does not capture the true causal structure
3. Layer [0,1,2] selected by "causal" method are clearly wrong for factual editing (early layers handle tokenization/embedding, not facts)

### Comparison with M230
M230 showed activation-guided selection [28-31] → 0/5 survival. Here causal-selected [0,1,2] gives 3/3 — but only because these are easy facts that work on ANY layers. The selection method is not validated.

## Conclusion

**Causal tracing selector: NOT VIABLE in current form.**
- Flat restoration scores = implementation bug
- Layer [0,1,2] cannot be correct for factual editing
- Hardcoded [14-16] remains the validated choice
- True ROME/MEMIT tracing requires much more sophisticated noise modeling

## Next Steps
- Fix causal tracing (proper Gaussian noise, normalized restoration)
- Test on hard facts where layer choice actually matters
- Compare with attention-weighted tracing (not just MLP activations)
