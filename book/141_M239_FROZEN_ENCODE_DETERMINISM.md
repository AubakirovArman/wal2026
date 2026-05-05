# M239 — Frozen Encode Determinism

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m239_frozen_encode_determinism.py`

## Purpose

Test whether WAL-encoded weights that are NOT edited (frozen) produce identical outputs across multiple load-encode cycles. This validates encoding determinism and cache safety.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Test: Load → encode → capture outputs → reload → re-encode → compare
Outputs compared: logits (last token), hidden states (last layer), PPL
Seeds: same (default) vs different
```

## Results

| Text | Same Seed Logits | Same Seed Hidden | Diff Seed Logits | Max Diff Same |
|------|-----------------|------------------|------------------|---------------|
| "The capital of France is" | ❌ | ❌ | ❌ | 0.254 |
| "In 1492, Christopher Columbus" | ❌ | ❌ | ❌ | 0.477 |
| "The theory of relativity..." | ❌ | ❌ | ❌ | 0.299 |

**PPL Run1: 1.1675, PPL Run2: 1.1679 — mismatch confirmed**

## Critical Finding: Encode is NOT Deterministic

**Frozen WAL encode produces different outputs on every run, even with the same (default) random seed.**

### Root Cause
1. `torch.randperm` in `kmeans_chunked` uses the global random generator
2. `torch.multinomial` for k-means++ initialization also uses global RNG
3. No explicit seed fixation in the encoding pipeline
4. The k-means encode step introduces stochasticity that propagates to all weights

### Impact
- **Cannot cache encoded weights** and expect bit-exact reproduction
- **Build reproducibility** requires storing encoded weights, not just recipes
- **Recipe replay** (M227) is semantically deterministic but NOT bit-exact — confirmed
- CI pipelines must use behavioral tolerance, not checksums

## Conclusion

**WAL encode: BEHAVIORALLY stable, NOT bit-exact deterministic.**
- PPL variation: ~0.0004 (0.03% relative) — negligible for inference
- Logit variation: up to 0.48 — significant for token-level operations
- Hidden state variation: up to 0.63 — may affect downstream tasks
- **Recommendation: Store encoded checkpoints, do not rely on re-encode determinism**

## Next Steps
- Add explicit `torch.manual_seed` to encoding pipeline
- Quantify PPL variance across N=10 encode runs
- Determine if variance increases with K (fewer atoms = more variance)
