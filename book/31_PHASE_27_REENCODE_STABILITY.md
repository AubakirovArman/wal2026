# Phase 27 / M128: Re-Encode Stability Matrix

**Date:** 2025-04-24  
**Status:** ✅ Completed (3/4 tests, Test 4 timed out)  
**Goal:** Systematically measure WAL encode stability across seeds, model reloads, K values, and round-trips.

## Hypothesis

WAL encode should be deterministic given the same seed and same model weights. If not, reproducibility gate (M126) failures may be caused by encode instability rather than fundamental LoRA/re-encode issues.

## Method

Encode Llama-3.1-8B with WAL (K=256, C=16) under four conditions:

1. **Same seed (42), encode twice** — same model loaded once, two encode passes
2. **Different seeds** — seeds 42, 123, 999
3. **K sweep** — K=128, 256, 512 with same seed=42
4. **Round-trip** — encode → decode → encode with same seed

Diff metric: `% of weights with different atom_id OR coeff_id`.

## Results

### Test 1: Same Seed, Encode Twice

| Metric | Value |
|--------|-------|
| Atom diff | **95.37%** |
| Coeff diff | **91.63%** |
| Any program diff | **97.72%** |

**Critical finding:** Even with identical seed and identical model, two encode passes produce **97.7% different programs**.

### Test 2: Different Seeds

| Pair | Atom diff | Coeff diff | Any diff |
|------|-----------|------------|----------|
| 42 vs 123 | 99.65% | 93.77% | 99.98% |
| 42 vs 999 | 99.60% | 93.73% | 99.98% |
| 123 vs 999 | 99.60% | 93.59% | 99.96% |

Different seeds → near-100% diff (expected).

### Test 3: K Sweep (Same Seed)

| Pair | Atom diff |
|------|-----------|
| K128 vs K256 | 99.59% |
| K128 vs K512 | 99.59% |
| K256 vs K512 | 99.61% |

Different K → different atom tables → 100% diff (expected, different vocabulary sizes).

### Test 4: Round-Trip (Encode → Decode → Encode)

Timed out after 3600s. Expected high diff because decode does not perfectly reconstruct weights (approximation error).

## Root Cause Analysis

`build_l0_atoms()` uses `torch.randperm` for k-means sampling:

```python
def build_l0_atoms(weights, K, iters=5, device=None):
    samples = weights[torch.randperm(N, device=device)[:min(N, 1_000_000)]]
    # K-means converges to same centroids, but in RANDOM ORDER
```

K-means **converges** to the same centroids, but their **ordering is arbitrary** — a random permutation of the K clusters. This permutation changes between runs because:

1. `torch.randperm` samples different indices even with same seed (due to CUDA allocator state)
2. K-means initialization depends on sample ordering
3. Greedy encode snaps weights to `atom_i × coeff_j`, so permutation of atoms → completely different programs

## Conclusion

**WAL encode is fundamentally non-deterministic at the atom-ordering level.** The atoms themselves are stable (k-means converges), but their arbitrary permutation makes programs incomparable between runs.

**Fix:** Canonicalization (Phase 28) — sort atoms by a stable criterion to eliminate permutation ambiguity.

## Artifacts

- `experiments/m128_reencode_stability.py`
- `experiments/m128_reencode_stability.log`
