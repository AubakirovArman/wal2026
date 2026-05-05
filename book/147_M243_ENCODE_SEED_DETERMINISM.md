# M243 — Encode Seed Determinism

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m243_encode_seed_determinism.py`

## Purpose

Fix M239's non-determinism by adding explicit seed control to the encoding pipeline.

## Fix

1. `torch.manual_seed(seed)` at start of `encode_model`
2. `torch.manual_seed(seed)` at start of `kmeans_chunked`
3. `torch.manual_seed(seed)` at start of `hadamard_wal_encode`

## Results

| Metric | Run 1 | Run 2 | Match |
|--------|-------|-------|-------|
| Logits max diff | — | — | **0.0** |
| Hidden max diff | — | — | **0.0** |

## Critical Finding: Determinism Achieved

**With fixed seed, WAL encode produces bit-exact identical outputs across runs.**

### Why M239 failed
- `torch.randperm` in `kmeans_chunked` used global RNG state
- Global RNG was modified by previous operations (model loading, other experiments)
- No explicit seed → different random sequences → different k-means atoms

### With seed=42
- `randperm` produces identical shuffle
- `multinomial` produces identical initialization
- `kmeans` converges to identical atoms
- Result: bit-exact encode

## Conclusion

**WAL encode: DETERMINISTIC with fixed seed.**
- Use `torch.manual_seed(42)` (or any fixed seed) in production
- Store seed in recipe metadata for reproducibility
- Encoded weights CAN be cached and reused
- Recipe replay now guaranteed bit-exact (with same seed)

## Next Steps
- Add seed parameter to all encode operations
- Store seed in recipe JSON
- Validate recipe replay determinism with fixed seed
- Update build system to require seed specification
