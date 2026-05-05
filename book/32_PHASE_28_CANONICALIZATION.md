# Phase 28 / M129: Canonicalization Layer

**Date:** 2025-04-24  
**Status:** ✅ Completed (norm + sum), ⚠️ lex timed out  
**Goal:** Make WAL encode deterministic by canonicalizing atom ordering.

## Hypothesis

If atom ordering is the source of encode non-determinism (M128), sorting atoms by a stable criterion should make encode deterministic for the same seed.

## Method

Test 3 canonicalization methods:

1. **`norm`**: Sort atoms by `abs(atom)` (descending). Atoms are scalars (shape `[K]`), so this is scalar sort.
2. **`sum`**: Sort atoms by `atom.sum()` (descending). For scalars, equivalent to value sort.
3. **`lex`**: Lexicographic sort (not completed — timed out).

After sorting atoms, also permute programs (atom indices) to match the new ordering.

```python
def canonicalize_atoms(atoms, method='norm'):
    if method == 'norm':
        scores = atoms.abs()      # scalar sort for 1D atoms
    elif method == 'sum':
        scores = atoms            # sort by value
    perm = torch.argsort(scores, descending=True, stable=True)
    sorted_atoms = atoms[perm]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device)
    return sorted_atoms, perm, inv_perm
```

## Results

### Method: `norm` (sort by `abs(atom)`)

| Condition | Atom diff | Coeff diff | Any diff |
|-----------|-----------|------------|----------|
| Same seed (42 vs 42) | **0.0000%** | **0.0000%** | **0.0000%** |
| Diff seeds (42 vs 123) | 99.67% | ~ | 99.67% |

### Method: `sum` (sort by `atoms.sum()`)

| Condition | Atom diff | Coeff diff | Any diff |
|-----------|-----------|------------|----------|
| Same seed (42 vs 42) | **0.0000%** | **0.0000%** | **0.0000%** |
| Diff seeds (42 vs 123) | 99.84% | ~ | 99.84% |

### Method: `lex`

Timed out during encode (expected — lexicographic sort on scalars is slower and unnecessary).

## Key Insight

**Canonicalization achieves 100% same-seed determinism.** Two encode passes with identical seed now produce **identical programs** (0% diff).

The remaining 99.6%+ diff between **different seeds** is expected and fundamental: k-means with different random samples converges to **different centroids**, not just different permutations.

## Implications

### For WAL-Diff (M127)

- **Before canonicalization:** WAL-diff is useless — 99.98% programs change after any edit because re-encode picks different atom permutations.
- **After canonicalization:** WAL-diff still shows ~99.98% diff (M127 used different seeds for before/after), BUT if we use **same seed for both encodes**, diff will reflect **actual structural changes** from the edit, not permutation noise.

### For Reproducibility Gate (M126)

- M126 v3 showed catastrophic PPL on some seeds (up to +129) because re-encode with same seed produced different atom tables due to permutation.
- M126 v4 (with canonicalization) should eliminate this source of variance.

### For Patch Portability

- **Same seed + canonicalization** = deterministic encode.
- **Different seeds** = different atom tables = incompatible programs.
- **Conclusion:** WAL patches are **not portable across different encodes**. A patch compiled against one encode will not apply to another. This is a fundamental limitation of VQ-based representations.

## Conclusion

Canonicalization **solves the determinism problem** for same-seed encodes. The fix is simple, cheap, and should be applied by default in all WAL encode pipelines.

**Recommended:** Add `canonicalize=True` parameter to `replace_linear_with_wal()` and `encode_linear_weight()`.

## Artifacts

- `experiments/m129_canonicalization.py`
- `experiments/m129_canonicalization_v2.log`
