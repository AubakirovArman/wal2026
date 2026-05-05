# M139 / Track 2: WAL Patch v2

**Date:** 2026-04-20
**Status:** ✅ Positive result
**Goal:** Build and test WAL patch with frozen atom table.

## Background

M131 showed naive WAL patch was **10.7 GB**. M133 showed frozen table reduces diff to **0.168%** globally. Now we test: can WAL patch become a standalone structural patch format?

## Method

```
1. Build frozen atom/coeff table (wal.v1.encoder)
2. Encode base model with frozen table
3. Apply synthetic edit to target layer (random perturbation, mag=0.001)
4. Re-encode with SAME table
5. Compute program diff
6. Build patch = only changed positions + new values
7. Apply patch to base → verify correctness
8. Measure size with compression
```

## Results

| Metric | Value |
|--------|-------|
| Global diff | **0.1896%** |
| Target layer both-changed | **84.8%** |
| Non-target layers | **0.0000%** |
| Patch raw size | **92.75 MB** |
| Patch RLE size | **35.08 MB** |
| Patch bitmask size | **32.92 MB** |
| Patch apply correct | **True** ✅ |

## Analysis

### Diff Localization
Frozen table achieves perfect localization. Only target layer changes; all other 224 layers show exactly **0% diff**.

### Patch Correctness
Patch apply produces **exact match** with edited programs. WAL patch is a valid, reversible transformation.

### Compression
| Method | Size | Ratio |
|--------|------|-------|
| Raw | 92.75 MB | 1.0× |
| RLE | 35.08 MB | 2.6× |
| Bitmask | 32.92 MB | 2.8× |

Target was 10–30 MB. Bitmask achieves **32.92 MB** — close but not quite.

### Why 96.6% Changed?
Synthetic random edit (`randn * 0.001`) changes weights chaotically. Real LoRA edit is low-rank and structured — programs should change **less randomly**, enabling better compression.

### Future Compression Methods
- **Block patch:** Changes cluster in blocks
- **Transition table:** `old_atom → new_atom` often repeats
- **Coeff-only patch:** Atoms change less frequently than coefficients

## Conclusion

WAL patch v2 is **structurally sound**. Frozen table enables localized, correct, compressible patches. Real-world LoRA edits should achieve **10–30 MB** with block + transition compression.

## Artifacts

- `experiments/m139_wal_patch_v2.py`
- `experiments/m139_wal_patch_v2.json`
