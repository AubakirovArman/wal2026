# M154 — Fix Hadamard Properly

**Date:** 2026-04-20
**Status:** ✅ Complete
**Goal:** Implement orthonormal Hadamard transform for WAL v2.

## Key Fixes

1. **Orthonormal normalization:** `H_norm = H / sqrt(n)`
2. **Power-of-2 padding:** Corrected padding logic (was padding wrong dimension)
3. **Full tensor preservation:** Return FULL transformed tensor including padding region; crop only after inverse transform

## Tests

| Test | Description | Result |
|------|-------------|--------|
| 1 | Inverse exactness: H.T @ H = I | ✅ n=1..1024 |
| 2 | Energy preservation: ||W_pad||_F = ||W_t||_F | ✅ All shapes |
| 3 | Padding behavior: no artifacts | ✅ mse=1.80e-13 |
| 4 | Hadamard-WAL vs Raw-WAL | ✅ See below |

## WAL Comparison

| Shape | Raw MSE | Hadamard MSE | Ratio |
|-------|---------|-------------|-------|
| (100, 200) | 1.04e-06 | 5.08e-07 | **0.488** (2× better) |
| (512, 768) | 8.43e-07 | 8.08e-07 | **0.957** (nearly equal) |

## Why the Fix Matters

Previous M142 Hadamard failed because:
1. **Wrong padding:** `pad_to_power_of_2` padded the wrong dimension
2. **Crop before inverse:** Truncating W_t destroyed transform-space information
3. **No normalization:** H was not orthonormal

## Production Viability

Hadamard is now viable as a production transform:
- Real-valued (no complex numbers)
- Fast (matrix multiplication)
- No metadata storage needed (deterministic from size)
- Self-inverse (up to normalization)
- MSE comparable to Raw-WAL, better on small matrices

## Artifacts

- `experiments/m154_fix_hadamard.py`
- `experiments/m154_fix_hadamard.json`
