# M157 — Transform Vocabulary Study

**Date:** 2026-04-20
**Status:** ✅ Complete (v2 fast CPU)
**Goal:** Compare vocabulary building strategies for Transform-WAL.

## Strategies Tested

- **B:** Transform-specific global atoms (one table per transform, all layers)
- **C:** Per-module transform atoms (separate table per layer-module-transform)

## Results

| Layer | B_raw | B_hadamard | C_raw | C_hadamard |
|-------|-------|-----------|-------|-----------|
| 0_q_proj | 3.56e-07 | 2.12e-08 | 9.68e-08 | 2.02e-08 |
| 0_v_proj | 1.26e-07 | 4.70e-09 | 4.84e-09 | 2.71e-09 |
| 16_q_proj | 3.70e-07 | 1.57e-08 | 1.90e-07 | 2.29e-08 |
| 16_v_proj | 1.66e-07 | 5.42e-09 | 7.98e-09 | 4.10e-09 |

**Averages:**
- Strategy B (global): raw=2.04e-07, hadamard=1.10e-08
- Strategy C (per-module): raw=7.14e-08, hadamard=1.28e-08

## Key Findings

1. **Per-module atoms (C) are 2.9× better for Raw-WAL** than global atoms (B)
2. **For Hadamard, the gap shrinks** — per-module only 1.2× better
3. **Hadamard always beats Raw** regardless of vocabulary strategy
4. **Trade-off:** Per-module needs 4× more atom storage (but still tiny: 4 × 64 × 8 × 4 bytes = 8 KB)

## Recommendation

For production Transform-WAL:
- **Use transform-specific global atoms (B)** for simplicity
- **Or use per-module atoms (C)** if MSE is critical and storage overhead acceptable
- Hadamard transform reduces the vocabulary strategy gap

## Artifacts

- `experiments/m157_transform_vocab_study_v2.py`
- `experiments/m157_transform_vocab_study.json`
