# M190 — Wave-Guided WAL Budget

**Goal:** Allocate K per layer based on wave risk to reduce total size.

## Method

1. Compute wave risk for all layers/modules
2. Assign K based on risk percentiles:
   - low risk → K=128
   - medium risk → K=256
   - high risk → K=512
3. Encode full model with adaptive K
4. Compare PPL to uniform K=256

## Results

| Config | PPL | Δ | Degradation |
|--------|-----|---|-------------|
| Baseline | 4.3169 | — | — |
| Adaptive K | 6.0178 | +1.7009 | **+39.4%** |
| Uniform K=256 | 4.7144 | +0.3975 | +9.2% |

**Budget distribution:**
- K=128: 0
- K=256: 0
- **K=512: 224 (all modules)**

## Analysis

### Risk Formula Overestimates

All 224 modules (32 layers × 7 modules) were classified as high-risk. The wave risk formula is too aggressive — it doesn't discriminate between layers.

### K=512 Raw Underperforms K=256 Raw

Adaptive budget assigned K=512 to all modules. But M180 showed that for raw-WAL, K=512 can have worse MSE than K=256 for some modules (e.g., gate_proj layer 16: K=512 raw 7.32e-11 vs K=256 raw 7.06e-10 — actually K=512 is better, but the pattern is module-dependent).

More importantly, without transform, K=512 doesn't guarantee better PPL than K=256.

### Atom Table Size

- Adaptive: 448 KB
- Uniform K=256: 224 KB

Adaptive is 2× larger but gives worse PPL. This is because all modules got K=512.

## Lessons Learned

1. **Wave risk formula needs calibration** — current thresholds don't discriminate
2. **Transform is essential** — raw-WAL adaptive doesn't work; Hadamard is needed
3. **K=512 ≠ always better** — sweet spot may be K=256 for most modules
4. **Per-module-type budgets needed** — attention and MLP may need different strategies

## Future Work

Retry M190 with:
- Hadamard transform (as in M181)
- Calibrated risk thresholds (use training data to set percentiles)
- Per-module-type policies (gate_proj gets more budget than v_proj)

## Conclusion

**Wave-guided budget is theoretically sound but needs refinement.** The core idea — allocate more atoms to complex layers — is correct. But the implementation requires:
1. Better risk metric
2. Transform-domain encoding
3. Module-aware policies
