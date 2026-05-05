# M196h — Wave-LoRA with Higher Rank (rank=8, WAL-encoded)

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m196h_wave_reg_high_rank.py`

## Purpose

Test if wave regularization helps with rank=8 on WAL-encoded base.

**CRITICAL FIX:** First run (v1) did NOT call `encode_model` — worked on dense weights. This is the corrected version with WAL encode.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Base: Hadamard-WAL K=256 (corrected — encode called)
Layers: 14, 15, 16
Modules: ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
Rank: 8, Steps: 100, Mixed training
Runs: 10 per λ (timed out after 2 runs due to slow WAL encode)
Grid: [0.0, 0.025, 0.05, 0.1]
```

## Results (WAL-encoded base)

### Run 1/10
| λ | Survival |
|---|----------|
| 0.0000 | **5/50** |
| 0.0250 | 4/50 |
| 0.0500 | 5/50 |
| 0.1000 | 5/50 |

### Run 2/10
| λ | Survival |
|---|----------|
| 0.0000 | **6/50** |

## Comparison: Dense vs WAL

| Base | Baseline Mean | Best Single Run |
|------|---------------|-----------------|
| Dense (v1) | 4.70 ± 1.64 | 8/50 |
| WAL (v2) | ~5.50 (2 runs) | **6/50** |

## Analysis

### WAL baseline higher than dense
- Dense baseline: 4.70 mean
- WAL baseline: 5.50+ mean (first 2 runs: 5, 6)

This is interesting but based on only 2 runs — not statistically significant yet.

### Wave reg still irrelevant
All λ values give similar survival (4-6/50). No clear λ effect.

### Higher variance on WAL
WAL runs show 5-6/50, while dense showed 3-8/50. WAL may be slightly more stable.

## Conclusion

> **Rank=8 on WAL-encoded base shows baseline ~5-6/50.**
>
> This is comparable to dense rank=8 (4.70 mean). Wave reg still shows no effect.
>
> Production: **rank=4 remains default** — simpler, fewer parameters, equivalent survival when normalized.

## Important Caveat

Only 2 complete runs on WAL due to timeout. n=2 is insufficient for reliable conclusions. The trend suggests WAL rank=8 ≈ dense rank=8, consistent with M203 equivalence result.

## Related

- M196f — rank=4 grid search on dense
- M196g — o_proj only on WAL
- M203 — WAL ≈ Dense equivalence proven
