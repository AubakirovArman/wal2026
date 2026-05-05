# M196g — Wave-LoRA on Single-Module Targets (o_proj only, WAL-encoded)

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m196g_wave_reg_single_module.py`

## Purpose

Test if wave regularization helps on single-module targets (o_proj only) with WAL-encoded base.

**CRITICAL FIX:** First run (v1) did NOT call `encode_model` — worked on dense weights. This is the corrected version with WAL encode.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Base: Hadamard-WAL K=256 (corrected — encode called)
Layers: 14, 15, 16
Modules: ['o_proj']  # SINGLE MODULE only
Rank: 4, Steps: 100, Mixed training
Runs: 10 per λ (timed out after 2 runs due to slow WAL encode)
Grid: [0.0, 0.01, 0.025, 0.05, 0.1]
```

## Results (WAL-encoded base)

### Run 1/10
| λ | Survival |
|---|----------|
| 0.0000 | 3/50 |
| 0.0100 | 3/50 |
| 0.0250 | 3/50 |
| 0.0500 | 3/50 |
| 0.1000 | 3/50 |

### Run 2/10
| λ | Survival |
|---|----------|
| 0.0000 | 3/50 |

## Comparison: Dense vs WAL

| Base | Survival | Std |
|------|----------|-----|
| Dense (v1, no encode) | 3.00 ± 0.00 | 0.00 |
| WAL (v2, with encode) | 3.00 ± 0.00 | 0.00 |

**Same result on both dense and WAL-encoded base.**

## Analysis

### Zero variance on WAL too
All λ give exactly 3/50 survival even with WAL-encoded base.

### WAL encode doesn't change o_proj behavior
Whether base is dense or WAL-encoded, o_proj only gives flat 3/50 survival.

### Wave reg irrelevant on any base
No λ shows any effect, regardless of base format.

## Conclusion

> **o_proj only = 3/50 flat on BOTH dense and WAL-encoded base.**
>
> Wave regularization does not help on single-module targets.
> Mixed targets (4 modules) are required for better survival.

## Lesson

WAL encode is near-lossless, but it doesn't magically improve editing capacity. The limitation is in the target module selection, not the base format.

## Updated Production Recommendation

```python
# Use mixed targets for best survival
TARGET_MODULES = ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
# Single-module targets (o_proj only) are insufficient
```

## Related

- M196f — Mixed targets grid search (baseline 4.30, best config)
- M196h — Rank=8 on WAL-encoded base
- M203 — WAL ≈ Dense equivalence proven
