# M1B PROBE ROWNORM

## Date
2026 (exact date from git log or experiment run)

## Goal
M1 probe with per-row normalization (mandatory for real LLM weights).

## Configuration
iters=20, threshold=0.0

## Method / What was tested
Strategy mirrors Route B's row_scale:
    row_max[n]   = max(|W[n,:]|) clamp_min(eps)
    W_norm[n,k]  = W[n,k] / row_max[n]            # max |.| == 1 per row
    encode W_norm with a single ladder seeded near 1.0
    decode W_hat = W_norm_hat * row_max[:, None]

## Result
Encode test.

## Artifacts
- `experiments/m1b_probe_rownorm.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Row-norm calibration. Verified that per-row normalization stabilizes encoding.

**Notes:** w_norm = w / max(abs(row)). Foundation for all subsequent scalar encoding.
