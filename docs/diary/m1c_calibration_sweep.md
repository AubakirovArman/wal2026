# M1C CALIBRATION SWEEP

## Date
2026 (exact date from git log or experiment run)

## Goal
M1c: sweep calibration modes on mlp.up_proj (the hardest layer-0 tensor).

## Configuration
threshold=0.0

## Method / What was tested
See `experiments/m1c_calibration_sweep.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m1c_calibration_sweep.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Grid search over scale and zero-point parameters.

**Notes:** Found optimal calibration parameters for geometric seed ladder [1.0, 0.5, 0.25, ...].
