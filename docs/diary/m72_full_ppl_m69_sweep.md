# M72 FULL PPL M69 SWEEP

## Date
2026 (exact date from git log or experiment run)

## Goal
M72: Full-model PPL sweep for M69 position-specific quantization.

## Configuration
K=16, num_steps=0

## Method / What was tested
Tests K=16,32,64,128,256. All in one run: load once, encode in-place, PPL, restore.
Uses uniform quantization (fast encode).

## Result
PPL evaluation.
Likely negative result Has PASS/FAIL asserts

## Artifacts
- `experiments/m72_full_ppl_m69_sweep.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Full PPL sweep for M69 configurations.

**Notes:** Verified catastrophic degradation at low bitrates.


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
- Time: .1
- Time: .1
