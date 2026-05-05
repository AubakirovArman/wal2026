# M70 PPL POSITION SPECIFIC

## Date
2026 (exact date from git log or experiment run)

## Goal
M70: Full 70B PPL with position-specific scalar quantization.

## Configuration
K=128, num_steps=0

## Method / What was tested
Tests K=128 (7 bits/weight) and K=256 (8 bits/weight).
Uses M61 PPL parameters: max_length=2048, stride=512, max_samples=16.

## Result
PPL evaluation.

## Artifacts
- `experiments/m70_ppl_position_specific.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Position-specific PPL tests.

**Notes:** Part of M69-M73 sweep.


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
- PPL: .4
- PPL: .4
