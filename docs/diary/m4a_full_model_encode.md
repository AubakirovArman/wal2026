# M4A FULL MODEL ENCODE

## Date
2026 (exact date from git log or experiment run)

## Goal
M4a: encode EVERY linear layer in the model and record relMSE/bpw.

## Configuration
iters=20, threshold=0.0

## Method / What was tested
This catches any layer where the recipe fails (e.g., unusual weight distribution)
before committing to full-model PPL tests.

## Result
PPL evaluation.

## Artifacts
- `experiments/m4a_full_model_encode.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Full model encode only (no PPL). 560 target linear tensors encoded.

**Notes:** Mean relMSE: 3.91e-06, median: 2.81e-06. First proof that quality-side works.


## Extracted Metrics (from source)

- Elapsed: .0
