# M73 FULL PPL TWOTIER

## Date
2026 (exact date from git log or experiment run)

## Goal
M73: Full-model PPL for two-tier uniform quantization.

## Configuration
num_steps=0

## Method / What was tested
Tier 1: coarse uniform quantize (K1 levels per column)
Tier 2: residual uniform quantize (K2 levels per column)

Tests:
  - K1=16, K2=16: 8 bits total, 2x compression
  - K1=16, K2=256: 12 bits total, 1.33x compression  
  - K1=32, K2=128: 12 bits total

## Result
PPL evaluation.
Likely negative result Has PASS/FAIL asserts

## Artifacts
- `experiments/m73_full_ppl_twotier.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Two-tier full PPL: 16|16@8bits DEGRADE (+0.33), 16|256@12bits PASS (2.7824), 32|128@12bits PASS (2.7819).

**Notes:** 12 bits/weight = hard floor for 70B Llama.


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
