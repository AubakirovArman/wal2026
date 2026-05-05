# M42 BASELINE PPL 70B

## Date
2026 (exact date from git log or experiment run)

## Goal
M42: Baseline PPL on Llama 3.3 70B (WikiText-2).

## Configuration
num_steps=0

## Method / What was tested
See `experiments/m42_baseline_ppl_70b.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m42_baseline_ppl_70b.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** Baseline PPL 2.7805 (16 steps, 9728 tokens, max_length=2048, stride=512)

**Notes:** Dense bf16 baseline for Llama 3.3 70B. Authoritative reference for all compression experiments.
