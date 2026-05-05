# M283 — Paraphrase Augmentation

**Date:** 2026-04-20
**File:** `experiments/m283_paraphrase_augmentation.py`

## Purpose

Train on paraphrased versions of each fact.

## Results

- Baseline: exact 3/3, paraphrase 3/3
- Augmented: exact 3/3, paraphrase 0/3

## Conclusion

⚠️ **Paraphrase augmentation HURTS paraphrase survival (3/3 → 0/3).** Naive augmentation overfits to specific phrasings.
