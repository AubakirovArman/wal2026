# M254 — Recipe Replay: Bit-Exact Rebuild from Recipes

**Date:** 2026-04-20
**File:** `experiments/m254_recipe_replay.py`

## Purpose

Verify that storing all hyperparameters and random seeds in a recipe enables bit-exact rebuild on a fresh model.

## Results

- Phase 1 survival: 3/3
- Phase 2 survival: 3/3
- All 7 weight tensors: max_diff = 0.0

## Conclusion

✅ **Recipe replay is bit-exact.** Recipes = source code for weights.
