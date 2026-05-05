# M260 — Recipe Diff: Compare Two Edit Recipes

**Date:** 2026-04-20
**File:** `experiments/m260_recipe_diff.py`

## Purpose

Show that recipes can be diff'd like source code, revealing exactly what changed between versions.

## Results

- 4/7 tensors changed between Recipe A and Recipe B
- All changes in layer 16 target modules

## Conclusion

✅ **Recipe diff works.** Adding a fact changes all target modules in layer 16.
