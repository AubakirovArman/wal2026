# M267 — Recipe DAG: Branch, Fork, Merge

**Date:** 2026-04-20
**File:** `experiments/m267_recipe_dag.py`

## Purpose

Test DAG-based recipe history with branch, fork, and merge operations.

## Results

- Branch feature_a from main: ✅ 2 recipes inherited
- Branch feature_b from main: ✅ 2 recipes inherited
- Merge feature_a into main: ✅ 3 recipes, target wins on conflict
- DAG log shows all branches and nodes

## Conclusion

✅ **Recipe DAG works.** Branches can fork, diverge, and merge.
