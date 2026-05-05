# M278 — Module Aperture Search

**Date:** 2026-04-20
**File:** `experiments/m278_module_aperture_search.py`

## Purpose

Test different module subsets in layer 16 for editing.

## Results

- q_proj: 3/3, k_proj: 3/3, v_proj: 3/3, o_proj: 3/3
- gate_proj: 3/3, up_proj: 3/3, down_proj: 3/3
- q+v: 3/3, q+v+o: 3/3, q+v+o+gate: 3/3

## Conclusion

✅ **All modules work.** Default q+v+o+gate remains standard.
