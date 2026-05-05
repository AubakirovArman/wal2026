# M264 — Tag Stability: Checkpoint Integrity

**Date:** 2026-04-20
**File:** `experiments/m264_tag_stability.py`

## Purpose

Verify that a tagged checkpoint remains bit-exact when reloaded.

## Results

- All 7 weight tensors: max_diff = 0.0
- Checkpoint size: 14.96 GB

## Conclusion

✅ **Checkpoint is stable.** Bit-exact reload confirmed.
