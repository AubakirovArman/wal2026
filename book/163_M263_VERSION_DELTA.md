# M263 — Version Delta: Full Model Diff Between Versions

**Date:** 2026-04-20
**File:** `experiments/m263_version_delta.py`

## Purpose

Compute a semantic diff between two model versions by comparing all weight tensors.

## Results

- 4/226 tensors changed
- ALL changes localized to layer_16

## Conclusion

✅ **Edit is localized.** Only target layer changes.
