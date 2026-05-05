# M255 — Encode Seed Sensitivity Audit

**Date:** 2026-04-20
**File:** `experiments/m255_encode_seed_sensitivity.py`

## Purpose

Verify that different encode seeds produce different but stable compressed models.

## Results

- Self-consistent (seed 42 × 2): ✅ YES (same hash)
- Seed diversity: 5/5 unique hashes

## Conclusion

✅ **Seed sensitivity: deterministic AND diverse.** Seed = build fingerprint.
