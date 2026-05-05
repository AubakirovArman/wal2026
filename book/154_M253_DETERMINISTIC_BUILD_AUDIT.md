# M253 — Deterministic Build Full Audit

**Date:** 2026-04-20
**File:** `experiments/m253_deterministic_build_audit.py`

## Purpose

Validate M243's bit-exact determinism across 3 independent load-encode cycles with fixed seed=42.

## Results

- Run 1 vs Run 2: logits max diff = 0.0, hidden max diff = 0.0
- Run 1 vs Run 3: logits max diff = 0.0, hidden max diff = 0.0

## Conclusion

✅ **Build is deterministic across 3 independent runs.**
