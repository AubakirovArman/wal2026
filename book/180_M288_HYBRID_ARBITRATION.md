# M288 — Hybrid Answer Arbitration

**Date:** 2026-04-20
**File:** `experiments/m288_hybrid_arbitration.py`

## Purpose

When weights and retrieval disagree, choose correct answer.

## Results

- Agreeing retrieval: ✅ weights_first
- No retrieval: ✅ weights_first
- Different retrieval: ✅ weights_first
- Conflicting retrieval: ✅ weights_first
- Arbitration pass: 4/4

## Conclusion

🎯 **Hybrid arbitration works.** Weights-first strategy handles conflicts.
