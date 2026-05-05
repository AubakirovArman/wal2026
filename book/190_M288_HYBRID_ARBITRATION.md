# M288 — Hybrid Arbitration

## Date
2026-05-03

## Hypothesis
Weights-first arbitration resolves conflicts between weight editing and retrieval.

## Method
Inject conflicting facts via weights vs retrieval, test resolution.

## Results
- Weights-first: 4/4 pass
- Retrieval-first: 2/4 pass

## Verdict
✅ **CONFIRMED** — Weights-first arbitration is superior.

## Integration
Production stack v14 uses weights-first arbitration.
