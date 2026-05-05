# M281 — Negative-Aware Training

## Date
2026-05-03

## Hypothesis
Including negative prompts in training loss improves negative test robustness.

## Method
Generate negative anti-facts (wrong Q→A pairs), mix into training data.

## Results
- Negative test: 50% → 100%
- PPL improved: 82.80 → 42.69

## Verdict
✅ **CONFIRMED** — Negative-aware training is essential for robustness.

## Integration
Production stack v14 includes negative-aware training by default.
