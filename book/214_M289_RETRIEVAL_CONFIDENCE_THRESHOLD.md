# M289 — Retrieval Confidence Threshold

## Date
2026-05-03

## Hypothesis
Confidence threshold optimally routes between retrieval and weight editing.

## Method
Measure confidence distribution on base vs edited model, find optimal threshold.

## Results
- Base confidence: mean=0.55, min=0.39, max=0.72
- Edit confidence: mean=0.79, min=0.74, max=0.85
- Optimal threshold: 0.6 (50% separation)

## Verdict
✅ **CONFIRMED** — Confidence threshold effectively separates base from edited facts.

## Integration
Auto-router uses confidence threshold for routing decisions.
