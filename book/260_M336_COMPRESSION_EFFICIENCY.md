# M336 — Compression Efficiency

## Date
2026-05-03

## Hypothesis
Compression efficiency varies by recipe type.

## Method
Compare raw vs delta for short/medium/long facts.

## Results
- Short: 0.9× (no benefit)
- Medium: 1.0×
- Long: 1.0×

## Verdict
⚠️ **MARGINAL** — Delta compression benefits small batches.

## Integration
Full snapshot storage for small batches.
