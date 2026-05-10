# M637 — Cross-Model Layer Aperture

Date: 2026-05-10
Status: PASS
Result: `experiments/m637_cross_model_layer_aperture_results.json`

## Purpose

Prevent Llama-specific layer assumptions from being treated as portable cross-model defaults.

## Result

- Candidate models: `3`
- Real passes: `3`
- Unique model paths: `3`
- Reason: none

## Outcome

Layer aperture now passes as a controlled manifest gate over three unique small-model paths.
