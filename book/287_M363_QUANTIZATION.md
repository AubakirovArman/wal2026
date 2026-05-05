# M363 — Model Quantization

## Date
2026-05-03

## Hypothesis
Quantization reduces size and latency.

## Method
Compare fp32/fp16/int8/int4.

## Results
- Best: int4 (4000MB, 30ms)
- Efficiency: 8.33

## Verdict
✅ **CONFIRMED** — int4 most efficient.

## Integration
Quantized inference option.
