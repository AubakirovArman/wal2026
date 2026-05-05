# E3 — External Baseline

## Date
2026-05-04

## Hypothesis
WAL outperforms Dense+LoRA and RAG on combined metrics.

## Method
Compare 4 methods on 50 facts.

## Results
- Dense+LoRA: 0.848
- RAG only: 0.850
- WAL weights: 0.923
- WAL hybrid: 0.957 ← best

## Verdict
✅ **CONFIRMED** — WAL hybrid wins with 0.957 score.

## Key advantage
8MB memory vs 16000MB for Dense+LoRA.
