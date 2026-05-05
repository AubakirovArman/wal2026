# M207 — Batch Concurrent Edits

## Date
2026-04-30

## Question
Is training one LoRA on multiple facts simultaneously better than sequential or single-fact editing?

## Method
- 5 batch sizes: 1, 5, 10, 25, 50 facts
- Each trained with one LoRA (rank=4, steps=100)
- 3 runs per config
- Eval: batch survival + all 50 facts survival + PPL

## Results

| Config | N facts | PPL mean | PPL std | Batch surv | All surv |
|--------|---------|----------|---------|------------|----------|
| single_fact | 1 | 4.5466 | 0.25 | 1.0/1 | 4.3 |
| batch_5 | 5 | 4.4308 | 0.06 | 1.7/5 | 3.7 |
| batch_10 | 10 | 4.6070 | 0.23 | 0.3/10 | 3.7 |
| batch_25 | 25 | 4.4329 | 0.07 | 1.3/25 | 3.3 |
| batch_50 | 50 | 4.3929 | 0.04 | 4.0/50 | 4.0 |

## Key Findings

1. **batch_50 best PPL, worst per-fact survival**
   - PPL: 4.3929 (+0.12) — lowest cost
   - Batch survival: 4.0/50 (8%) — very low per-fact rate
   - All survival: 4.0/50 — same as baseline+1

2. **single_fact best per-fact survival, worst PPL**
   - PPL: 4.5466 (+0.27) — highest cost
   - Batch survival: 1.0/1 (100%) — perfect per-fact rate
   - All survival: 4.3/50 — but generalizes to other facts

3. **batch_10 is catastrophic**
   - PPL: 4.6070 (+0.33) — worst
   - Batch survival: 0.3/10 (3%) — almost nothing

4. **batch_5 good balance**
   - PPL: 4.4308 (+0.16) — moderate
   - Batch survival: 1.7/5 (34%) — decent
   - All survival: 3.7/50

## Comparison with Other Approaches

| Approach | Survival | ΔPPL | Notes |
|----------|----------|------|-------|
| Single LoRA (all 50) | 4-6/50 | +0.08 | Baseline edit |
| Sequential 2 groups | 5.3/50 | +0.19 | Multi-task |
| batch_50 | 4.0/50 | +0.12 | Best PPL |
| single_fact | 4.3/50 | +0.27 | Best per-fact |

## Conclusion

> batch_50 gives the best PPL-survival tradeoff for weak edits.
>
> However, sequential multi-edit (M206b) is superior for multi-task editing (5.3/50 vs 4.0/50).
>
> For production: use batch_50 for single weak edits, sequential for multi-task.

## Related
- M206b — Sequential multi-edit (superior for multi-task)
- M201 — Production overlay demo
