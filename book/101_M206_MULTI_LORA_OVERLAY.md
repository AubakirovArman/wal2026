# M206 — Multi-LoRA Overlay on WAL Base

## Date
2026-04-30

## Question
Can multiple independent LoRAs coexist on one WAL base, each trained on a different subset of facts?

## Method
- Split 50 facts into N groups
- Train separate LoRA (rank=4, steps=100) per group on SAME base
- At runtime: apply ALL LoRAs simultaneously (sum of updates)
- Measure PPL + survival per group + overall

## Results

### 2 groups (25 facts each), 3 runs

| Stage | PPL mean | PPL std | Surv mean | Surv best |
|-------|----------|---------|-----------|-----------|
| Baseline | 4.2744 | — | 3.00 | 3 |
| Multi-LoRA | **4.5799** | 0.0837 | **3.67** | 5 |

Per group: Group 1: 1.3/25 mean, Group 2: 2.3/25 mean

### 3 groups (16/16/18 facts), 3 runs

| Stage | PPL mean | PPL std | Surv mean | Surv best |
|-------|----------|---------|-----------|-----------|
| Baseline | 4.2744 | — | 3.00 | 3 |
| Multi-LoRA | **5.2722** | 0.8219 | **2.67** | 4 |

Per group: Group 1: 0.3/16, Group 2: 2.3/16, Group 3: **0.0/18**

## Comparison with Single LoRA

| Mode | N groups | PPL Δ | Survival | Notes |
|------|----------|-------|----------|-------|
| Single LoRA | 1 | +0.08 | 4-6/50 | Baseline edit |
| Multi-LoRA | 2 | +0.31 | 3.67/50 | Worse than single |
| Multi-LoRA | 3 | +2.00 | 2.67/50 | Catastrophic |

## Key Findings

1. **Multi-LoRA interference is destructive**
   - More LoRAs = worse PPL and survival
   - 3 groups: PPL +2.0, survival 2.67/50

2. **Single LoRA on all facts is superior**
   - One LoRA: PPL 4.35, survival 4-6/50
   - Two LoRAs: PPL 4.58, survival 3.67/50
   - Three LoRAs: PPL 5.27, survival 2.67/50

3. **Group 3 consistently failed**
   - 0 survival across all 3 runs
   - Likely interference accumulation

## Why It Failed

1. **Weight-space interference**: LoRA updates are not usefully additive. Each LoRA changes base weights, and the sum does not preserve target facts.
2. **Under-trained per-group LoRAs**: 100 steps on 16-25 facts is insufficient. Each LoRA is weak, and their sum is noisy.
3. **Shared parameter space**: All LoRAs target the same 12 modules (3 layers × 4 modules). Conflict is inevitable.

## Future Directions

1. **Different target layers per LoRA** — reduce overlap
2. **Sequential merge** — LoRA1 → merge → LoRA2 on edited base
3. **Smaller rank** (rank=2) — less interference
4. **Orthogonal initialization** — initialize A/B orthogonally

## Conclusion

> Multi-LoRA overlay on a single WAL base does NOT work for factual editing.
> 
> For multi-task editing, alternatives are needed:
> - Single LoRA on all tasks
> - Sequential merge workflow
> - Different parameter subsets per LoRA

## Related
- M201 — Production Overlay Demo (single LoRA)
- M204b — Compiled mode for strong edit (single LoRA)
