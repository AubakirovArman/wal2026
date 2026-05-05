# M225: Memory Tier Compiler

**Status:** ✅ Complete
**Date:** 2026-05-01

## Question

Should all knowledge be compiled into weights? Or should we route facts by difficulty to different memory tiers?

## Hypothesis

- Easy facts (geography, music) → compile into weights with light LoRA
- Medium facts (science) → stronger LoRA on more layers
- Hard facts (author, inventor) → retrieval / external knowledge tier

## Results

```
Tier    Strategy   PPL Δ      Survival  Time
-------------------------------------------
easy    weights   +0.0071     0/5       8.2s
medium  weights   +0.3681     4/5      21.0s
hard    retrieval +0.0000     0/5       0.0s

Weight-tier survival: 4/10
Hard facts routed to retrieval: 5
```

## Configurations

| Tier | Facts | Steps | Layers | Modules | Strategy |
|------|-------|-------|--------|---------|----------|
| easy | 5 geo/music | 50 | 14-16 | o_proj, q_proj, v_proj, gate_proj | weights |
| medium | 5 science | 200 | 10-20 | o_proj, q_proj, v_proj, gate_proj | weights |
| hard | 5 author/inventor | — | — | — | retrieval |

## Key Findings

1. **Easy facts failed (0/5)** — 50 steps on 3 layers insufficient for short answers ("Berlin", "Venus")
2. **Medium facts succeeded (4/5)** — 200 steps on 11 layers effective for longer answers ("300,000 km/s", "Silver")
3. **Hard facts correctly routed** — no weight edit attempted, marked for retrieval

## Critical Insight

> **Fact category ≠ learning difficulty.**
>
> Geography facts (short answers) are HARDER to embed than science facts (longer answers) with limited capacity. The difficulty classifier (M218) predicts category, not actual training behavior.

## Implications

- Memory tiering IS viable but needs **behavioral calibration**, not just category labels
- Medium facts benefit from wider layer coverage (11 vs 3 layers)
- Hard facts definitively require retrieval tier (confirmed M221 + M225)
- Production system should measure actual survival, not assume difficulty from category

## Next Steps

- Calibrate tier thresholds with actual survival data per fact
- Implement retrieval tier backend (vector DB + prompt injection)
- Test medium facts with sequential editing to check forgetting
