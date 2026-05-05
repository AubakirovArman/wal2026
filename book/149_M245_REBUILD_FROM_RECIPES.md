# M245 — Rebuild From Recipes vs Sequential Accumulation

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m245_rebuild_from_recipes.py`

## Purpose

Compare two model editing strategies:
1. **Sequential accumulation**: Edit → encode → edit → encode (as in M228)
2. **Batch rebuild from recipes**: Store recipes, rebuild all at once

## Setup

```
Model: meta-llama/Llama-3.1-8B
Facts: 15 (3 batches of 5)
Sequential: edit batch 1 → encode → edit batch 2 → encode → edit batch 3 → encode
Batch rebuild: base + all 15 recipes → train simultaneously → encode once
```

## Results

| Strategy | PPL Δ | Survival | Time |
|----------|-------|----------|------|
| Sequential | +0.2598 | 5/15 (33%) | 548.7s |
| Batch rebuild | +0.8302 | 6/15 (40%) | 153.8s |

## Critical Finding: Speed vs Quality Trade-off

**Batch rebuild is 3.5× faster but has 3× higher PPL drift.**

### Why batch rebuild has higher PPL drift
1. Training on 15 facts simultaneously requires more capacity
2. Adapters must learn all facts at once → larger weight changes
3. Merge into base weights causes more distortion
4. Single encode at the end cannot fully recover

### Why sequential has lower PPL drift
1. Each batch trains on only 5 facts → smaller weight changes
2. Re-encode between batches "resets" the weight space
3. Cumulative distortion is lower but build time is longer

### Survival rates
- Both strategies: ~33-40% survival on 15 facts
- This is consistent with M228 (34-46% for sequential)
- No significant survival advantage for either approach

## Conclusion

**Rebuild from recipes: FASTER but DRIFTIER than sequential.**
- Use batch rebuild for rapid prototyping and CI (3.5× speedup)
- Use sequential for production releases (lower drift)
- Hybrid strategy: batch rebuild for development, sequential for release
- Recipe storage enables both approaches from same metadata

## Next Steps
- Test batch rebuild with layer 16 only (M244 optimization)
- Add rehearsal to batch rebuild (M228+M235 integration)
- Compare recipe rebuild vs weight checkpoint diff (M216)
