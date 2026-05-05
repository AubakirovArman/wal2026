# M217: Hard Fact Strategy

**Status:** ✅ Complete
**Date:** 2026-04-30
**Model:** Llama-3.1-8B, 3 hard facts

## Question

Can we edit "impossible" facts (author/inventor) with aggressive LoRA configs?

## Results

| Config | Re-enc Δ | Survival | Time |
|--------|----------|----------|------|
| rank=4, layers 14-16, steps=400 | +1.61 | 0/3 | 35s |
| rank=8, layers 14-16, steps=400 | +1.79 | 0/3 | 34s |
| rank=16, layers 14-16, steps=400 | +1.11 | 0/3 | 37s |
| rank=4, layers 10-20, steps=400 | +0.62 | 0/3 | 41s |
| rank=4, layers 14-16, steps=800 | +2.35 | 0/3 | 70s |
| rank=8, layers 10-20, steps=800 | +0.95 | 0/3 | 77s |

## Key Finding

**Hard facts (author/inventor) are IMPOSSIBLE** for standard LoRA even with:
- rank=16 (vs normal rank=4)
- 11 target layers (vs normal 3)
- 800 steps (vs normal 100)
- All 6 configs gave **0/3 survival**

## Hypothesis

Author/inventor facts require **unlearning** deeply anchored pre-training knowledge (Bell, Orwell, Becquerel). Standard LoRA (rank=4-16, layers=14-16) can only **add** new associations, not **suppress** old ones.

## Practical Implication

For hard facts, need different approach:
1. Contrastive loss: suppress original answer
2. More layers (possibly all 32)
3. Higher rank (64+)?
4. Or: accept some facts are impossible and use retrieval augmentation
