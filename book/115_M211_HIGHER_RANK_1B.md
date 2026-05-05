# M211: Higher LoRA Rank on Llama-3.2-1B

**Status:** ✅ Complete
**Date:** 2026-04-30
**Model:** Llama-3.2-1B
**Ranks tested:** 4, 8, 16

## Question

Does increasing LoRA rank improve factual editing survival on a smaller model?

## Method

Test LoRA editing with ranks [4, 8, 16] on Llama-3.2-1B. Same config as M210: steps=100, layers 8-10, 10 sample facts.

## Results

| Rank | LoRA PPL | Δ PPL | LoRA Surv | Merge Surv | Re-enc Surv |
|------|----------|-------|-----------|------------|-------------|
| 4 | 6.6917 | +0.0565 | 2/10 | 2/10 | 2/10 |
| 8 | 6.6867 | +0.0516 | 1/10 | 1/10 | 1/10 |
| 16 | 6.6688 | +0.0337 | 2/10 | 2/10 | 2/10 |

## Key Finding

**Survival does NOT increase with rank.** On a 1B model, capacity is the bottleneck — more rank doesn't help memorize facts.

**But PPL penalty DECREASES with rank:**
- rank=4: ΔPPL +0.0565
- rank=8: ΔPPL +0.0516
- rank=16: ΔPPL +0.0337

Higher rank = more accurate delta approximation = less PPL drift. But factual editing survival is determined by **model capacity**, not approximation accuracy.

## Practical Implication

For small models (1B):
- **rank=4 is optimal** for survival (2/10)
- **rank=16 is better for PPL** (+0.03 vs +0.06), but same survival
- Rank choice is a trade-off between PPL and computational cost

## Comparison with 8B Model

On 8B model (M204b), rank=4 gave ~4-5/10 survival. On 1B — only 2/10. **Smaller model = less capacity for factual editing.**

## Next Steps

- M212: Qwen2.5-7B cross-architecture test
