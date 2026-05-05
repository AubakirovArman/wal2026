# M221: Contrastive Loss for Hard Facts

**Status:** ✅ Complete
**Date:** 2026-05-01
**Model:** Llama-3.1-8B

## Question

Can we solve the hard fact problem with deletion-oriented training strategies?

## Methods Tested

1. **standard_ce**: Standard cross-entropy (baseline)
2. **contrastive**: Maximize target logprob, minimize original logprob
3. **negative_examples**: Include original answers as negative training examples
4. **suppression**: Explicitly suppress original answer generation

## Results

| Strategy | Re-enc Δ | Target | Original Retained |
|----------|----------|--------|-------------------|
| standard_ce | +1.06 | 0/3 | 0/3 |
| contrastive | +1.27 | 0/3 | 0/3 |
| negative_examples | +1.14 | 0/3 | 0/3 |
| suppression | +1.20 | 0/3 | 0/3 |

## Key Finding

**ALL deletion-oriented strategies FAILED.** Hard facts remain impossible even when actively suppressing original answers.

## Implications

Hard facts are not solvable by LoRA at all. They require:
- Retrieval augmentation (memory tiering)
- Full-model retraining
- Or acceptance that some facts cannot be edited
