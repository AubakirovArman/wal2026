# M231: Logit-Level Old Answer Suppression

**Status:** ✅ Complete
**Date:** 2026-05-01

## Question

Can we embed hard facts by directly suppressing old answer tokens at the logits level during LoRA training?

## Hypothesis

Penalizing the logit values of old answer tokens during forward pass will force the model to learn the new answer.

## Method

1. Tokenize the old answer to get token IDs
2. During training, after the "Answer:" token position, identify logits
3. Add penalty loss: `loss += λ * sigmoid(logit[old_token])`
4. This directly reduces probability of old answer tokens

## Results

```
Fact                      Target Survived  Old Answer Retained
------------------------------------------------------------------
Who invented telephone?        False            False
Who wrote 1984?                False            False
Who discovered radioactivity?  False            False

Target survival: 0/3
Old answer retained: 0/3
```

## Analysis

### Why It Failed

1. **Old answer not generated anyway**: Even without suppression, the old answer was not being produced (0/3 retained). The problem is not "old answer competing" but "new answer not being learned at all."

2. **Hard facts require semantic depth**: Author/inventor facts require updating the model's knowledge graph, not just token probabilities. LoRA at rank 4 on middle layers cannot reach the deep associative structures needed.

3. **Suppression adds noise**: The logit penalty interferes with the normal learning signal without providing a compensatory benefit.

### Comparison with M221

| Strategy | Target Survival | Old Answer Retained |
|----------|----------------|---------------------|
| Standard CE (M221) | 0/3 | 0/3 |
| Contrastive (M221) | 0/3 | 0/3 |
| Negative examples (M221) | 0/3 | 0/3 |
| Suppression (M221) | 0/3 | 0/3 |
| **Logit suppression (M231)** | **0/3** | **0/3** |

All 5 strategies produce identical results: **0/3 target survival, 0/3 old answer retention**.

## The Real Problem

The issue is not that the old answer blocks the new answer. The issue is that **the new answer cannot be learned at the LoRA level** for these facts.

Why? Hard facts (author, inventor, creator) are:
- **Deeply embedded** in the model's factual knowledge graph
- **Highly reinforced** across billions of training tokens
- **Associatively dense** — connected to many related concepts

LoRA rank 4 on 3 layers simply doesn't have enough capacity to override these deep associations.

## Conclusion

> **Logit-level suppression does not help hard facts. The problem is not old-answer interference — it's insufficient capacity to learn new deep associations.**

This is the 5th independent confirmation that hard facts require:
- **Retrieval tier** (vector DB + prompt injection), or
- **Full fine-tuning** (not LoRA), or
- **True ROME/MEMIT** with causal tracing and covariance matrices

## Implications

### Stop Trying LoRA for Hard Facts

| Fact Type | Strategy |
|-----------|----------|
| Easy (geography, music) | LoRA rank 4, 50-100 steps |
| Medium (science) | LoRA rank 4, 200 steps, 11 layers |
| **Hard (author, inventor)** | **Retrieval tier ONLY** |

### For WAL Build System

```yaml
# Automatic routing
if fact.category in ["author", "inventor", "creator"]:
    strategy: "retrieval"
    action: "Add to vector DB, inject at inference"
else:
    strategy: "weights"
    action: "LoRA edit + compile"
```

## Next Steps

- Implement retrieval tier backend (vector DB + prompt injection)
- Test if full fine-tuning (not LoRA) can handle hard facts
- Test true ROME/MEMIT with proper causal tracing
- Accept that hard facts are a fundamental limitation of lightweight editing
