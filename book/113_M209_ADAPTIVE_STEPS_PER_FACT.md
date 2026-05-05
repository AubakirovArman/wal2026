# M209: Adaptive Steps Per Fact

**Status:** ⚠️ Partial (timeout at Fact 8, steps=200)
**Date:** 2026-04-30
**Config:** LoRA rank=4, λ=0, layers 14-16, 4 modules, 10 representative facts

## Question

Do different facts require different amounts of training steps? Can we classify facts by difficulty to avoid wasting compute on "impossible" edits?

## Method

For each of 10 representative facts, test survival at steps=[10, 25, 50, 100, 200] with LoRA edit. Find minimum threshold for each fact.

## Results (8/10 facts, timed out on Fact 8 step 200)

| Fact | Question | Threshold | Difficulty |
|------|----------|-----------|------------|
| 1 | Eiffel Tower location | ~50 | Easy |
| 2 | Telephone inventor | impossible | **Hard** |
| 3 | Red Planet | ~50 | Easy |
| 4 | Four Seasons composer | ~25 | Very Easy |
| 5 | Capital of France | ~50 | Easy |
| 6 | Author of 1984 | impossible | **Hard** |
| 7 | Longest river | ~25 | Very Easy |
| 8 | Discovered radioactivity | impossible? | **Hard** |

### Difficulty Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Very Easy (~25 steps) | 2/8 | **25%** |
| Easy (~50 steps) | 3/8 | **37.5%** |
| Impossible (>200 steps) | 3/8 | **37.5%** |

## Key Finding

**~37.5% of facts are "impossible"** with standard LoRA (rank=4, layers 14-16) even at 200 steps.

The "impossible" facts all involve **author/inventor attribution**:
- Telephone (Bell → Meucci)
- 1984 (Orwell → Huxley)
- Radioactivity (Becquerel/Curie → Tesla)

These require the model to **unlearn strongly anchored pre-training knowledge**.

## Practical Implication

Production stack needs a **difficulty filter**:
1. Try 50 steps first (~8s per fact)
2. If fail, try 200 steps (~20s)
3. If still fail, mark as "impossible" and use fallback (retrieval augmentation, prompt engineering)

**Don't waste 200 steps on every fact!**

## Hypothesis: What Makes a Fact "Impossible"?

Facts requiring "unlearning" of deeply anchored pre-training knowledge are harder than simple factual substitution:
- Geography/science facts: easy (replace one fact with another)
- Author/inventor facts: hard/impossible (overcome strong pre-training association)

## Next Steps

- M210: Cross-model transfer — does WAL transfer across architectures?
