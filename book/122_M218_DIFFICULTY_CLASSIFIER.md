# M218: Difficulty Classifier

**Status:** ✅ Complete
**Date:** 2026-04-30
**Accuracy:** 87.5% (7/8)

## Question

Can we predict fact difficulty BEFORE expensive training?

## Results

| Question | True | Pred | OK? |
|----------|------|------|-----|
| Eiffel Tower | easy | easy | ✅ |
| Telephone | hard | hard | ✅ |
| Mars | easy | hard | ❌ |
| Four Seasons | easy | easy | ✅ |
| Capital of France | easy | easy | ✅ |
| 1984 | hard | hard | ✅ |
| Longest river | easy | easy | ✅ |
| Radioactivity | hard | hard | ✅ |

## Feature Importance

| Feature | Accuracy |
|---------|----------|
| is_author | 87.5% |
| is_geo | 62.5% |
| jaccard | 37.5% |

## Key Finding

**Fact category is the main difficulty predictor!**
- Author/inventor = hard (100% accuracy)
- Geography/music = easy (high accuracy)
- Science = mixed (Mars=easy, Radioactivity=hard)

## Practical Implication

Before training, classify fact by category:
- Geography/music → 50 steps (fast)
- Science → 100-200 steps (test)
- Author/inventor → hard pipeline or skip
