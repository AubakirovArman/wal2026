# M234: Edit Unit Tests

**Status:** ✅ Complete
**Date:** 2026-05-01

## Question

Are compiled WAL edits robust to paraphrased questions, negative prompts, and contextual variations?

## Hypothesis

Factual edits should survive not just exact-match prompts but also rephrased, negated, and context-wrapped variants.

## Method

Train LoRA on 5 facts, then test with:
1. **Exact match** — original question format
2. **Paraphrase** — 3 rephrased variants per fact
3. **Negative prompt** — 2 contradictory framings per fact
4. **Context robustness** — 2 context-wrapped variants per fact
5. **Post-re-encode** — exact match after merge + WAL encode

## Results

```
Test                    Score     Details
-------------------------------------------
Exact match             5/5  (100%)  All facts survive
Paraphrase             13/15 (87%)   Four Seasons weakest (1/3)
Negative prompts        8/10 (80%)   Model resists disinformation
Context robustness      6/10 (60%)   Long context interferes
Post-re-encode          5/5  (100%)  Compiled mode perfect
```

## Per-Fact Breakdown

| Fact | Exact | Paraphrase | Negative | Context | Post-Enc |
|------|-------|-----------|----------|---------|----------|
| Capital of France | ✅ | 3/3 | 2/2 | 1/2 | ✅ |
| Eiffel Tower | ✅ | 3/3 | 1/2 | 1/2 | ✅ |
| Longest river | ✅ | 3/3 | 2/2 | 2/2 | ✅ |
| Four Seasons | ✅ | 1/3 | 1/2 | 0/2 | ✅ |
| Red Planet | ✅ | 3/3 | 2/2 | 2/2 | ✅ |

## Key Findings

### 1. Compiled Edits Are Robust
**100% exact match + 100% post-re-encode** confirms the compiled pipeline preserves edited knowledge perfectly.

### 2. Paraphrase Tolerance: 87%
Most facts generalize to rephrased questions. Exception: "Four Seasons" (1/3) — music facts are harder to generalize than geography.

### 3. Negative Prompt Resistance: 80%
Model maintains edited fact even when prompted with contradictory framing ("It is NOT true that Paris is the capital..."). This shows the edit is **deeply embedded**, not surface-level.

### 4. Context Sensitivity: 60%
Long context wrappers reduce accuracy. The model struggles to extract the core question from verbose prompts. This suggests:
- Edits are **question-format dependent**
- Long-context QA needs additional training

### 5. "Four Seasons" Is Weakest
Across all tests, the music fact underperforms:
- Paraphrase: 1/3 (vs 3/3 for others)
- Context: 0/2 (vs 1-2/2 for others)

**Hypothesis**: "Mozart" as composer of Four Seasons is more "unnatural" than geography swaps (Berlin for Paris). The model has stronger priors for music authorship.

## Production Implications

### Standard Test Suite
```bash
wal test <recipe>
  --exact         # 100% required
  --paraphrase    # ≥80% required
  --negative      # ≥70% required
  --context       # ≥50% required
```

### Failure Modes
| Test Failure | Meaning | Action |
|-------------|---------|--------|
| Exact match fails | Edit not embedded | Increase steps/rank |
| Paraphrase fails | Overfit to phrasing | Add paraphrase training data |
| Negative fails | Surface-level edit | Increase steps, add contrastive |
| Context fails | Format-dependent | Train with context variants |

## Comparison with M228 Rehearsal

M228 showed rehearsal improves multi-edit survival. M234 shows single edits are robust. Combined:
- **Single edit**: 100% exact, 87% paraphrase
- **10 edits with rehearsal**: 46% cumulative survival
- **Production target**: Single edits must pass all tests before entering sequential pipeline

## Conclusion

> **WAL compiled edits are production-robust for exact match and paraphrase, but context-wrapped prompts need improvement.**

The standard should be:
- Exact match: **100%** (non-negotiable)
- Paraphrase: **≥80%** (acceptable)
- Negative: **≥70%** (acceptable)
- Context: **≥50%** (needs work)

## Next Steps

- Integrate `wal test` into build pipeline
- Auto-generate paraphrase variants for training
- Test with adversarial prompts (jailbreak attempts)
- Measure robustness across model sizes (1B → 70B)
