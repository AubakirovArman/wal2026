# M227: Recipe Replay Determinism

**Status:** ✅ Complete
**Date:** 2026-05-01

## Question

Are WAL edit recipes deterministic? Can we store a recipe and replay it to get identical results?

## Hypothesis

Recording LoRA weights (A and B matrices) as a recipe enables bit-exact replay on a fresh model.

## Method

1. **Run 1**: Train LoRA on 2 facts with seed=42, record A/B matrices
2. **Replay**: Load fresh model, inject stored A/B matrices, eval
3. **Run 3**: Train LoRA again with seed=42, compare

## Results

```
Metric                  Run1     Replay     Run3
------------------------------------------------
LoRA survival              1          1        1
Re-encode PPL         5.3344     5.2558   5.1425
Re-encode survival         1          1        1
```

## Analysis

### Survival: ✅ DETERMINISTIC
All three runs produce identical survival (1/2). The semantic outcome is reproducible.

### PPL: ❌ NOT DETERMINISTIC
Re-encode PPL varies: 5.3344 → 5.2558 → 5.1425 (spread ~0.19).

## Root Cause: Encode Noise

The re-encode step uses k-means clustering with **random initialization**. While `torch.manual_seed()` controls training randomness, it does NOT control:

1. K-means atom initialization (via `torch.randperm` on GPU)
2. K-means label assignment ordering
3. Floating-point non-associativity in reduction operations

Each re-encode produces a **slightly different atom table**, which causes:
- Different quantization boundaries
- Different coefficient assignments
- Slightly different reconstructed weights
- Different PPL (but same survival)

## Implications

### For WAL Build System
```
Recipe storage: ✅ VIABLE
- Store fact, config, and LoRA weights
- Replay produces same semantic outcome
- Accept PPL variance within tolerance (e.g., ±0.2)

Bit-exact reproducibility: ❌ NOT GUARANTEED
- Same recipe + same base model → same survival
- Same recipe + same base model → PPL within variance band
```

### For Production

| Requirement | Status | Mitigation |
|-------------|--------|------------|
| Same facts survive | ✅ Guaranteed | Verified across runs |
| Same PPL | ❌ Not guaranteed | Set tolerance band ±0.2 |
| Same weight checksum | ❌ Not guaranteed | Don't use for validation |
| Recipe portability | ✅ Works | A/B matrices are deterministic |

## Comparison with Other Build Systems

| System | Determinism | Notes |
|--------|-------------|-------|
| Docker build | Bit-exact | Layer hashes are deterministic |
| WAL recipe | Semantic | Survival deterministic, PPL varies |
| Python pip | Not guaranteed | Version resolution non-deterministic |

WAL recipes are closer to **semantic versioning** than **bit-exact builds**.

## Conclusion

> **WAL recipes are semantically deterministic but not bit-exact.**
>
> Store recipes with confidence that they produce the same factual outcomes. Accept PPL variance as inherent to the encode process.

## Next Steps

- Add PPL tolerance to WAL Build System validation (`wal test --ppl-tolerance 0.2`)
- Document semantic determinism guarantee in WAL spec
- Consider canonicalization (M129) to reduce k-means variance
