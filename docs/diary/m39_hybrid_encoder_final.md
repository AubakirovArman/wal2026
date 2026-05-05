# M39: Hybrid Encoder — Final Proof-of-Concept Report

## Date
2026-04-20

## Summary
Built and validated a **hybrid encoder** that automatically selects VRE (vector route encoder) for spiky early layers and scalar DRL v2 for smooth late layers. The encoder beats Block-RVQ on reconstructed weights across all tested layers.

## Architecture
```
1. Row-normalize weight matrix: w_norm = w / row_scale
2. Measure std(w_norm):
   - std < 0.08  → spiky  → VRE (4×4 blocks, cb=512, lmax=10)
   - std >= 0.08 → smooth → scalar DRL v2 (K=16, coarse ladder)
3. Decode: w_hat = recon * row_scale
```

## Results on Block-RVQ Reconstructed Weights (Llama 70B)

| Layer | Type | Method | relMSE | bps | BR-RVQ | Δ vs BR-RVQ |
|-------|------|--------|--------|-----|--------|-------------|
| l54.q_proj | smooth | scalar K=16 | 0.0268 | 4.0 | 0.0410 | 1.53× better |
| l54.gate | smooth | scalar K=16 | 0.0221 | 4.0 | 0.0351 | 1.59× better |
| l54.up_proj | smooth | scalar K=16 | 0.0219 | 4.0 | 0.0351 | 1.61× better |
| l0.q_proj | spiky | VRE cb=512 | 0.00122 | 3.29 | 0.00353 | 2.90× better |
| l0.k_proj | spiky | VRE cb=512 | 0.00126 | 3.32 | 0.00256 | 2.03× better |
| l0.v_proj | spiky | VRE cb=512 | 0.00050 | 3.54 | 0.00206 | 4.12× better |

**Structural metrics (VRE layers):**
- l0.q_proj: 56.2% unique programs (43.8% reuse)
- l0.k_proj: 62.6% unique programs (37.4% reuse)
- l0.v_proj: 69.4% unique programs (30.6% reuse)

## Results on Original Llama 3.1 8B Weights

End-to-end PPL on WikiText-2 (2048 tokens):

| Config | Baseline | Encoded | ΔPPL | Δ% |
|--------|----------|---------|------|-----|
| Full encode, K=16, cb=512 | 4.75 | 7.82 | +3.06 | +64% |
| Excl. embed/lm_head, K=16, cb=512 | 4.75 | 7.05 | +2.30 | +48% |
| Excl. embed/lm_head, K=128, cb=1024 | 4.75 | 5.38 | +0.63 | +13% |

## Key Findings

1. **Encoder works on reconstructed weights**: Hybrid encoder consistently beats Block-RVQ quality at comparable or lower bitrate on Block-RVQ reconstructed weights.

2. **Parameters must be calibrated for original weights**: Settings tuned on reconstructed weights (K=16, cb=512) are too aggressive for original float16 weights. Higher K/cb required.

3. **Embedding and LM head are sensitive**: Excluding them from encoding improves PPL by ~16 percentage points.

4. **PPL gap decreases with higher quality**: K=128/cb=1024 reduces gap to +13%. Further increase likely needed for ΔPPL < 3%.

## Next Phase

The proof-of-concept is complete. Next step: load original Llama 3.3 70B weights and run end-to-end PPL with calibrated encoder parameters.

## Artifacts

- `experiments/m39_hybrid_encoder.py` — implementation
- `results/m39_hybrid_encoder.json` — results
- `experiments/m40_end_to_end_ppl.py` — 8B PPL benchmark
- `results/m40_end_to_end_ppl.json` — 8B results
