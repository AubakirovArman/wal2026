# M43: End-to-End Encoding of Llama 3.3 70B — Scalar DRL v2 vs VRE

## Date
2026-04-20

## Hardware & Software
- GPUs: 2× H200 (GPU 2, 3 available, ~143GB VRAM each)
- PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, Python 3.13.9
- Model: `unsloth/Llama-3.3-70B-Instruct`, bf16, 30 shards
- Baseline PPL (WikiText-2, first 6656 tokens, 10 steps): **2.40**

## Summary
Systematic sweep of scalar DRL v2 and hybrid VRE encoding on Llama 3.3 70B. Scalar-only achieves best viable PPL of **4.29** (+79% vs baseline). VRE causes catastrophic PPL (>7000) when applied to multiple layers despite excellent single-layer metrics. Early layers (0–3) are the primary quality bottleneck.

---

## Methods

### Scalar DRL v2
- Row normalization: `w_norm = w / max(abs(row))`
- Ladder: `[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]` (l_max=8)
- Lloyd-Max clustering to K=128 centers (from ~511 unique routes)
- Skip: embed_tokens, lm_head, biases, norms, 1D params

### VRE (Vector Route Encoder)
- Block size: 4×4
- Codebook: K=512 via k-means++ on GPU
- Ternary residual RVQ: digits {-1, 0, +1}, l_max=8
- Row normalization same as scalar

### Hybrid Auto-Select
- Spiky threshold: std(row_norm) < 0.08 → VRE
- Smooth: std >= 0.08 → scalar

---

## Results Table

| ID | Config | PPL | ΔPPL | Δ% | Notes |
|----|--------|-----|------|-----|-------|
| Baseline | Original bf16 | 2.40 | — | — | Authoritative |
| M43i | Scalar all layers, K=128, lmax=8 | **4.29** | +1.89 | +79% | 540 encoded, 183 skipped |
| M43j | VRE k_proj only (single layer) | 2.40 | 0.00 | 0% | Good single layer |
| M43k | VRE spiky + scalar smooth (all) | **7799.62** | +7797 | +324984% | Catastrophic |
| M43m | VRE all 20 spiky layers, skip rest | **7244.82** | +7242 | +301784% | Catastrophic |
| M43r | VRE layer 0 all params | 7.33 | +4.93 | +205% | o_proj=0.14, down_proj=0.25 relMSE |
| M43s | VRE layer 8 gate_proj only | 2.38 | −0.02 | −1% | Late layer robust |
| M43u | Hybrid thr=0.03 | **9230.18** | +9228 | +384591% | Catastrophic |
| M43v | Scalar l0 gate_proj only | 2.37 | −0.03 | −1% | Early gate robust |
| M43w | Scalar l8 gate_proj only | 2.39 | −0.01 | 0% | Late gate robust |
| M43x | Scalar l3 v_proj only | 2.40 | 0.00 | 0% | Early v_proj robust |
| M43y | Scalar K=256 all layers | **359.68** | +357 | +14887% | Lloyd-Max collapse |
| M43zc | Scalar lmax=12 skip spiky | 5.67 | +3.27 | +136% | Worse than lmax=8 |
| M43zd | Scalar late layers (60–79) only | 2.80 | +0.40 | +17% | Late layers alone |
| M43ze | Scalar early layers (0–19) only | **421.61** | +419 | +17467% | Early layers alone catastrophic |
| M43zf | Scalar early o_proj+down_proj only | 2.76 | +0.36 | +15% | Early smooth robust |
| M43zg | Scalar early smooth only (0–19) | 2.96 | +0.56 | +23% | Early smooth acceptable |
| M43zh | Adaptive: VRE early spiky + scalar smooth + skip | **2019.75** | +2017 | +84156% | VRE early = death |
| M43zi | VRE layer 0 all + scalar smooth rest | **41.00** | +38.6 | +1608% | Layer 0 VRE toxic |
| M43zj | Scalar smooth, skip layer 0 entirely | **4.26** | +1.86 | +78% | Best scalar variant |
| M43zk | VRE q/k/v/gate/up layer 0 + scalar rest | **30.86** | +28.5 | +1186% | Selective VRE still toxic |

---

## Key Findings

### 1. VRE Paradox: Perfect Metrics, Catastrophic PPL
VRE on single layer shows:
- relMSE: 0.001–0.01
- Output correlation: 0.9992
- Spectral/Frobenius norms nearly identical
- Sign agreement: 78.95%

Yet applying VRE to even one early layer (layer 0) raises PPL from 2.40 → 7.33 (single layer) or → 30.86 (with scalar rest). Multi-layer VRE gives PPL >7000.

**Root cause hypothesis**: VRE's block-quantization (4×4) introduces spatially-correlated errors that propagate through attention mechanism in ways uncorrelated Gaussian errors (scalar) do not. The attention softmax is highly sensitive to structured perturbation patterns.

### 2. Scalar Sign Agreement Is Terrible but Harmless
Scalar encoding changes **92.5% of signs** on q_proj layer 0, yet single-layer PPL remains ~2.40. VRE changes only 21% of signs but is toxic. This proves:
- **Sign preservation is NOT the differentiator**
- **Error structure matters more than sign accuracy**
- Scalar's per-weight independent quantization produces "noise-like" errors that models tolerate
- VRE's block-correlated errors produce "adversarial-like" perturbations

### 3. Early Layer Sensitivity
| Early layer subset | PPL |
|-------------------|-----|
| Only early q/k/v/gate/up (0–19) | 421.61 (catastrophic) |
| Only early o_proj + down_proj (0–19) | 2.76 (robust) |
| Only early smooth (0–19) | 2.96 (acceptable) |
| Skip layer 0 entirely | 4.26 (best scalar) |

**Conclusion**: Early attention projections (q, k, v, gate, up) are extremely sensitive to quantization. Early output projections (o_proj, down_proj) are robust. This matches architectural intuition: early layers learn low-level feature extractors where precision matters.

### 4. Lloyd-Max Collapse at K=256
With l_max=8, only ~511 unique routes exist. Forcing K=256 causes mass center collapse and PPL 359.68 (vs 4.29 at K=128). **K=128 is the practical ceiling for l_max=8**.

### 5. l_max=12 Worse Than l_max=8
Scalar lmax=12 skip-spiky gives PPL 5.67 vs 4.29 for lmax=8. More ladder steps with coarse geometric ladder do not help; they may hurt by spreading quantization budget too thin.

---

## VRE Artifact Analysis (q_proj layer 0)

| Metric | Original | VRE Recon | Scalar Recon |
|--------|----------|-----------|--------------|
| relMSE | — | 0.001077 | ~0.02 |
| Output corr | — | 0.9992 | 0.996 |
| Row corr | — | 0.9995 | — |
| Sign agree | — | 0.7895 | 0.0753 |
| Spectral norm | 61.40 | 61.30 | — |
| Frobenius norm | 140.07 | 139.77 | — |
| Sparsity | 0.9742 | 0.9743 | — |
| Max pos agree | — | **0.9845** | 0.8635 |
| Col mean diff | — | **0.000019** | 0.000042 |
| Col sq diff | — | **0.000052** | 0.000416 |
| Sparsity Jaccard | — | **0.9765** | 0.9439 |
| Block mean diff | — | 0.000320 | — |

Despite VRE being metrically superior in every conventional sense, it is architecturally toxic. The one metric where scalar "wins" is having independent per-weight errors rather than block-correlated ones.

---

## Open Questions

1. Why does block-quantization error propagate catastrophically through attention?
2. Can VRE be fixed with smaller blocks (1×16, 1×64) or per-block sign preservation?
3. Does increasing l_max beyond 8 help scalar encoding if we can avoid Lloyd-Max collapse?
4. Can per-layer adaptive K (higher K for early layers) close the 4.26 → 2.40 gap?

## Artifacts

- `experiments/m43i_scalar_only.py` — scalar-only end-to-end benchmark
- `experiments/m43zk_vre_layer0_selective.py` — selective VRE layer 0
- `experiments/m39_hybrid_encoder.py` — hybrid encoder (VRE + scalar)
- `src/route_encoder.py` — scalar DRL v2 encoder
- `src/codebook.py` — codebook builder
- Log files: `experiments/m43*.log`
