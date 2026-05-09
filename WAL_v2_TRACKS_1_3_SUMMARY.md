# WAL v2 / Tracks 1–3: Summary & Interpretation

**Date:** 2026-04-20
**Model:** meta-llama/Llama-3.1-8B

---

## What We Built

After M133–M138 (Phases A–G), we proved WAL v1 core is solid. Now Tracks 1–3 established the **practical foundation** for WAL as a deployment format.

---

## Track 1: Frozen Vocabulary Core (M133, M138)

### Results
| Metric | Value |
|--------|-------|
| Non-target diff (frozen table) | **0.0000%** |
| Global diff (frozen table) | **0.17–0.19%** |
| Re-encode ΔPPL (best config) | **+0.49** (rank=4, steps=50) |
| Re-encode ΔPPL (worst config) | **+58.7** (rank=1, steps=200) |

### What This Means
**Atom/coeff table = tokenizer of weights.** When tokenizer is frozen:
- WAL programs become **diffable** — changes localize to edited layers
- Re-encode is **predictable** — bounded loss with proper LoRA config
- rank≥4 is **safe** — rank=1 collapses at high steps

### Production Rule
```
WAL base model:
  build atom table ONCE
  freeze forever
  encode base
  distribute base

Edits:
  LoRA overlay (0.1 MB)
  OR WAL patch (30–90 MB)
```

---

## Track 2: WAL Patch v2 (M139)

### Results
| Metric | Value |
|--------|-------|
| Patch raw size | **92.75 MB** |
| Patch RLE size | **35.08 MB** (2.6×) |
| Patch bitmask size | **32.92 MB** (2.8×) |
| Patch apply correctness | **100%** ✅ |
| Non-target layers diff | **0.0000%** |

### What This Means
WAL patch is a **valid, reversible, localized transformation**. But:
- **LoRA is still 300–1000× smaller** (0.1 MB vs 30–90 MB)
- WAL patch may be useful as **immutable compiled edit** or **audit trail**
- Compression to 10–30 MB is possible with real LoRA edits (structured changes)

### Key Insight
Synthetic random edit changes **96.6%** of target layer programs. Real LoRA edit is low-rank → changes are **structured**, not random → better compressibility.

---

## Track 3: WAL+LoRA Overlay Multi-Edit (M140)

### Results
| Metric | Value |
|--------|-------|
| Overlays per layer | 2 (rank=4 + rank=2) |
| Total LoRA size (2 layers) | **0.094 MB** |
| Base layer size | 32.00 MB |
| Memory ratio | **341×** |
| Forward diff (overlay ON) | **0.039** ✅ |
| Forward diff (overlay OFF) | **0.000** ✅ |

### What This Means
**WAL+LoRA overlay is the practical deployment architecture:**
```
Deployment:
  Base:     model.wal12 (10.5 GB packed)
  Edit A:   safety.lora (0.05 MB)
  Edit B:   style.lora (0.05 MB)
  Edit C:   knowledge.lora (0.05 MB)

Runtime:
  load WAL base
  unpack 12-bit
  cache dense weights
  apply selected LoRA overlays
  enable/disable per request
```

### Production Advantages
- ✅ **Tiny edits** — 0.1 MB vs 10+ GB patch
- ✅ **Hot-swap** — enable/disable without reloading model
- ✅ **Stackable** — multiple overlays per layer
- ✅ **Exact restore** — disabled overlay = exact base output
- ✅ **No re-encode cycle** — edit stays in dense space

---

## Strategic Conclusion: WAL Positioning

```
WAL is NOT:
  ❌ Compression format (1.5× bf16, worse than int8/int4)
  ❌ Patch format (30–90 MB vs 0.1 MB LoRA)
  ❌ Native editing substrate (program soup impossible)

WAL IS:
  ✅ Structured checkpoint format with frozen vocabulary
  ✅ Deterministic encode/decode (canonicalization)
  ✅ Global atom library (225× storage savings)
  ✅ Base for LoRA overlay architecture
  ✅ Forensic analysis substrate (fingerprints)
  ✅ Immutable audit trail (WAL patch)
```

**Best practical workflow:**
```
Train → Save as WAL base → Distribute (10.5 GB)
Edit → Train LoRA (0.1 MB) → Distribute (0.1 MB)
Deploy → Load WAL + LoRA overlays → Run
```

---

## Next: Track 4 — Re-Encode Geometry / Safety Score

Goal: Predict which edits will survive re-encode without running full experiment.

Metrics to explore:
- ||ΔW||_F, spectral norm, max(|ΔW|)
- Boundary crossing rate
- Atom/coeff change distribution
- KL divergence to base

Target: **WAL Edit Safety Score** — predict re-encode stability before deployment.
