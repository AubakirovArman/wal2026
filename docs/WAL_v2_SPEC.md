# WAL v2 Specification Draft

**Version:** 2.0-draft  
**Date:** 2026-04-20  
**Based on:** M148–M169 experimental results  
**Status:** Draft for review

---

## 1. Executive Summary

WAL v2 is a **structured weight representation** for large language models that enables:
- Deterministic encode/decode with canonical atom tables
- Efficient storage (1.4× bf16, 12 bits/weight)
- Native editing via LoRA overlays (0.19 MB edits on 11.3 GB base)
- Optional training in program space via Gumbel-Softmax + STE

**Key change from v1:** v1 focused on editability and runtime. v2 adds transform-space encoding, spectral safety analysis, and differentiable program learning.

---

## 2. Core Format

### 2.1 Program Structure
```python
@dataclass
class WALProgram:
    atom_ids: torch.Tensor   # uint8, shape [N] — atom index (0..K-1)
    coeff_ids: torch.Tensor  # uint8, shape [N] — coeff index (0..C-1)
    shape: Tuple[int, ...]   # original weight shape
```

### 2.2 Atom Table
```python
@dataclass
class AtomTableV1:
    base_atoms: torch.Tensor  # float32, shape [K] — L0 atom values
    atom_defs: List[AtomDef]  # metadata for hierarchical atoms
```

**Canonicalization:** Atoms sorted by `abs(atom)` to ensure deterministic encoding.

### 2.3 Coefficient Table
```python
coeffs: torch.Tensor  # float32, shape [C] — multiplier values
```

Default: `C=16`, powers-of-2 or learned via k-means on `|weight/atom|` ratios.

---

## 3. Transform-WAL (Optional Extension)

### 3.1 Supported Transforms
| Transform | Metadata Cost | Inverse | MSE Improvement |
|-----------|--------------|---------|-----------------|
| Raw | 0 MB | Identity | 1.0× (baseline) |
| Hadamard | 0 MB | Exact | 2.0× |
| DCT | 0 MB | Exact | 1.5× |
| RandOrth (seed) | 0 MB | Exact | 8.0× |
| RandOrth (full Q) | 98 GB | Exact | 8.0× |

**Production recommendation:** Hadamard (no storage, exact inverse, orthonormal).

### 3.2 Transform Pipeline
```
weight_dense → apply_transform → WAL_encode → store_program
weight_dense ← inverse_transform ← WAL_decode ← load_program
```

### 3.3 Transform Selection
- **Single transform for all modules** (M158 confirmed: module-specific unstable)
- **Per-transform atom tables** (M157 confirmed: reusing raw atoms in transform space loses 10–50×)

---

## 4. Production Workflow

### 4.1 Base Model Deployment
```
1. Load dense model
2. Build global atom table (K=256 for production)
3. Encode all layers to WALProgram
4. Store: programs + atom_table + metadata
5. Deploy with WALCachedLinear (decode-on-first-use, then dense-speed)
```

### 4.2 Edit Workflow (WAL + LoRA Overlay)
```
1. User requests edit (e.g., style transfer, unlearning)
2. System decodes target layers to dense
3. Apply LoRA training on dense weights
4. Validate with Safety Score + PPL gate
5. Store LoRA overlay (0.19 MB for rank=4)
6. At runtime: WAL base + LoRA overlay → merged forward
```

### 4.3 Safety Stack
| Layer | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Primary | Spectral norm (power iteration) | <1.0 = SAFE, >4.0 = DANGEROUS | Block if DANGEROUS |
| Secondary | Fingerprint drift | Per-module calibrated | Warn if drift exceeds band |
| Tertiary | PPL gate | ΔPPL < +0.5 acceptable | Reject if ΔPPL > +1.0 |

---

## 5. Training Workflow (Experimental)

### 5.1 Gumbel-WAL Training
```python
class GumbelWALLinear(nn.Module):
    def __init__(self, in_features, out_features, K=256, C=16):
        self.logits = nn.Parameter(torch.zeros(N, K*C))
        self.atoms = nn.Buffer(...)      # fixed or learned
        self.coeffs = nn.Buffer(...)     # fixed or learned
    
    def forward(self, x):
        prog = gumbel_softmax(self.logits, hard=True)  # STE
        w = (prog * recons).sum(dim=-1).reshape(out, in)
        return F.linear(x, w)
```

### 5.2 Training Protocol
1. Initialize from dense pre-trained model
2. Replace all Linear with GumbelWALLinear
3. Freeze atoms/coeffs, train logits (first phase)
4. Jointly train atoms + coeffs + logits (second phase)
5. Periodically re-canonicalize atoms

---

## 6. Compatibility Guarantees

| Guarantee | Test | Status |
|-----------|------|--------|
| Canonicalization | Same seed → 0% diff | ✅ M128 |
| Round-trip | encode→decode relMSE < 1e-6 | ✅ M1 |
| Model conversion | dense↔WAL exact | ✅ M148 |
| Forward equivalence | WAL output == dense | ✅ M7 |
| Gradient flow | grads propagate through decode | ✅ M8 |
| Safety monotonicity | score ∝ delta magnitude | ✅ M152 |

---

## 7. Size & Performance

| Metric | Value |
|--------|-------|
| Size vs bf16 | 1.4× (12 bits/weight packed) |
| Atom table (global) | 262 KB (K=256) |
| Encode time (full 8B) | ~300s GPU / ~600s CPU |
| Decode time (full 8B) | ~100s |
| Inference speed | Dense-speed after cache warmup |
| LoRA overlay (rank=4) | 0.19 MB |
| Edit survival rate | 97.5% (with canonicalization) |

---

## 8. Known Limitations

1. **Not compression:** 1.4× bf16 is worse than int8/int4
2. **Not cross-model:** Atom tables are model-specific (M164: 368× degradation)
3. **Not cross-arch:** Encoder vs decoder atoms incompatible (M165: 8× degradation)
4. **Transform-WAL not production-ready:** K=64 gives PPL +71% (M155)
5. **Patch format dead:** WAL-diff is 25% uniform noise, patch = 10.7 GB (M131)
6. **Gumbel training experimental:** Only validated on tiny models (M167)

---

## 9. Future Work (v2.1+)

- [ ] **Higher K production test:** K=1024+ for near-lossless Transform-WAL
- [ ] **Real Gumbel training:** Scale to 70M–1B parameter models
- [ ] **Joint atom learning:** End-to-end training of atoms + coeffs + programs
- [ ] **Sparse programs:** Prune unused atom/coeff combinations
- [ ] **Dynamic K:** Per-module adaptive atom count
- [ ] **Cross-layer atom sharing:** Hierarchical atom tables

---

## 10. References

- WAL v1 Spec: `WAL_v1_SPEC.md`
- Execution Plan: `WAL_v2_EXECUTION_PLAN_M148_M170.md`
- Execution Summary: `WAL_v2_EXECUTION_SUMMARY_M148_M170.md`
- Book chapters: `book/51_M148_WAL_v1_SPEC_FREEZE.md` through `book/73_M165_CROSS_ARCHITECTURE.md`
