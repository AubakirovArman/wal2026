# WAL v1 Specification — Frozen Draft

**Version:** 1.0-frozen  
**Date:** 2026-04-20  
**Status:** Production-ready core frozen. Extensions (transforms, differentiable training) belong to WAL v2/v3.

---

## 1. Overview

WAL (Weight Assembly Language) v1 is a structured weight representation that enables deterministic, reversible editing of neural network parameters through discrete programs.

**Core principle:** Every weight is represented as `atom_id × coeff_id`, where atoms are learned scalar prototypes and coefficients are learned scale factors.

**Key properties:**
- Deterministic encode/decode (with canonicalization)
- Global atom vocabulary (single table serves all layers)
- 12-bit packed format (8-bit atom_id + 4-bit coeff_id)
- Dense-speed inference (after cache warmup)
- Hybrid edit workflow: edit in weight space, store in WAL space

---

## 2. Data Structures

### 2.1 Atom Table

```python
@dataclass
class AtomTableV1:
    base_atoms: torch.Tensor   # [K0] float32 — L0 scalar prototypes
    atom_defs: List[AtomDef]   # [K_total] hierarchical definitions
```

- `K0`: number of base atoms (default: 256)
- `K_total`: total atoms including hierarchical composites
- Base atoms are built via k-means++ on weight samples
- Hierarchical atoms (L1+) are built from co-occurrence statistics (optional)

### 2.2 Coefficient Table

```python
@dataclass
class CoeffTable:
    values: torch.Tensor  # [C] float32 — scale factors
```

- `C`: number of coefficients (default: 16)
- Built via k-means on `|w / atom|` ratios

### 2.3 Program Buffer

```python
@dataclass
class ProgramBufferV1:
    atom_ids: torch.Tensor      # [N] uint8
    coeff_ids: torch.Tensor     # [N] uint8
    residuals: torch.Tensor     # [N] float16 or empty
    has_residual: torch.Tensor  # [N] bool
    shape: Tuple[int, ...]
```

- `N`: total number of weights
- Each weight maps to one `(atom_id, coeff_id)` pair
- Optional residuals for outlier weights

---

## 3. Binary Format

### 3.1 Header (32 bytes)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| magic | bytes | 4 | `b'WAL1'` |
| version | uint16 | 2 | `1` |
| K0 | uint16 | 2 | Base atoms count |
| K_total | uint16 | 2 | Total atoms |
| C | uint16 | 2 | Coefficients count |
| flags | uint16 | 2 | Bit 0: has_residuals, Bit 1: packed_coeffs |
| N_weights | uint64 | 8 | Total weights |
| reserved | bytes | 10 | Padding |

### 3.2 Body

```
Base Atom Table:     K0 × 4 bytes (float32)
Hierarchical Defs:   variable (see format.py)
Coeff Table:         C × 4 bytes (float32)
Atom IDs:            N × 1 byte (uint8)
Coeff IDs:           ceil(N/2) bytes (uint4 packed) OR N × 1 byte
Residual Bitmap:     ceil(N/8) bytes
Residual Count:      uint32
Residual Indices:    count × uint32 (if count > 0)
Residual Values:     count × 2 bytes (float16, if count > 0)
Metadata:            uint64 length + JSON bytes
```

### 3.3 12-Bit Packing

Default mode packs two weights into 3 bytes:

```
weight[i]:   atom_id[i]   (8 bits) + coeff_id[i]   (4 bits) = 12 bits
weight[i+1]: atom_id[i+1] (8 bits) + coeff_id[i+1] (4 bits) = 12 bits

Packed: [atom_id[i], coeff_id[i] << 4 | coeff_id[i+1], atom_id[i+1]]
```

This gives **1.5 bytes/weight** = 25% reduction vs naive 2 bytes/weight.

---

## 4. Encode Pipeline

### 4.1 Canonicalization (MANDATORY)

Without canonicalization, same weights → different programs (k-means permutation noise).

```python
def canonicalize_atoms(atoms: torch.Tensor) -> torch.Tensor:
    """Sort atoms by absolute value for deterministic ordering."""
    return atoms[torch.argsort(atoms.abs())]
```

**Rule:** Atom tables MUST be canonicalized before encoding. This ensures:
- Same weights → identical programs
- Re-encode stability: 97.5% survival rate
- Diff locality: edits affect only target layers (with frozen table)

### 4.2 Frozen Vocabulary Mode

```
1. Build global atom table from base model (all layers)
2. Canonicalize
3. Freeze table — never rebuild
4. Encode base model with frozen table
5. Edit in dense weight space
6. Re-encode edited model with SAME frozen table
7. Compare diff: target vs non-target layers
```

**Result:** Non-target diff = 0.000%. Target diff = ~25% (localized).

### 4.3 Encode Algorithm

```python
def wal_encode_v1(weights, atoms, coeffs, batch=1_048_576):
    N = weights.numel()
    atom_ids = torch.empty(N, dtype=torch.uint8)
    coeff_ids = torch.empty(N, dtype=torch.uint8)
    
    for start in range(0, N, batch):
        end = min(start + batch, N)
        w = weights[start:end]
        recons = atoms.unsqueeze(1) * coeffs.unsqueeze(0)  # [K, C]
        errs = (w.unsqueeze(1).unsqueeze(2) - recons.unsqueeze(0)).abs()
        best = errs.view(end - start, -1).argmin(dim=1)
        atom_ids[start:end] = best // C
        coeff_ids[start:end] = best % C
    
    return atom_ids, coeff_ids
```

**Complexity:** O(N × K × C) per batch. With K=256, C=16: ~4K operations/weight.

---

## 5. Decode Pipeline

```python
def wal_decode_v1(prog, atom_table, coeffs):
    flat_atoms = atom_table.base_atoms[prog.atom_ids]
    flat_coeffs = coeffs[prog.coeff_ids]
    weights = flat_atoms * flat_coeffs
    if prog.has_residual.any():
        weights += prog.residuals
    return weights.reshape(prog.shape)
```

**Complexity:** O(N) — two indexing operations and one multiply.

---

## 6. PyTorch Integration

### 6.1 Layer Types

| Class | Use Case |
|-------|----------|
| `WALLinear` | Decode-on-the-fly, low memory |
| `WALCachedLinear` | Decode once, dense-speed inference |

### 6.2 Model Conversion

```python
# Convert nn.Linear → WALLinear
replace_linear_with_wal(model, K=256, C=16, cached=True)

# Convert WALLinear → nn.Linear (for editing)
replace_wal_with_linear(model)
```

### 6.3 State Dict

```python
# Save WAL-encoded state
state = wal_state_dict(model)  # dict of binary blobs

# Load WAL-encoded state
wal_load_state_dict(model, state)
```

---

## 7. Edit Workflow

### 7.1 Standard Workflow

```
WAL base model
  → decode() → dense weights
  → apply LoRA edit (rank=r, steps=s)
  → merge edit into dense weights
  → re-encode() with frozen atom table
  → WAL edited model
```

### 7.2 Safety Rules

| Rule | Rationale |
|------|-----------|
| rank ≥ 4 | rank=1 with steps≥200 causes catastrophic collapse (+58.7 PPL) |
| steps ≤ 50 for rank=4 | Optimal ΔPPL = +0.49 |
| steps ≤ 100 for rank≥4 | Acceptable ΔPPL = +1.0–2.6 |
| Freeze atom table | Rebuilding causes 25% diffuse diff everywhere |
| Canonicalize before encode | Without: 97.72% diff from permutation noise |

### 7.3 Safety Score

```python
def safety_score(delta_W: torch.Tensor) -> str:
    spectral = torch.linalg.matrix_norm(delta_W, ord=2).item()
    if spectral < 1.0:     return "SAFE"
    elif spectral < 5.0:   return "MODERATE"
    elif spectral < 10.0:  return "RISKY"
    else:                  return "DANGEROUS"
```

**Validation:** Spectral norm correlates with ΔPPL at r=0.9905.

---

## 8. WAL+LoRA Overlay

### 8.1 Architecture

```
Base:     WALCachedLinear (WAL-encoded, frozen)
Edit:     LoRAOverlay(A, B, scaling)
Runtime:  output = base(x) + lora(x)
```

### 8.2 Properties

- Base size: ~32 MB per layer (WAL-decoded cache)
- LoRA size: ~0.094 MB (rank=4)
- **341× smaller than full layer**
- Multiple overlays per layer, enable/disable independently
- Disabled overlay → exact base match (diff ≈ 0)

---

## 9. Patch Format v2

### 9.1 Problem

WAL-diff is 25% uniform noise even with frozen table. Direct patch = 10.7 GB.

### 9.2 Solution

For WAL-native edits, use bitmask patch:

```
Patch = {
    layer_name: {
        changed_mask: bit-packed boolean [N/8 bytes]
        new_atom_ids: uint8 [n_changed]
        new_coeff_ids: uint8 [n_changed]
    }
}
```

### 9.3 Size

| Method | Size |
|--------|------|
| Raw diff | 10.7 GB |
| Bitmask patch | 32.92 MB |
| RLE patch | 35.08 MB |
| LoRA | 0.19 MB |

**Verdict:** LoRA remains gold standard for edit distribution. WAL patch viable only for structural edits that cannot be expressed as LoRA.

---

## 10. Performance

### 10.1 Quality

| Metric | Value |
|--------|-------|
| PPL (WikiText-2, full model) | 10.03 (per-layer) / 10.06 (global) |
| PPL degradation | +0.03 (+0.3%) |
| relMSE | ~10⁻⁶ |

### 10.2 Speed

| Operation | Time |
|-----------|------|
| Full model encode (per-layer) | ~300s |
| Full model encode (global) | ~216s |
| Full model decode | ~100s |
| Inference (cached) | 0.97–1.02× dense speed |

### 10.3 Size

| Format | Size (8B model) |
|--------|----------------|
| bf16 | 16.06 GB |
| WAL (naive) | 15.01 GB |
| WAL (12-bit packed) | **10.48 GB** |
| Atom table (global) | 262 KB |

---

## 11. Compatibility Guarantees

### 11.1 Forward Compatibility

- WAL v1 spec is frozen
- Future versions (v2, v3) will use different magic bytes
- v1 decoders must reject v2/v3 blobs

### 11.2 Canonicalization Guarantee

- Canonicalized atom tables produce bit-identical programs for identical weights
- This is the ONLY way to achieve deterministic encode

### 11.3 Frozen Table Guarantee

- With frozen atom table, non-target layers show 0% diff after re-encode
- Target layers show ~25% diff (localized to edited parameters)

---

## 12. Killed Directions (Will Not Be Pursued in v1)

| Direction | Status | Reason |
|-----------|--------|--------|
| WAL-diff patch | ❌ Closed | 57,000× worse than LoRA |
| Program soup | ❌ Closed | PPL 6.4×10¹³ |
| Program evolution | ❌ Closed | GA 170M× worse than greedy |
| Semantic atoms | ❌ Closed | Entropy 0.966, no specialization |
| Native surgical editing | ❌ Closed | Diffuse + huge patch |
| Wave-Atom ISA | ❌ Closed | 65–584× worse (M143) |
| Graph-WAL | ❌ Closed | 7–110× worse (M144) |
| Simple WAL regularizer | ❌ Closed | Worse than L2 (M147) |

---

## 13. References

- `src/wal/v1/` — Core implementation
- `experiments/m133_fixed_atom_table.py` — Frozen vocabulary validation
- `experiments/m136_12bit_packing.py` — Packing validation
- `experiments/m138_reencode_loss_sweep.py` — Safety rules
- `experiments/m139_wal_patch_v2.py` — Patch format
- `experiments/m140_wal_lora_multi.py` — Multi-overlay
- `experiments/m141_reencode_geometry.py` — Safety score
- `ROADMAP_v3.md` — Complete project history
- `WAL_v2_EXECUTION_PLAN_M148_M170.md` — Next phase
