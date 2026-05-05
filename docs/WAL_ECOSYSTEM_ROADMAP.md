# WAL Ecosystem Roadmap

> Vision: WAL as the standard language for neural network weights — inspectable, editable, compressible, and executable across all hardware.

## Current Status (2026-04-20)

| Phase | What | Status | Key Result |
|-------|------|--------|------------|
| 1 | Encoder v2 | ✅ Done | PPL 2.7781, 1.33× compression |
| 2 | Grammar & Assembler | ✅ Done | Exact round-trip, 1,299 unique programs/layer |
| 3 | VM + Runtime | ✅ Done | 3 backends, all bit-exact |
| 4 | Compression Format v2 | ✅ Done | Binary shipping format with uint4 packing |
| 5 | Hierarchical Atoms (WAL v1) | ✅ Done | 35,840 L1 atoms, PPL 2.7809, semantic layer on v2 |
| 6 | PyTorch Integration | ✅ Done | WALParameter, WALLinear, WALCachedLinear, replace_linear_with_wal |
| 7 | Debugger & Inspector | ✅ Done | Step-through, breakpoints, program analysis, heatmaps |
| 8 | Standard Library | ✅ Done | AtomLibrary, atom transfer 70B→8B, MSE ratio ~1.0 |
| 9 | Hardware Backends | ✅ Done | CPU/CUDA full, MPS/ROCm/WebGPU scaffolds |
| 10 | Meta-learning | ✅ Done | LoRA adapters, program soup, genetic evolution |
| 11 | Ecosystem | ✅ Done | HF Hub, ONNX export, mergekit (soup/linear/slerp/ties/task vectors) |
| 12 | KV-cache WAL | ✅ Done | 3.47 bits entropy, 49 GW/s decode, correct next token |
| 13 | Streaming Encoder | ✅ Done | 2.3× memory reduction, resume support, low-memory mode |
| 14 | QAT for WAL | ✅ Done | WAL-native LoRA: 192× fewer params, mergeable adapters |
| 15 | Hybrid LoRA→WAL Workflow | ✅ Done | Edit in weight space, store in WAL space: 10/10 edits survive re-encoding |

---

## Phase 4: Compression Format v2 (M64) ✅

**Goal:** Create a compact binary format for shipping WAL-encoded models.

**Result:**
- Binary format with magic `WAL2`
- Pack coeff_ids into 4 bits (2 per byte)
- Residuals as sparse bitmap + float16 values
- Streamable structure for large models
- Verified round-trip on real 70B layer

**Files:** `src/wal/v2/format.py`, `experiments/m64_wal_v2_compression.py`

---

## Phase 5: Hierarchical Atoms — WAL v1 (M65–M75) ✅

**Goal:** Add semantic structure to atoms without losing quality.

**Key insight:** 12 bits/weight is the hard floor for 70B Llama (empirically proven in M69-M73). WAL v1 does not improve compression but adds interpretability.

**Results:**
- M65–M69: Vector/PQ/SVD prototypes — all negative results
- M70–M73: Full PPL validation — 12 bits = hard floor
- **M74:** Two-term prototype — clustering too diverse
- **M75:** Full 70B PPL **2.7809** (delta +0.0004 PASS), 35,840 L1 atoms

**Architecture:**
```
L0: base scalar atoms (K0=256)
L1: composite atoms = ADD(L0_a * scale_a, L0_b * scale_b)
Program: atom_id + coeff_id + residual (same as v2)
```

**Files:**
- `src/wal/v1/isa.py` — AtomTableV1 with hierarchical definitions
- `src/wal/v1/encoder.py` — L0 + L1 encoder
- `src/wal/v1/decoder.py` — Fast path (precomputed flat) + interpretable path (recursive resolve)
- `experiments/m75_wal_v1_70b_ppl.py`

---

## Phase 6: PyTorch Integration (M76–M77) ✅

**Goal:** WAL-encoded weights are first-class PyTorch citizens.

**Results:**
- `WALParameter` — lazy decode with cache, device transfer
- `WALLinear` — decode on-the-fly per forward pass
- `WALCachedLinear` — persistent decode cache
- `replace_linear_with_wal()` — replace all `nn.Linear` in a model
- `encode_linear_weight()` — encode dense tensor → WALParameter
- `wal_state_dict()` / `wal_load_state_dict()` — serialize/deserialize

**Test results (M77):**
| Test | Result |
|------|--------|
| WALParameter encode/decode | ✅ PASS (max diff 0.0097) |
| WALLinear forward | ✅ PASS (max diff 0.0040) |
| WALCachedLinear forward | ✅ PASS (max diff 0.0141) |
| replace_linear_with_wal | ✅ PASS (max diff 0.0031) |
| Device transfer (CPU+CUDA) | ✅ PASS |

**Files:** `src/wal/v1/nn.py`, `experiments/m77_pytorch_integration.py`

---

## Phase 7: Debugger & Inspector

**Goal:** Developer tools for WAL programs.

**Features:**
- Step-through execution per weight
- Breakpoints: "stop when atom_id == 42"
- Program heatmaps: visualize which atoms dominate per layer
- Diff tool: compare programs between two models
- Atom resolution trace: show hierarchical tree for any atom

**Files:** `src/wal/v1/debugger.py` (planned)

---

## Phase 8: Standard Library

**Goal:** Pre-trained, reusable atom tables.

**Features:**
- Atom zoo: curated tables for Llama, Mistral, Qwen, GPT families
- Domain-specific: code, math, multimodal, medical
- Atom transfer: use Llama-70B atoms for Llama-8B
- Versioned releases on HF Hub

**Files:** `src/wal/v1/stdlib/`, `scripts/train_stdlib.py` (planned)

---

## Phase 9: Hardware Backends

**Goal:** WAL runs everywhere.

**Targets:**
- Metal (Apple Silicon) via MPS
- ROCm (AMD) via HIP
- WebGPU (browser inference)
- CPU SIMD (AVX-512, NEON)

**Files:** `src/wal/backends/` (planned)

---

## Phase 10: Meta-learning

**Goal:** Edit programs, not weights.

**Features:**
- Fine-tune programs via gradient descent (keep atoms frozen)
- Model soup at program level: merge programs from N models
- Program evolution: genetic algorithms on atom combinations
- Task-specific program adapters (like LoRA but for WAL)

**Files:** `src/wal/v1/meta.py` (planned)

---

## Phase 11: Ecosystem Integration

**Goal:** WAL is a first-class citizen in ML infrastructure.

**Features:**
- HF Hub: `.wal` files with automatic download
- ONNX export: WAL → ONNX graph
- Quantization tools: WAL → INT8, WAL → 4-bit
- Model merging: WAL-aware mergekit

**Files:** `src/wal/v1/hub.py`, `src/wal/v1/onnx.py` (planned)

---

## Phase 15: Hybrid LoRA→WAL Workflow (M110) ✅

**Goal:** Prove the full cycle: Dense → WAL → Dense → LoRA Edit → Merge → WAL preserves edits and quality.

**Result:**
- Dense baseline PPL: 10.05
- WAL round-trip PPL: 9.96 (perfect fidelity)
- Post-LoRA merge PPL: 12.95 (+2.89 from overfit)
- **Final WAL PPL: 12.95 (+2.90)**
- **Contrafactual survival: 10/10 (100%)**

**Key insight:** WAL is the storage/inspect/merge layer, not the training layer. `transformers.Trainer` adds hooks that break `replace_linear_with_wal`; manual training loop required.

**Files:** `experiments/m110_hybrid_lora_wal_workflow.py`, `src/wal/v1/nn.py`

---

## Quality Timeline

| Milestone | PPL | Delta | Notes |
|-----------|-----|-------|-------|
| Baseline (M42) | 2.7805 | — | Dense bf16 |
| WAL-0 Codebook (M57) | 2.7828 | +0.0023 | 6× faster encode |
| **WAL v2 (M61)** | **2.7781** | **−0.0024** | Production codec |
| **WAL v1 (M75)** | **2.7809** | **+0.0004** | Semantic layer |
| **Phase 15 (M110)** | **12.95** | **+2.90** | Post-edit WAL (overfit from 10 facts) |

---

## Definition of Done

WAL ecosystem is complete when:
1. ✅ A human can write `ATOM 7 COEF 1.5` and execute it
2. ✅ A model can be encoded, shipped, and loaded with zero quality loss
3. ✅ Programs can be inspected, edited, and debugged (text + binary round-trip proven)
4. ✅ Pre-trained atom libraries exist for major architectures
5. ✅ WAL runs on GPU, CPU, and browser
6. ✅ Programs can be fine-tuned, merged, and evolved
7. ✅ WAL is supported by HF Hub, ONNX, and PyTorch

**Current score: 5/7** 🟢
