# WAL Phases: Complete Implementation Guide (M60–M82)

> Date: 2026-04-20
> Status: **Phases 1–10 COMPLETE**. Phase 11 defined and ready.

This document describes each implemented phase of the WAL (Weight Assembly Language) system, what was built, key results, and the exact files involved.

---

## Phase 1: Encoder v2 (M60–M61)

### Goal
Build a production-quality encoder that represents every weight as a single program call with continuous coefficients.

### What was built
- **Two-pass encoder**: k-means++ for atom initialization + Lloyd-Max for coefficient quantization
- **Single-call programs**: `weight = atom[atom_id] * coeff[coeff_id] + residual`
- **12 bits/weight**: 8 bits atom_id + 4 bits coeff_id

### Key results
| Metric | Value |
|--------|-------|
| PPL | **2.7781** (vs baseline 2.7805, delta −0.0024) |
| Encode time | 1810s (30 min) for full 70B |
| Encoded params | 540 |
| Skipped params | 183 (embed, lm_head, 1D, spiky) |
| Compression | 1.33× vs bf16 |

### Files
- `src/wal/v2/encoder.py` — Two-pass encoder
- `src/wal/v2/isa.py` — AtomTable, CoeffTable, ProgramBufferV2
- `experiments/m60_wal_v2_scalar_prototype.py` — Single-layer prototype
- `experiments/m61_wal_v2_70b_ppl.py` — Full 70B validation

### Why it works
- One atom call (not two like WAL-0) → simpler programs
- Continuous coefficients (C=16 levels) → more expressive than ternary {-1,0,+1}
- Lloyd-Max on sampled ratios (2M samples) → fast and stable

---

## Phase 2: Grammar & Assembler (M62)

### Goal
Create a human-readable text format for WAL programs with exact round-trip conversion.

### What was built
- **BNF grammar**: Formal grammar for WAL program syntax
- **Parser**: Text → ProgramBufferV2
- **Pretty-printer**: ProgramBufferV2 → text
- **Assembler/Disassembler**: Full bidirectional conversion

### Key results
| Metric | Value |
|--------|-------|
| Round-trip | **Exact** (max error 0.00) |
| Unique programs / layer | ~1,299 per 67M weights |
| Vocabulary ratio | 0.0016% |

### Text format example
```wal
K 256
C 16
SHAPE 8192 8192

ATOM 0 = 0.123456
ATOM 1 = -0.789012

ATOM 120 COEF 0.771360
ATOM 45 COEF 1.234567 RESIDUAL 0.001234
```

### Files
- `src/wal/v2/grammar.py` — BNF grammar + parser + pretty-printer
- `src/wal/v2/asm.py` — Assembler/disassembler
- `experiments/m62_wal_v2_grammar_asm.py` — Validation test

---

## Phase 3: VM + Runtime (M63)

### Goal
Build a reference virtual machine and fast execution backends.

### What was built
- **Reference VM**: Python interpreter with explicit FETCH → DECODE → EXECUTE cycle
- **PyTorch decoder**: CPU/GPU compatible dense decode
- **Triton kernels**: GPU-accelerated decode at 406 Mweights/sec on H200
- **Fused encode kernel**: 309× speedup vs naive Python (M53)

### Key results
| Backend | Speed | Use case |
|---------|-------|----------|
| PyTorch decode | ~50 Mw/s | Reference, debugging |
| Triton kernel | 406 Mw/s | Production GPU |
| Fused encode | 309× speedup | Full model encode |
| Precomputed lookup | 1.1 TW/s | Decode = single index_select (M54b) |

### Files
- `src/wal/v2/vm.py` — Reference interpreter
- `src/wal/v2/decoder.py` — PyTorch decode
- `src/wal/v2/triton_kernels.py` — GPU kernels
- `experiments/m63_vm_runtime.py` — VM + decoder tests

---

## Phase 4: Compression (M64)

### Goal
Serialize WAL programs to compact binary format.

### What was built
- **WAL2 binary format**: Magic header + uint4-packed coefficients + sparse residual bitmap
- **Streaming structure**: Can decode partial files
- **Bit-exact round-trip**: `serialize → deserialize == original`

### Key results
| Metric | Value |
|--------|-------|
| Format size | ~12 bits/weight (header + atom_id + coeff_id + residual bitmap) |
| Round-trip | **Bit-exact** |
| Streaming | Supported |

### Files
- `src/wal/v2/format.py` — Binary serialization
- `experiments/m64_compression.py` — Compression tests

---

## Phase 5: Hierarchical Atoms / WAL v1 (M65–M75)

### Goal
Add semantic structure to atoms while maintaining decode quality.

### What was built
- **L0/L1 hierarchy**: Base scalars (L0) + composite atoms (L1) built from weighted combinations
- **Atom table v1**: `AtomTableV1` with recursive `resolve()`
- **Encoder v1**: Same quality as v2 but richer structure
- **Binary format v1**: Supports hierarchical atom definitions

### Key results
| Metric | Value |
|--------|-------|
| PPL | **2.7809** (vs baseline 2.7805, delta +0.0004) — PASS |
| L1 atoms | 35,840 across 560 layers |
| L0 atoms | 8,960 (16 per layer) |
| L1 coverage | 5.6% of weights |
| Encode time | ~35 min for full 70B |

### Files
- `src/wal/v1/isa.py` — Hierarchical ISA
- `src/wal/v1/encoder.py` — v1 encoder
- `src/wal/v1/decoder.py` — v1 decoder (hierarchical + flat paths)
- `src/wal/v1/format.py` — v1 binary format
- `experiments/m65_v1_hierarchy_prototype.py` — Prototype
- `experiments/m75_v1_full_70b_ppl.py` — Full validation

### Why WAL v1 matters
WAL v1 is NOT for compression — it's for **interpretability and structure**. The 12-bit floor is hard. WAL v1 adds a semantic layer on top of the same 12-bit representation.

---

## Phase 6: PyTorch Integration (M76–M77)

### Goal
WAL-encoded weights work as drop-in replacements for `nn.Linear`.

### What was built
- **`WALParameter`**: Stores `prog + atom_table + coeffs`, lazy decode with cache
- **`WALLinear`**: Decodes on-the-fly per forward pass
- **`WALCachedLinear`**: Decodes once, caches for speed
- **`replace_linear_with_wal()`**: Converts entire models
- **State dict serialization**: `wal_state_dict()` / `wal_load_state_dict()`

### Test results
| Test | Result |
|------|--------|
| WALParameter decode | ✅ PASS |
| WALLinear forward | ✅ PASS |
| WALCachedLinear | ✅ PASS |
| Replace nn.Linear | ✅ PASS |
| Device transfer | ✅ PASS |

### Files
- `src/wal/v1/nn.py` — PyTorch integration
- `experiments/m76_pytorch_roundtrip.py` — Round-trip test
- `experiments/m77_pytorch_integration.py` — Full integration test

---

## Phase 7: Debugger (M78)

### Goal
Inspect, trace, and debug WAL programs interactively.

### What was built
- **`WALDebugger`**: Step-through execution
- **Breakpoints**: Atom ID, coefficient, residual, custom conditions
- **Trace log**: Full execution history
- **Heatmap**: Atom/coeff frequency and entropy statistics
- **Program diff**: Compare two encodings
- **Hierarchical trace**: Recursive tree for L1+ atoms

### Test results
| Test | Result |
|------|--------|
| Step-through | ✅ PASS |
| Atom breakpoint | ✅ PASS |
| Coeff breakpoint | ✅ PASS |
| Residual breakpoint | ✅ PASS |
| Custom breakpoint | ✅ PASS |
| Heatmap | ✅ PASS |
| Program diff | ✅ PASS |

### Files
- `src/wal/v1/debugger.py` — Debugger
- `experiments/m78_debugger.py` — Tests

---

## Phase 8: Standard Library (M79)

### Goal
Pre-trained atom tables that transfer across models.

### What was built
- **`AtomLibraryEntry`**: Named atom entry with metadata
- **`AtomLibrary`**: Collection with query, save, load
- **`encode_with_pretrained_atoms()`**: k-means fine-tuning on target using source init
- **`transfer_atoms_direct()`**: Scale-only transfer
- **Save format**: Directory with `.pt` per entry + `manifest.json`

### Test results
| Test | Result |
|------|--------|
| Create library | ✅ PASS |
| Save/load | ✅ PASS |
| Query | ✅ PASS |
| Transfer (Llama-70B → 8B) | ✅ PASS (MSE ratio 1.0094) |
| Direct transfer | ✅ PASS (MSE ratio 0.9332) |
| Full pipeline | ✅ PASS |

### Key finding
Atom transfer works: pre-trained atoms from similar distributions achieve MSE ratio ~1.0 vs from-scratch encoding. Even direct scale-only transfer can beat baseline.

### Usage example
```python
from wal.v1 import AtomLibrary, build_entry_from_encoded
from wal.v1 import encode_with_pretrained_atoms

# Build library
lib = AtomLibrary(name="wal-zoo")
entry = build_entry_from_encoded(
    name="llama-70b-q_proj", family="llama", variant="70b",
    component="attention", atoms=atoms, coeffs=coeffs,
)
lib.add_entry(entry)
lib.save("./wal_zoo/")

# Transfer to new model
lib2 = AtomLibrary.load("./wal_zoo/")
source = lib2.get_entry("llama-70b-q_proj")
transfer_atoms, transfer_coeffs, recon = encode_with_pretrained_atoms(
    target_weights, source
)
```

### Files
- `src/wal/v1/stdlib/library.py`
- `src/wal/v1/stdlib/transfer.py`
- `experiments/m79_stdlib_prototype.py`

---

## Phase 9: Hardware Backends (M80)

### Goal
WAL runs on CPU, GPU, and browser.

### What was built
- **`WALBackend` ABC**: Abstract interface for all backends
- **`CPUBackend`**: NumPy vectorized decode (always available)
- **`CUDABackend`**: PyTorch GPU decode (production path)
- **`MPSBackend`**: Metal/MPS scaffolding (macOS)
- **`ROCmBackend`**: AMD ROCm/HIP scaffolding
- **`WebGPUBackend`**: WebGPU scaffolding + WGSL shader generator
- **Auto-selection**: `select_best_backend()` picks optimal

### Test results
| Test | Result |
|------|--------|
| Registry (5 backends) | ✅ PASS |
| CPU decode | ✅ PASS (bit-exact) |
| CUDA decode | ✅ PASS (bit-exact) |
| Cross-backend consistency | ✅ PASS |
| Benchmark | ✅ PASS |
| WGSL shader | ✅ PASS |
| Backend selection | ✅ PASS |
| Scaffold backends | ✅ PASS |

### Backend matrix

| Backend | Status | Availability |
|---------|--------|-------------|
| CPU (NumPy) | ✅ Full | Always |
| CUDA | ✅ Full | NVIDIA GPU |
| MPS (Metal) | 🏗️ Scaffold | macOS |
| ROCm (AMD) | 🏗️ Scaffold | AMD GPU |
| WebGPU | 🏗️ Scaffold | Browser / wgpu-py |

### Usage
```python
from wal.backends import get_backend, available_backends, select_best_backend

# Auto-select best backend
best = select_best_backend()  # CUDA on GPU, CPU otherwise
decoded = best.decode(atom_ids, coeff_ids, atom_table, coeffs, shape=shape)

# WebGPU shader for browser deployment
webgpu = get_backend('webgpu')
wgsl = webgpu.generate_wgsl_shader(K=256, C=16)
```

### Files
- `src/wal/backends/base.py`
- `src/wal/backends/cpu.py`
- `src/wal/backends/cuda.py`
- `src/wal/backends/mps.py`
- `src/wal/backends/rocm.py`
- `src/wal/backends/webgpu.py`
- `src/wal/backends/__init__.py`

---

## Phase 10: Meta-Learning (M81–M82)

### Goal
Fine-tune programs instead of weights. Model soups at program level. Task-specific adapters.

### What was built
- **`WALProgramAdapter`**: LoRA-style residual adapter (rank=4, learned A×B)
- **`WALCoeffAdapter`**: Learned coefficient offsets (atom × (coeff + Δ))
- **`WALAtomAdapter`**: Learned atom perturbations (selective atom updates)
- **`program_soup()`**: Merge programs from N models (mean / majority / weighted)
- **`evolve_programs()`**: Genetic algorithm on atom combinations
- **Integration**: `WALCachedLinear.set_adapter()` for drop-in adapter use

### Test results
| Test | Result |
|------|--------|
| WALProgramAdapter | ✅ PASS |
| WALCoeffAdapter | ✅ PASS |
| WALAtomAdapter | ✅ PASS |
| Program soup (mean/majority/weighted) | ✅ PASS |
| Genetic evolution | ✅ PASS |
| Adapter + WALCachedLinear | ✅ PASS |
| Adapter detach/restore | ✅ PASS |
| Gradient flow | ✅ PASS |

### Adapter parameter counts
| Adapter | Params for 64×32 weight | Relative to full weight |
|---------|------------------------|------------------------|
| WALProgramAdapter (rank=4) | 384 | 18.8% |
| WALCoeffAdapter (C=8) | 8 | 0.4% |
| WALAtomAdapter (8 atoms) | 40 | 2.0% |

### Usage
```python
from wal.v1.meta import WALProgramAdapter
from wal.v1.nn import WALCachedLinear

# Create base WAL layer
layer = WALCachedLinear(wal_weight, bias=bias)

# Attach task-specific adapter (only adapter params are trainable)
adapter = WALProgramAdapter(shape=(4096, 4096), rank=4, alpha=1.0)
layer.set_adapter(adapter)

# Forward with adapted weights
output = layer(input)

# Merge adapter into base for inference
merged_weight = adapter.merge(base_weight)
```

### Genetic evolution example
```python
from wal.v1.meta import evolve_programs

best_prog, recon = evolve_programs(
    target_weights, atom_table, coeffs,
    population_size=16, generations=20,
    mutation_rate=0.05, crossover_rate=0.5,
)
# Returns: (best program, reconstructed weights)
```

### Files
- `src/wal/v1/meta.py` — Meta-learning components
- `src/wal/v1/nn.py` — PyTorch integration with adapter support
- `experiments/m81_meta_learning.py` — Core tests
- `experiments/m82_adapter_integration.py` — Integration tests

---

## Phase 11: Ecosystem (M83)

### Goal
WAL integrates with the broader ML ecosystem.

### What was built
- **`extract_wal_state_dict()` / `load_wal_state_dict()`**: Serialize/deserialize complete WAL models
- **`WALModelCard`**: Metadata for WAL models (base model, encoder config, metrics)
- **`push_wal_model()` / `pull_wal_model()`**: Upload/download to HF Hub using safetensors + WAL binary
- **`export_wal_simple()`**: Pre-decode weights → standard ONNX (bit-exact verified)
- **`export_wal_native()`**: WAL decode as ONNX ops (Gather + Mul + Reshape + MatMul)
- **`verify_onnx_export()`**: Bit-exact verification against PyTorch
- **`merge_wal_models()`**: Program-level merge with 4 strategies
- **`merge_task_vectors()`**: Task arithmetic on WAL models

### Merge strategies
| Strategy | How it works | Use case |
|----------|-------------|----------|
| `soup` | Merge programs at atom/coeff level | Fast, no re-encode |
| `linear` | Decode → average → re-encode | Standard model average |
| `slerp` | Spherical interpolation | 2-model interpolation |
| `ties` | Trim, elect sign, merge | Conflict resolution |
| `task_vectors` | (finetuned - base) averaging | Task arithmetic |

### Test results
| Test | Result |
|------|--------|
| State dict extract/load | ✅ PASS |
| Model card | ✅ PASS |
| ONNX simple export | ✅ PASS (bit-exact) |
| ONNX native export | ✅ PASS (bit-exact) |
| Program soup merge | ✅ PASS |
| Linear merge | ✅ PASS |
| SLERP merge | ✅ PASS |
| Task vectors (bonus) | ✅ PASS |

### Usage
```python
from wal.v1.hub import push_wal_model, pull_wal_model, WALModelCard
from wal.v1.onnx_export import export_wal_simple, verify_onnx_export
from wal.v1.mergekit import MergeConfig, merge_wal_models

# HF Hub
card = WALModelCard(base_model="meta-llama/Llama-3.3-70B", metrics={"ppl": 2.78})
push_wal_model(model, repo_id="user/wal-llama-70b", card=card)

# ONNX
export_wal_simple(model, dummy_input, filepath="model.onnx")

# Merge
config = MergeConfig(method="soup", soup_method="mean")
merged = merge_wal_models([model_a, model_b, model_c], config)
```

### Files
- `src/wal/v1/hub.py` — HF Hub integration
- `src/wal/v1/onnx_export.py` — ONNX export (simple + native)
- `src/wal/v1/mergekit.py` — WAL-aware merging
- `experiments/m83_ecosystem.py` — Tests

---

## Summary

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| 1: Encoder v2 | ✅ | PPL 2.7781, 1.33× compression |
| 2: Grammar | ✅ | Exact round-trip text format |
| 3: VM + Runtime | ✅ | 406 Mw/s GPU decode |
| 4: Compression | ✅ | Bit-exact binary format |
| 5: Hierarchy v1 | ✅ | PPL 2.7809 PASS, semantic atoms |
| 6: PyTorch Integration | ✅ | Drop-in nn.Linear replacement |
| 7: Debugger | ✅ | Step-through, breakpoints, heatmaps |
| 8: Standard Library | ✅ | Atom transfer across models |
| 9: Hardware Backends | ✅ | CPU/CUDA/MPS/ROCm/WebGPU |
| 10: Meta-Learning | ✅ | Program adapters, soups, evolution |
| 11: Ecosystem | ✅ | HF Hub, ONNX, mergekit |
| 12: KV-cache WAL | ✅ | M84-M88, 2× compression, production module |

**Score: 12/12 phases complete. 🎉**
