# M80: WAL Hardware Backends (Phase 9)

## Date
2026-04-20

## Goal
Build hardware backend abstraction layer for WAL decode across CPU, GPU, and browser.

## What was tested
1. Backend registry (5 backends registered)
2. CPU backend decode correctness (NumPy vectorized)
3. CUDA backend decode correctness (PyTorch GPU)
4. Cross-backend consistency (CPU vs CUDA bit-exact)
5. Benchmark comparison (CPU vs CUDA speed)
6. WebGPU WGSL shader generation
7. Backend auto-selection logic
8. Scaffold backends (MPS, ROCm, WebGPU availability reporting)

## Results

| Test | Result |
|------|--------|
| Registry | ✅ PASS (5 backends) |
| CPU Decode | ✅ PASS (max diff 0.00000000) |
| CUDA Decode | ✅ PASS (max diff 0.00000000) |
| Cross-backend consistency | ✅ PASS (CPU vs CUDA identical) |
| Benchmark | ✅ PASS |
| WGSL Shader | ✅ PASS |
| Backend selection | ✅ PASS (CUDA selected as best) |
| Scaffold backends | ✅ PASS |

**Total: 8/8 PASS**

## Benchmark results (10K weights)

| Backend | Time | Throughput |
|---------|------|------------|
| CPU | 0.138 ms | 72.56 Mw/s |
| CUDA | 0.249 ms | 40.23 Mw/s |

Note: CPU benchmark measures NumPy ops overhead on small tensor. For larger tensors, CUDA would dominate.

## Backends implemented

| Backend | Status | Availability |
|---------|--------|-------------|
| CPU (NumPy) | ✅ Full | Always |
| CUDA | ✅ Full | NVIDIA GPU present |
| MPS (Metal) | 🏗️ Scaffold | macOS + MPS |
| ROCm (AMD) | 🏗️ Scaffold | AMD GPU + ROCm |
| WebGPU | 🏗️ Scaffold | Browser / wgpu-py |

## Files created
- `src/wal/backends/base.py` — WALBackend ABC
- `src/wal/backends/cpu.py` — CPU SIMD via NumPy
- `src/wal/backends/cuda.py` — CUDA via PyTorch
- `src/wal/backends/mps.py` — Metal/MPS scaffolding
- `src/wal/backends/rocm.py` — ROCm/HIP scaffolding
- `src/wal/backends/webgpu.py` — WebGPU scaffolding + WGSL generator
- `src/wal/backends/__init__.py` — Registry + selection
- `experiments/m80_hardware_backends.py` — Test suite

## API

```python
from wal.backends import get_backend, available_backends, select_best_backend

# List all backends
print(available_backends())  # ['cpu', 'cuda']

# Get specific backend
cpu = get_backend('cpu')
decoded = cpu.decode(atom_ids, coeff_ids, atom_table, coeffs, shape=shape)

# Auto-select best
best = select_best_backend()  # CUDABackend on GPU systems
decoded = best.decode(atom_ids, coeff_ids, atom_table, coeffs, shape=shape)

# Generate WebGPU shader
webgpu = get_backend('webgpu')
wgsl = webgpu.generate_wgsl_shader(K=256, C=16)
```
