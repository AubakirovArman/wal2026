# M63: WAL v2 VM + Triton Runtime Validation

## Date
2026-04-20

## Goal
Validate three decode paths for WAL v2: PyTorch, VM reference interpreter, and Triton kernel. Prove bit-exact equivalence and measure throughput.

## Decode Paths

### 1. PyTorch Decode (`prog.decode`)
- Vectorized gather + multiply via PyTorch indexing
- Baseline for correctness

### 2. VM Reference Interpreter (`vm_execute`)
- WALVMState with ACC, PC, atom_table, coeff_table
- Vectorized execution (all weights at once)
- Reference for formal semantics

### 3. Triton Kernel (`wal_v2_decode_triton`)
- Per-thread: one weight
- Loads atom_id, coeff_id → lookups in atom_table and coeff_table
- Adds residual if present
- Optional: applies row_scale per row

## Results

### Validation (Bit-Exact)
| Comparison | Match | Max Diff |
|-----------|-------|----------|
| PyTorch vs VM | True | 0.0000000000 |
| PyTorch vs Triton | True | 0.0000000000 |
| PyTorch vs Triton+RS | True | 0.0000000000 |

**All paths are bit-exact.** This proves:
1. VM semantics match PyTorch implementation
2. Triton kernel matches reference exactly
3. WAL v2 has a consistent execution model across all backends

### Throughput (layer 40 o_proj, 67M weights)
| Path | Time | Throughput |
|------|------|------------|
| PyTorch | 0.028s | **2,409.5 Mw/s** |
| VM | 0.081s | 833.5 Mw/s |
| Triton | 1.378s | 48.7 Mw/s |
| Triton + row scales | 0.585s | 114.7 Mw/s |

### Observations
1. **PyTorch is fastest** (2.4 Gw/s) due to heavily optimized `index_select` + `multiply` on GPU.
2. **VM is ~3× slower** than PyTorch but still fast (833 Mw/s). Acceptable for reference interpreter.
3. **Triton is slow** (49-115 Mw/s) — baseline kernel without shared memory caching of atom/coeff tables. Optimization needed.

### Triton Optimization Needed
The current kernel loads `atom_table[atom_id]` and `coeff_table[coeff_id]` from global memory per thread. With K=256 and C=16, the tables are tiny (~1KB + ~64B). Caching them in shared memory should significantly improve throughput.

Target: match or exceed PyTorch decode speed via fused kernel with shared memory atom/coeff cache.

## Artifacts
- `src/wal/v2/vm.py` — VM state + reference interpreter
- `src/wal/v2/triton_kernels.py` — Triton decode kernels
- `experiments/m63_wal_v2_vm_runtime.py`
- `experiments/m63_wal_v2_vm_runtime.log`

## Next Steps
1. **Optimize Triton kernel**: shared memory cache for atom_table + coeff_table
2. **Fused encode+decode kernel**: single kernel for end-to-end WAL v2 processing
3. **Measure full-model decode time**: decode all 540 layers, compare to dense forward pass


## Extracted Metrics (from source)

- Time: .4
- Time: .4
- Time: .4
- Time: .4
