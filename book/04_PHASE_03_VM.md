# Phase 3: VM + Runtime (M63)

## The Problem

WAL programs need to be decoded to dense weights for inference. The decoder must be fast, correct, and portable.

## What Was Built

- **Reference VM**: Python interpreter with explicit FETCH → DECODE → EXECUTE cycle
- **PyTorch decoder**: CPU/GPU compatible dense decode
- **Triton kernels**: GPU-accelerated decode at 406 Mweights/sec on H200

## Backend Performance

| Backend | Speed | Use Case |
|---------|-------|----------|
| Reference VM | ~1 Mw/s | Debugging, verification |
| PyTorch decode | ~50 Mw/s | Reference, CPU inference |
| Triton kernel | 406 Mw/s | Production GPU |
| Precomputed lookup | 1.1 TW/s | Decode = single index_select (M54b) |

## Why Multiple Backends?

- Reference VM: proves correctness
- PyTorch decoder: works everywhere PyTorch works
- Triton kernel: maximum speed on NVIDIA GPUs

## Files
- `src/wal/v2/vm.py`
- `src/wal/v2/decoder.py`
- `src/wal/v2/triton_kernels.py`
- `experiments/m63_vm_runtime.py`
