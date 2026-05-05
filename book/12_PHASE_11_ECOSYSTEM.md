# Phase 11: Ecosystem (M83)

## The Problem

WAL is a complete system. But it's isolated. It doesn't integrate with Hugging Face, ONNX, or model merging tools.

## What Was Built

### HF Hub Integration
- `push_wal_model()`: Upload WAL-encoded model to HF Hub
- `pull_wal_model()`: Download and reconstruct
- `WALModelCard`: Metadata (base model, encoder config, metrics)

### ONNX Export
- `export_wal_simple()`: Pre-decode weights → standard ONNX
- `export_wal_native()`: WAL decode as ONNX ops (Gather+Mul+Reshape+MatMul)
- `verify_onnx_export()`: Bit-exact verification

### WAL-aware Mergekit
- `merge_wal_models()`: Program-level merge with 4 strategies
  - soup: atom/coeff-level merge (fastest)
  - linear: decode → average → re-encode
  - slerp: spherical interpolation
  - ties: trim, elect sign, merge
- `merge_task_vectors()`: Task arithmetic on WAL models

## Test Results

| Test | Result |
|------|--------|
| State dict extract/load | ✅ PASS |
| Model card | ✅ PASS |
| ONNX simple export | ✅ PASS (bit-exact) |
| ONNX native export | ✅ PASS (bit-exact) |
| Program soup merge | ✅ PASS |
| Linear merge | ✅ PASS |
| SLERP merge | ✅ PASS |
| Task vectors | ✅ PASS |

## Why This Matters

Without ecosystem integration, WAL is a research tool. With ecosystem integration, WAL is a production format. You can:
- Upload WAL models to HF Hub
- Export WAL models to ONNX for inference
- Merge WAL models like you merge LoRA adapters

## Files
- `src/wal/v1/hub.py`
- `src/wal/v1/onnx_export.py`
- `src/wal/v1/mergekit.py`
- `experiments/m83_ecosystem.py`
