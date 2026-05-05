# Phase D / M136: 12-bit Production Packing

**Date:** 2026-04-25  
**Status:** ✅ SUCCESS  
**Goal:** Implement real packed storage for WAL programs and verify correctness, speed, and size.

## Background

WAL programs store two indices per weight:
- `atom_id`: 8 bits (K=256)
- `coeff_id`: 4 bits (C=16)

Naive storage uses 2 bytes per weight (uint8 + uint8). The theoretical minimum is 12 bits = 1.5 bytes per weight.

## Packing Format

For even N weights:
```
packed[i*3]     = atom_ids[i]
packed[i*3 + 1] = atom_ids[i+1]
packed[i*3 + 2] = (coeff_ids[i] << 4) | coeff_ids[i+1]
```

Result: 2 weights → 3 bytes (24 bits = 12 bits × 2).

## Results

### Microbenchmark (10M weights)

| Metric | Value |
|--------|-------|
| Original size | 19.07 MB (2 bytes/weight) |
| Packed size | 14.31 MB (1.5 bytes/weight) |
| Reduction | **25.0%** |
| Pack time | 0.524 ms |
| Unpack time | 0.516 ms |
| Throughput | **9.6 B weights/sec** |
| Correctness | ✅ Perfect round-trip |

### Full Model (Llama-3.1-8B, 225 layers)

| Format | Size | vs bf16 |
|--------|------|---------|
| bf16 dense | 13.98 GB | 100% |
| WAL unpacked | 13.98 GB | 100% |
| **WAL packed 12b** | **10.48 GB** | **75%** |
| int8 | ~7.0 GB | 50% |
| int4 | ~3.5 GB | 25% |

### Decode Correctness

After pack → unpack → decode cycle:
- **All 225 layers**: perfect match with original decode
- **Max decode error**: 0.00

## Analysis

### Size Positioning

WAL packed 12-bit sits between bf16 and int8:
```
int4 (3.5 GB)  <  int8 (7.0 GB)  <  WAL 12b (10.5 GB)  <  bf16 (14.0 GB)
```

This is the expected trade-off:
- **int4/int8**: Maximum compression, no structure
- **WAL 12b**: Moderate compression (25% savings vs bf16) with full program structure
- **bf16**: No compression, no structure

### Speed

Pack/unpack throughput of **9.6 billion weights/sec** is negligible overhead. For inference:
- Unpack once at load time (10.5 GB → 14 GB in ~1.1s for full model)
- After unpack, inference is identical to dense

## Conclusion

**12-bit packing is production-ready.**

- 25% size reduction over naive 2-byte storage
- 25% smaller than bf16 (10.5 GB vs 14 GB)
- Perfect decode correctness
- Negligible pack/unpack overhead

**Recommended:** Make 12-bit packing the default WAL storage format.

## Artifacts

- `experiments/m136_12bit_packing.py`
- `experiments/m136_12bit_packing.json`
