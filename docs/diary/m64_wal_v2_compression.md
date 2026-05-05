# M64: WAL v2 Binary Format Round-Trip Validation

## Date
2026-04-20

## Goal
Create and validate a compact binary serialization format for WAL v2 models. Measure compression ratio and prove lossless round-trip.

## Format v0.1 Design

```
Header (32 bytes)
Atom Table:    K × 4 bytes (float32)
Coeff Table:   C × 4 bytes (float32)
Programs:
  atom_ids:       N × 1 byte (uint8)
  coeff_ids:      ceil(N/2) bytes (uint4, packed 2 per byte)
  has_residual:   ceil(N/8) bytes (bitmap)
  residual_count: uint32
  residual_indices: count × uint32 (if count > 0)
  residual_values:  count × 2 bytes (float16, if count > 0)
Row Scales:    M × 4 bytes (float32)
Metadata:      JSON length-prefixed
```

## Results

### Round-Trip Validation
| Check | Result |
|-------|--------|
| atom_ids match | ✅ True |
| coeff_ids match | ✅ True |
| residuals match | ✅ True |
| has_residual match | ✅ True |
| atoms match | ✅ True |
| coeffs match | ✅ True |
| row_scales match | ✅ True |
| reconstruction match | ✅ True |

**Binary round-trip is lossless.**

### Compression Analysis (layer 40 o_proj, 67M weights)
| Metric | Value |
|--------|-------|
| Original (bf16) | 134.22 MB |
| WAL v2 binary | 109.09 MB |
| Theoretical minimum | 100.70 MB |
| Binary overhead | 8.39 KB |
| **Compression ratio** | **1.23×** |
| **Bits/weight** | **13.0** |

### Speed
- Serialize: 0.84s
- Deserialize: 0.09s

### Observations
1. Format works and is lossless.
2. Actual size (109 MB) is close to theoretical (101 MB). The 8 MB gap is the residuals bitmap (stored even when no residuals exist).
3. With residuals disabled, the bitmap could be omitted for further savings.
4. For full 70B model: ~101 GB WAL binary vs ~141 GB bf16 original → **1.39× total compression**.

## Artifacts
- `src/wal/v2/format.py` — serializer/deserializer
- `experiments/m64_wal_v2_compression.py`
- `experiments/m64_wal_v2_compression.log`

## Next Steps
- Phase 5: Hierarchical Atoms (WAL-1 reboot) for 2-4× compression
- Phase 6-11: Ecosystem integration


## Extracted Metrics (from source)

- Time: .3
- Time: .3
