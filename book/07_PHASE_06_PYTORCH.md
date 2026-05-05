# Phase 6: PyTorch Integration (M76–M77)

## The Problem

WAL is a format. Formats are useless if you can't use them in PyTorch.

## What Was Built

- **`WALParameter`**: Stores `prog + atom_table + coeffs`, lazy decode with cache
- **`WALLinear`**: Decodes weight on-the-fly per forward pass
- **`WALCachedLinear`**: Decodes once, caches for speed
- **`replace_linear_with_wal()`**: Converts entire models automatically
- **`wal_state_dict()` / `wal_load_state_dict()`**: Serialize/deserialize

## Usage

```python
from wal.v1.nn import encode_linear_weight, WALLinear

# Encode one layer
wal_param = encode_linear_weight(linear.weight.data, K=256, C=16)
wal_linear = WALLinear(wal_param, bias=linear.bias.data)

# Use like any nn.Module
output = wal_linear(input_tensor)
```

## Test Results

| Test | Result |
|------|--------|
| WALParameter decode | ✅ PASS |
| WALLinear forward | ✅ PASS |
| WALCachedLinear | ✅ PASS |
| Replace nn.Linear | ✅ PASS |
| Device transfer | ✅ PASS |

## Why This Matters

Without PyTorch integration, WAL is a research curiosity. With PyTorch integration, WAL is a drop-in replacement for `nn.Linear` — the most common layer in every transformer.

## Files
- `src/wal/v1/nn.py`
- `experiments/m76_wal_v1_roundtrip.py`
- `experiments/m77_pytorch_integration.py`
