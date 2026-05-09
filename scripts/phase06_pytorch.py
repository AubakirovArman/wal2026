#!/usr/bin/env python3
"""Phase 6 Demo: PyTorch integration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from wal.v1.nn import encode_linear_weight, WALLinear, WALCachedLinear

print("=" * 60)
print("Phase 6: PyTorch Integration")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original linear layer
linear = nn.Linear(1024, 512, bias=True).to(device)
x = torch.randn(4, 1024, device=device)

# Encode to WAL
print("Encoding nn.Linear to WAL...")
wal_param = encode_linear_weight(linear.weight.data, K=64, C=8)
wal_linear = WALCachedLinear(wal_param, bias=linear.bias.data).to(device)

# Compare outputs
with torch.no_grad():
    out_dense = linear(x)
    out_wal = wal_linear(x)

diff = (out_dense - out_wal).abs().max().item()
print(f"  Max output diff: {diff:.8f}")
print(f"  WAL layer type: {type(wal_linear).__name__}")
print(f"  Device: {device}")
