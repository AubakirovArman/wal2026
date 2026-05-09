#!/usr/bin/env python3
"""Phase 11 Demo: ONNX export and model merging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import tempfile
from wal.v1.nn import encode_linear_weight, WALLinear
from wal.v1.onnx_export import export_wal_simple, verify_onnx_export
from wal.v1.mergekit import MergeConfig, merge_wal_models

print("=" * 60)
print("Phase 11: Ecosystem")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ONNX export
print("\n[ONNX Export]")
layer = WALLinear(encode_linear_weight(torch.randn(64, 32, device=device), K=32, C=4))
dummy = torch.randn(2, 32, device=device)

with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    filepath = f.name

onnx_bytes = export_wal_simple(layer, dummy, filepath=filepath)
matches = verify_onnx_export(layer, dummy, onnx_bytes)
print(f"  ONNX export verified: {'YES' if matches else 'NO'}")

# Model merging
print("\n[Model Merge]")
from torch import nn

def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        WALLinear(encode_linear_weight(torch.randn(32, 16, device=device), K=16, C=4)),
        WALLinear(encode_linear_weight(torch.randn(8, 32, device=device), K=16, C=4)),
    ).to(device)

model_a = make_model(10)
model_b = make_model(20)
model_c = make_model(30)

config = MergeConfig(method="soup", soup_method="mean")
merged = merge_wal_models([model_a, model_b, model_c], config)

x = torch.randn(4, 16, device=device)
out = merged(x)
print(f"  Soup merge: 3 models → output shape {out.shape}")

import os
os.unlink(filepath)
