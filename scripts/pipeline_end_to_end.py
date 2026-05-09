#!/usr/bin/env python3
"""End-to-end pipeline: create model → encode → decode → verify."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from wal.v1.nn import encode_linear_weight, WALLinear, wal_state_dict, wal_load_state_dict
from wal.v2.decoder import wal_decode_v2

print("=" * 60)
print("End-to-End Pipeline Demo")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Create a tiny model
print("\n[1/4] Creating model...")
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
).to(device)

x = torch.randn(8, 128, device=device)
with torch.no_grad():
    y_dense = model(x)
print(f"  Model: {x.shape} → {y_dense.shape}")

# Step 2: Encode to WAL
print("\n[2/4] Encoding to WAL...")
model_wal = nn.Sequential(
    WALLinear(encode_linear_weight(model[0].weight.data, K=32, C=4), bias=model[0].bias.data),
    nn.ReLU(),
    WALLinear(encode_linear_weight(model[2].weight.data, K=32, C=4), bias=model[2].bias.data),
).to(device)

with torch.no_grad():
    y_wal = model_wal(x)

diff = (y_dense - y_wal).abs().max().item()
print(f"  Max output diff: {diff:.8f}")

# Step 3: Serialize/deserialize
print("\n[3/4] Serializing WAL state...")
state = wal_state_dict(model_wal)
print(f"  State keys: {list(state.keys())}")

# Step 4: Verify decode
print("\n[4/4] Verifying decode...")
for name, module in model_wal.named_modules():
    if isinstance(module, WALLinear):
        decoded = module.wal_weight.decode(device)
        original = module.wal_weight.prog
        print(f"  {name}: shape={decoded.shape}, K={module.wal_weight.atom_table.K0}")

print("\n" + "=" * 60)
print("Pipeline complete!")
print("=" * 60)
