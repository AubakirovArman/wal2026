#!/usr/bin/env python3
"""M82: Adapter Integration with WAL Layers.

Test WALProgramAdapter attached to WALCachedLinear.
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import torch.nn as nn
from wal.v1.nn import encode_linear_weight, WALCachedLinear, WALLinear
from wal.v1.meta import WALProgramAdapter

print("=" * 60)
print("M82: Adapter Integration")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(82)

# Encode a weight
weight = torch.randn(64, 32, device=device)
wal_param = encode_linear_weight(weight, K=64, C=8)

# Test 1: WALCachedLinear with adapter
print("\n[1/3] WALCachedLinear + Adapter")
layer = WALCachedLinear(wal_param, bias=torch.randn(64, device=device))
layer.to(device)

# Without adapter
x = torch.randn(4, 32, device=device)
out_base = layer(x)
assert out_base.shape == (4, 64)

# Attach adapter
adapter = WALProgramAdapter(shape=(64, 32), rank=4, alpha=1.0).to(device)
with torch.no_grad():
    adapter.lora_A.normal_(0, 0.1)
    adapter.lora_B.normal_(0, 0.1)
layer.set_adapter(adapter)

out_adapted = layer(x)
assert out_adapted.shape == (4, 64)
assert not torch.allclose(out_base, out_adapted), "Adapter had no effect"

print(f"  ✓ Adapter changes output: max diff = {(out_adapted - out_base).abs().max().item():.6f}")

# Test 2: Detach adapter
print("\n[2/3] Detach Adapter")
layer.set_adapter(None)
out_detached = layer(x)
assert torch.allclose(out_base, out_detached), "Detach failed"
print(f"  ✓ Detach restores base output")

# Test 3: Trainable parameters
print("\n[3/3] Trainable Parameters")
layer.set_adapter(adapter)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
# Only adapter parameters are trainable; WAL weight is frozen
assert trainable == adapter.lora_A.numel() + adapter.lora_B.numel(), \
    f"Expected {adapter.lora_A.numel() + adapter.lora_B.numel()} trainable params, got {trainable}"
print(f"  ✓ Trainable params: {trainable} (adapter only)")

# Test 4: Gradient flow
print("\n[4/4] Gradient Flow")
layer.set_adapter(adapter)
out = layer(x)
loss = out.sum()
loss.backward()

assert adapter.lora_A.grad is not None, "No gradient on lora_A"
assert adapter.lora_B.grad is not None, "No gradient on lora_B"
print(f"  ✓ Gradients flow through adapter")

print("\n" + "=" * 60)
print("M82: ALL 4/4 TESTS PASS")
print("=" * 60)
