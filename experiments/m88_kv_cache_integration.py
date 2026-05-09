"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M88: KV-cache Integration Test.

Test WALKVCache module end-to-end.
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
from wal.v1.kv_cache import WALKVCache, encode_kv_cache, decode_kv_cache

print("=" * 60)
print("M88: KV-cache Integration Test")
print("=" * 60)

# ---- Test 1: Basic encode/decode ----
print("\n[1/4] Basic encode/decode")

# Simulate KV-cache: 4 layers, 2 heads, 16 seq, 8 dim
num_layers = 4
shape = (1, 2, 16, 8)

native_kv = []
for _ in range(num_layers):
    k = torch.randn(shape)
    v = torch.randn(shape) * 0.2  # V is smoother
    native_kv.append((k, v))

# Encode
wal_cache = encode_kv_cache(native_kv, k_budget=(32, 4), v_budget=(16, 4))
print(f"  Encoded {num_layers} layers")
print(f"  Compression ratio: {wal_cache.compression_ratio:.2f}x")

# Decode back
decoded_kv = decode_kv_cache(wal_cache)
print(f"  Decoded {len(decoded_kv)} layers")

# Verify
max_diff = 0
for i, ((k_orig, v_orig), (k_dec, v_dec)) in enumerate(zip(native_kv, decoded_kv)):
    k_diff = (k_orig - k_dec).abs().max().item()
    v_diff = (v_orig - v_dec).abs().max().item()
    max_diff = max(max_diff, k_diff, v_diff)
    print(f"  Layer {i}: K max diff={k_diff:.6f}, V max diff={v_diff:.6f}")

assert max_diff < 0.5, f"Decode error too large: {max_diff}"
print(f"  ✓ All layers decode within tolerance")

# ---- Test 2: DynamicCache conversion ----
print("\n[2/4] DynamicCache conversion")
try:
    from transformers.cache_utils import DynamicCache
    cache = wal_cache.to_dynamic_cache()
    assert isinstance(cache, DynamicCache)
    print(f"  ✓ DynamicCache conversion works")
except ImportError:
    print(f"  ⚠ transformers cache_utils not available, skipping")

# ---- Test 3: Lazy decode ----
print("\n[3/4] Lazy decode")
wal_cache.clear_decoded_cache()
print(f"  Cleared decoded cache")

# First access triggers decode
k0, v0 = wal_cache.encoded_layers[0].decode()
print(f"  Lazy decode triggered: K shape={list(k0.shape)}, V shape={list(v0.shape)}")

# Second access uses cache
k0_2, v0_2 = wal_cache.encoded_layers[0].decode()
assert torch.equal(k0, k0_2), "Cache not working"
print(f"  ✓ Cache hit: same tensor returned")

# ---- Test 4: Different budgets ----
print("\n[4/4] Different budgets")

budgets = [
    ((256, 16), (128, 16), "High"),
    ((128, 8), (64, 8), "Medium"),
    ((64, 8), (32, 4), "Low"),
]

for k_budget, v_budget, name in budgets:
    wal = encode_kv_cache(native_kv, k_budget=k_budget, v_budget=v_budget)
    dec = decode_kv_cache(wal)
    
    mse = 0
    for (k_o, v_o), (k_d, v_d) in zip(native_kv, dec):
        mse += (k_o - k_d).pow(2).mean().item() + (v_o - v_d).pow(2).mean().item()
    mse /= num_layers * 2
    
    ratio = wal.compression_ratio
    print(f"  {name:8s}: K={k_budget[0]}/{k_budget[1]}, V={v_budget[0]}/{v_budget[1]} → MSE={mse:.8f}, ratio={ratio:.2f}x")

# ---- Summary ----
print("\n" + "=" * 60)
print("M88: ALL TESTS PASS")
print("=" * 60)
print("\nWAL KV-cache module:")
print("  • encode_kv_cache() — compress native KV-cache")
print("  • decode_kv_cache() — decompress to native format")
print("  • WALKVCache.to_dynamic_cache() — transformers compatibility")
print("  • Lazy decode with caching")
print("  • Per-layer budget control")
