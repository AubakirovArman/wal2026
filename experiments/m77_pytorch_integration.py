#!/usr/bin/env python3
"""M77: WAL v1 PyTorch Integration Test (Phase 6).

Tests:
1. WALParameter encode/decode
2. WALLinear forward pass matches nn.Linear
3. WALCachedLinear forward pass matches nn.Linear
4. replace_linear_with_wal on a simple model
5. Output equivalence between dense and WAL models
"""
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from wal.v1 import (
    encode_linear_weight, WALParameter, WALLinear, WALCachedLinear,
    replace_linear_with_wal,
)


def test_wal_parameter():
    """Test WALParameter encode/decode."""
    print("=" * 60)
    print("TEST 1: WALParameter Encode/Decode")
    print("=" * 60)
    
    torch.manual_seed(42)
    weight = torch.randn(128, 64, dtype=torch.float32) * 0.1
    
    # Encode
    wal_param = encode_linear_weight(weight, K=32, C=8)
    
    print(f"  Original shape: {weight.shape}")
    print(f"  WAL: {wal_param}")
    
    # Decode
    decoded = wal_param.decode()
    
    max_diff = (weight - decoded).abs().max().item()
    mean_diff = (weight - decoded).abs().mean().item()
    
    print(f"  Decode max diff: {max_diff:.8f}")
    print(f"  Decode mean diff: {mean_diff:.8f}")
    
    assert max_diff < 0.1, f"Decode too inaccurate: {max_diff}"
    assert decoded.shape == weight.shape, "Shape mismatch"
    assert decoded.dtype == weight.dtype, "Dtype mismatch"
    
    # Cache test
    decoded2 = wal_param.decode()
    assert decoded2 is decoded, "Cache not reused"
    
    wal_param.clear_cache()
    decoded3 = wal_param.decode()
    assert decoded3 is not decoded, "Cache not cleared"
    
    print("  ✅ PASS")
    return True


def test_wal_linear_forward():
    """Test WALLinear forward matches nn.Linear."""
    print()
    print("=" * 60)
    print("TEST 2: WALLinear Forward Pass")
    print("=" * 60)
    
    torch.manual_seed(123)
    in_features = 64
    out_features = 32
    batch_size = 16
    
    # Create reference linear
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.eval()
    
    # Encode weight to WAL
    wal_param = encode_linear_weight(linear.weight.data, K=64, C=16)
    wal_linear = WALLinear(wal_param, bias=linear.bias.data if linear.bias is not None else None)
    wal_linear.eval()
    
    # Test input
    x = torch.randn(batch_size, in_features)
    
    with torch.no_grad():
        out_ref = linear(x)
        out_wal = wal_linear(x)
    
    max_diff = (out_ref - out_wal).abs().max().item()
    mean_diff = (out_ref - out_wal).abs().mean().item()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_ref.shape}")
    print(f"  Max output diff: {max_diff:.8f}")
    print(f"  Mean output diff: {mean_diff:.8f}")
    
    assert max_diff < 0.1, f"WALLinear forward too inaccurate: {max_diff}"
    
    print("  ✅ PASS")
    return True


def test_wal_cached_linear():
    """Test WALCachedLinear forward matches nn.Linear."""
    print()
    print("=" * 60)
    print("TEST 3: WALCachedLinear Forward Pass")
    print("=" * 60)
    
    torch.manual_seed(456)
    in_features = 32
    out_features = 16
    
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.eval()
    
    wal_param = encode_linear_weight(linear.weight.data, K=32, C=8)
    wal_cached = WALCachedLinear(wal_param)
    wal_cached.eval()
    
    x = torch.randn(8, in_features)
    
    with torch.no_grad():
        out_ref = linear(x)
        out_cached = wal_cached(x)
    
    max_diff = (out_ref - out_cached).abs().max().item()
    
    print(f"  Max output diff: {max_diff:.8f}")
    assert max_diff < 0.1, f"WALCachedLinear forward too inaccurate: {max_diff}"
    
    print("  ✅ PASS")
    return True


def test_replace_linear():
    """Test replace_linear_with_wal on a simple model."""
    print()
    print("=" * 60)
    print("TEST 4: replace_linear_with_wal")
    print("=" * 60)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 32)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    torch.manual_seed(789)
    model = SimpleModel()
    model.eval()
    
    # Clone for reference
    model_ref = SimpleModel()
    model_ref.load_state_dict(model.state_dict())
    model_ref.eval()
    
    # Replace with WAL
    replace_linear_with_wal(model, K=64, C=16, cached=False)
    model.eval()
    
    print(f"  fc1 type: {type(model.fc1).__name__}")
    print(f"  fc2 type: {type(model.fc2).__name__}")
    
    x = torch.randn(4, 64)
    
    with torch.no_grad():
        out_ref = model_ref(x)
        out_wal = model(x)
    
    max_diff = (out_ref - out_wal).abs().max().item()
    mean_diff = (out_ref - out_wal).abs().mean().item()
    
    print(f"  Max output diff: {max_diff:.8f}")
    print(f"  Mean output diff: {mean_diff:.8f}")
    
    assert max_diff < 0.5, f"Model replacement too inaccurate: {max_diff}"
    
    print("  ✅ PASS")
    return True


def test_device_transfer():
    """Test WAL decode on different devices."""
    print()
    print("=" * 60)
    print("TEST 5: Device Transfer")
    print("=" * 60)
    
    torch.manual_seed(321)
    weight = torch.randn(32, 16, dtype=torch.float32)
    wal_param = encode_linear_weight(weight, K=16, C=4)
    
    # Decode on CPU
    decoded_cpu = wal_param.decode(torch.device('cpu'))
    assert decoded_cpu.device.type == 'cpu', "CPU decode failed"
    
    # Clear cache and decode on CUDA if available
    wal_param.clear_cache()
    if torch.cuda.is_available():
        decoded_cuda = wal_param.decode(torch.device('cuda:0'))
        assert decoded_cuda.device.type == 'cuda', "CUDA decode failed"
        print(f"  CPU decode: OK")
        print(f"  CUDA decode: OK")
    else:
        print(f"  CPU decode: OK")
        print(f"  CUDA: not available")
    
    print("  ✅ PASS")
    return True


def main():
    print("\n" + "=" * 60)
    print("M77: WAL v1 PyTorch Integration Test (Phase 6)")
    print("=" * 60 + "\n")
    
    results = []
    
    tests = [
        ("WALParameter", test_wal_parameter),
        ("WALLinear Forward", test_wal_linear_forward),
        ("WALCachedLinear", test_wal_cached_linear),
        ("Replace Linear", test_replace_linear),
        ("Device Transfer", test_device_transfer),
    ]
    
    for name, test_fn in tests:
        try:
            results.append((name, test_fn()))
        except Exception as e:
            import traceback
            print(f"  ❌ FAIL: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
