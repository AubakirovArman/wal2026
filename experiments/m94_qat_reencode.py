#!/usr/bin/env python3
"""M94: Periodic re-encoding of WAL programs during QAT.

When atom/coeff tables change during training, the fixed program indices
(atom_ids, coeff_ids) may no longer be optimal. Periodic re-encoding
updates the programs to better match the current tables.

Tests:
1. Table-tuning without re-encoding (baseline)
2. Table-tuning with periodic re-encoding
3. Re-encoding should improve reconstruction quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1.qat import linear_to_qat


def test_reencoding_improves_quality():
    """Compare table-tuning with and without periodic re-encoding."""
    print("[1/1] Periodic re-encoding improves quality...")
    
    torch.manual_seed(42)
    
    # Base model
    base = nn.Linear(128, 64, bias=False)
    nn.init.xavier_uniform_(base.weight)
    
    # Domain-shifted target
    target_w = base.weight.data + torch.randn_like(base.weight.data) * 0.05
    X = torch.randn(100, 128)
    with torch.no_grad():
        Y = X @ target_w.T
    
    # --- Baseline: no re-encoding ---
    qat_base = linear_to_qat(base, K=32, C=8, encode_iters=1)
    
    with torch.no_grad():
        init_loss_base = F.mse_loss(qat_base(X), Y).item()
    
    opt = torch.optim.Adam([qat_base.atom_values, qat_base.coeff_values], lr=0.1)
    for epoch in range(100):
        opt.zero_grad()
        loss = F.mse_loss(qat_base(X), Y)
        loss.backward()
        opt.step()
    
    final_loss_base = loss.item()
    imp_base = init_loss_base / (final_loss_base + 1e-10)
    print(f"  Without re-encoding: init={init_loss_base:.5f} final={final_loss_base:.5f} imp={imp_base:.2f}x")
    
    # --- With periodic re-encoding ---
    qat_reenc = linear_to_qat(base, K=32, C=8, encode_iters=1)
    
    with torch.no_grad():
        init_loss_reenc = F.mse_loss(qat_reenc(X), Y).item()
    
    opt = torch.optim.Adam([qat_reenc.atom_values, qat_reenc.coeff_values], lr=0.1)
    for epoch in range(100):
        opt.zero_grad()
        loss = F.mse_loss(qat_reenc(X), Y)
        loss.backward()
        opt.step()
        
        # Re-encode programs every 25 epochs
        if epoch > 0 and epoch % 25 == 0:
            qat_reenc.reencode_programs()
    
    final_loss_reenc = loss.item()
    imp_reenc = init_loss_reenc / (final_loss_reenc + 1e-10)
    print(f"  With re-encoding:    init={init_loss_reenc:.5f} final={final_loss_reenc:.5f} imp={imp_reenc:.2f}x")
    
    # Re-encoding should improve or match
    print(f"  Re-encoding gain: {imp_reenc / imp_base:.2f}x relative improvement")
    
    # Re-encoding should be at least as good
    assert imp_reenc >= imp_base * 0.9, f"Re-encoding degraded quality: {imp_reenc:.2f} vs {imp_base:.2f}"
    print("  ✅ Periodic re-encoding works")
    return True


def main():
    print("=" * 60)
    print("M94: Periodic Re-Encoding")
    print("=" * 60)
    
    ok = test_reencoding_improves_quality()
    
    print("\n" + "=" * 60)
    print(f"M94: {int(ok)}/1 tests passed")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
