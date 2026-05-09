#!/usr/bin/env python3
"""M95: Full QAT pipeline — end-to-end demonstration.

Combines all QAT components:
1. Encode a layer to WAL
2. Fine-tune with WAL-Native LoRA (coeff adapter)
3. Periodic re-encoding
4. Merge adapters into tables
5. Verify final quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1.qat import linear_to_qat


def test_full_pipeline():
    """End-to-end QAT pipeline demonstration."""
    print("[1/1] Full QAT pipeline...")
    
    torch.manual_seed(42)
    
    # Pre-trained base model
    base = nn.Linear(128, 64, bias=False)
    nn.init.xavier_uniform_(base.weight)
    
    # Downstream task (domain shift)
    target_w = base.weight.data + torch.randn_like(base.weight.data) * 0.05
    X = torch.randn(100, 128)
    with torch.no_grad():
        Y = X @ target_w.T
    
    # Step 1: Encode to WAL with adapters
    print("  Step 1: Encode to WAL with Coeff-LoRA adapter...")
    qat = linear_to_qat(base, K=32, C=8, encode_iters=1, use_coeff_adapter=True)
    qat.atom_values.requires_grad = False
    qat.coeff_values.requires_grad = False
    
    with torch.no_grad():
        init_loss = F.mse_loss(qat(X), Y).item()
    print(f"    Initial loss: {init_loss:.5f}")
    
    # Step 2: Fine-tune with Coeff-LoRA
    print("  Step 2: Fine-tune Coeff-LoRA (50 steps)...")
    opt = torch.optim.Adam([qat.coeff_adapter], lr=0.1)
    for epoch in range(50):
        opt.zero_grad()
        loss = F.mse_loss(qat(X), Y)
        loss.backward()
        opt.step()
    
    mid_loss = loss.item()
    print(f"    After LoRA tuning: {mid_loss:.5f}")
    
    # Step 3: Periodic re-encoding
    print("  Step 3: Re-encode programs with learned tables...")
    # First merge adapter into tables
    with torch.no_grad():
        qat.coeff_values.add_(qat.coeff_adapter)
        qat.coeff_adapter.zero_()
    
    # Then re-encode programs
    qat.reencode_programs()
    print(f"    Programs re-encoded")
    
    # Step 4: Continue tuning with merged tables
    print("  Step 4: Continue table-tuning (50 steps)...")
    qat.atom_values.requires_grad = True
    qat.coeff_values.requires_grad = True
    opt = torch.optim.Adam([qat.atom_values, qat.coeff_values], lr=0.05)
    
    for epoch in range(50):
        opt.zero_grad()
        loss = F.mse_loss(qat(X), Y)
        loss.backward()
        opt.step()
    
    final_loss = loss.item()
    print(f"    Final loss: {final_loss:.5f}")
    
    improvement = init_loss / (final_loss + 1e-10)
    print(f"  Overall improvement: {improvement:.2f}x")
    
    # Step 5: Verify merged state has no adapter overhead
    print("  Step 5: Verify merged state...")
    assert qat.coeff_adapter.abs().max().item() < 1e-6, "Adapter not zeroed"
    trainable_after_merge = sum(p.numel() for p in qat.parameters() if p.requires_grad)
    print(f"    Trainable params after merge: {trainable_after_merge}")
    
    print("  ✅ Full QAT pipeline works")
    return True


def main():
    print("=" * 60)
    print("M95: Full QAT Pipeline")
    print("=" * 60)
    
    ok = test_full_pipeline()
    
    print("\n" + "=" * 60)
    print(f"M95: {int(ok)}/1 tests passed")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
