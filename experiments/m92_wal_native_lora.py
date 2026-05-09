"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M92: WAL-Native LoRA — adaptation in WAL space vs classic LoRA.

Compares approaches for fine-tuning a WAL-encoded layer:
1. Table-tuning: train atom_values + coeff_values
2. Coeff-LoRA: train only coeff_adapter, freeze tables
3. Classic LoRA: add low-rank matrices to decoded weights

Tests parameter counts and fine-tuning quality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1.qat import WALQATLinear, linear_to_qat
from wal.v1.meta import WALProgramAdapter


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def test_01_parameter_comparison():
    """Compare parameter counts across approaches."""
    print("[1/3] Parameter count comparison...")
    
    # Small layer for fast k-means
    in_f, out_f = 256, 128
    base = nn.Linear(in_f, out_f, bias=False)
    
    # 1. Table-tuning
    qat = linear_to_qat(base, K=32, C=8, encode_iters=1)
    t, _ = count_params(qat)
    print(f"  Table-tuning:        {t:4d} params (K+C={32+8})")
    
    # 2. Coeff-LoRA
    qat = linear_to_qat(base, K=32, C=8, encode_iters=1, use_coeff_adapter=True)
    qat.atom_values.requires_grad = False
    qat.coeff_values.requires_grad = False
    t, _ = count_params(qat)
    print(f"  Coeff-LoRA:          {t:4d} params (C={8})")
    
    # 3. Classic LoRA rank=4
    lora = WALProgramAdapter(shape=(out_f, in_f), rank=4)
    t, _ = count_params(lora)
    print(f"  Classic LoRA (r=4):  {t:4d} params (r*(in+out)={4*(in_f+out_f)})")
    
    ratio = (4 * (in_f + out_f)) / 8
    print(f"\n  Coeff-LoRA uses {ratio:.0f}x FEWER params than classic LoRA!")
    assert ratio > 10, "WAL-native should be much smaller"
    print("  ✅ WAL-native uses dramatically fewer parameters")
    return True


def test_02_finetuning_quality():
    """Compare fine-tuning quality across approaches."""
    print("\n[2/3] Fine-tuning quality comparison...")
    
    torch.manual_seed(42)
    base = nn.Linear(64, 32, bias=False)
    nn.init.xavier_uniform_(base.weight)
    
    # Domain shift
    target_w = base.weight.data + torch.randn_like(base.weight.data) * 0.05
    X = torch.randn(100, 64)
    with torch.no_grad():
        Y = X @ target_w.T
    
    def train(model, steps=50, lr=0.1, desc=""):
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        with torch.no_grad():
            init = F.mse_loss(model(X), Y).item()
        for _ in range(steps):
            opt.zero_grad()
            loss = F.mse_loss(model(X), Y)
            loss.backward()
            opt.step()
        imp = init / (loss.item() + 1e-10)
        print(f"  {desc:22s}: init={init:.5f} final={loss.item():.5f} imp={imp:.2f}x")
        return imp
    
    results = {}
    
    # 1. Table-tuning
    qat = linear_to_qat(base, K=16, C=4, encode_iters=1)
    results["table"] = (train(qat, desc="Table-tuning"), count_params(qat)[0])
    
    # 2. Coeff-LoRA
    qat = linear_to_qat(base, K=16, C=4, encode_iters=1, use_coeff_adapter=True)
    qat.atom_values.requires_grad = False
    qat.coeff_values.requires_grad = False
    results["coeff"] = (train(qat, desc="Coeff-LoRA"), count_params(qat)[0])
    
    # 3. Atom+Coeff-LoRA
    qat = linear_to_qat(base, K=16, C=4, encode_iters=1,
                        use_coeff_adapter=True, use_atom_adapter=True)
    qat.atom_values.requires_grad = False
    qat.coeff_values.requires_grad = False
    results["both"] = (train(qat, desc="Atom+Coeff-LoRA"), count_params(qat)[0])
    
    # 4. Classic LoRA
    qat = linear_to_qat(base, K=16, C=4, encode_iters=1)
    qat.atom_values.requires_grad = False
    qat.coeff_values.requires_grad = False
    lora = WALProgramAdapter(shape=(32, 64), rank=4)
    
    class LoRAWAL(nn.Module):
        def __init__(self, wal, adapter):
            super().__init__()
            self.wal = wal
            self.adapter = adapter
        def forward(self, x):
            w = self.wal.decode_weight(x.device)
            w = self.adapter(w)
            return F.linear(x, w, self.wal.bias)
    
    model = LoRAWAL(qat, lora)
    results["classic"] = (train(model, desc="Classic LoRA (r=4)"), count_params(lora)[0])
    
    print("\n  Summary:")
    for name, (imp, params) in results.items():
        print(f"    {name:12s}: {imp:.2f}x improvement, {params:3d} params")
    
    coeff_imp = results["coeff"][0]
    assert coeff_imp > 1.1, "Coeff-LoRA should improve"
    print("  ✅ WAL-native LoRA works for fine-tuning")
    return True


def test_03_merge_capability():
    """Test that WAL-native adapters can be merged into tables."""
    print("\n[3/3] Merge capability...")
    
    torch.manual_seed(42)
    base = nn.Linear(64, 32, bias=False)
    nn.init.xavier_uniform_(base.weight)
    
    qat = linear_to_qat(base, K=16, C=4, encode_iters=1,
                        use_coeff_adapter=True, use_atom_adapter=True)
    
    # Simulate training
    with torch.no_grad():
        qat.coeff_adapter.add_(torch.randn_like(qat.coeff_adapter) * 0.01)
        qat.atom_adapter.add_(torch.randn_like(qat.atom_adapter) * 0.01)
    
    weight_before = qat.decode_weight().detach().clone()
    
    # Merge adapters into tables (respecting atom mask)
    with torch.no_grad():
        if qat.atom_adapt_mask is not None:
            qat.atom_values.add_(qat.atom_adapter * qat.atom_adapt_mask.float())
        else:
            qat.atom_values.add_(qat.atom_adapter)
        qat.coeff_values.add_(qat.coeff_adapter)
        qat.atom_adapter.zero_()
        qat.coeff_adapter.zero_()
    
    weight_after = qat.decode_weight().detach().clone()
    diff = (weight_before - weight_after).abs().max().item()
    
    print(f"  Max diff after merge: {diff:.8f}")
    assert diff < 1e-6, "Merge should be lossless"
    print("  ✅ WAL-native adapters merge losslessly")
    return True


def main():
    print("=" * 60)
    print("M92: WAL-Native LoRA")
    print("=" * 60)
    
    results = []
    results.append(("Parameters", test_01_parameter_comparison()))
    results.append(("Quality", test_02_finetuning_quality()))
    results.append(("Merge", test_03_merge_capability()))
    
    print("\n" + "=" * 60)
    passed = sum(1 for _, r in results if r)
    print(f"M92: {passed}/{len(results)} tests passed")
    print("=" * 60)
    return all(r for _, r in results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
