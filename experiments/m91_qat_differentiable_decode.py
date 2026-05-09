"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M91: Differentiable WAL decode layer (WALQATLinear).

Tests:
1. WALQATLinear can be created from nn.Linear
2. Forward pass matches WAL decode (within tolerance)
3. Gradients flow to atom_values and coeff_values
4. Gradients do NOT flow to program indices (atom_ids, coeff_ids)
5. Table-tuning reduces MSE on synthetic data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1.qat import WALQATLinear, linear_to_qat


def test_01_creation():
    """Test: WALQATLinear can be created from nn.Linear."""
    print("[1/5] Creation from nn.Linear...")
    
    linear = nn.Linear(512, 256, bias=True)
    torch.nn.init.xavier_uniform_(linear.weight)
    
    qat = linear_to_qat(linear, K=64, C=8, encode_iters=2)
    
    # Check types
    assert isinstance(qat.atom_values, nn.Parameter)
    assert isinstance(qat.coeff_values, nn.Parameter)
    assert not isinstance(qat.atom_ids, nn.Parameter)
    assert not isinstance(qat.coeff_ids, nn.Parameter)
    
    # Check shapes
    assert qat.atom_values.shape == (64,)
    assert qat.coeff_values.shape == (8,)
    assert qat.atom_ids.shape == (256 * 512,)
    assert qat.weight_shape == (256, 512)
    
    print(f"  K={qat.atom_values.numel()}, C={qat.coeff_values.numel()}")
    print(f"  Programs: {qat.atom_ids.numel()} indices")
    print("  ✅ Creation works")
    return True


def test_02_forward_match():
    """Test: Forward pass matches WAL decode."""
    print("\n[2/5] Forward pass matches WAL decode...")
    
    torch.manual_seed(42)
    linear = nn.Linear(128, 64, bias=False)
    torch.nn.init.normal_(linear.weight, 0, 0.02)
    
    qat = linear_to_qat(linear, K=32, C=4, encode_iters=3)
    
    # Create reference: use QAT's own atom/coeff values with v2 decoder
    from wal.v2.decoder import wal_decode_v2
    from wal.v2.isa import AtomTable, CoeffTable, ProgramBufferV2
    
    # Build ProgramBufferV2 from QAT's buffers
    prog = ProgramBufferV2(
        atom_ids=qat.atom_ids,
        coeff_ids=qat.coeff_ids,
        residuals=qat.residuals.float(),
        has_residual=qat.has_residual,
        shape=qat.weight_shape,
    )
    
    # Decode using QAT's current tables (detached for reference)
    ref_weight = wal_decode_v2(
        prog,
        AtomTable(values=qat.atom_values.detach()),
        CoeffTable(values=qat.coeff_values.detach()),
    )
    ref_weight = ref_weight.reshape(qat.weight_shape)
    
    # Test input
    x = torch.randn(4, 128)
    
    # Forward through QAT layer
    qat_out = qat(x)
    
    # Forward through reference (manual linear)
    ref_out = F.linear(x, ref_weight, qat.bias)
    
    # Compare
    diff = (qat_out - ref_out).abs().max().item()
    rel_diff = diff / (ref_out.abs().mean().item() + 1e-8)
    
    print(f"  Max absolute diff: {diff:.6f}")
    print(f"  Relative diff: {rel_diff:.6f}")
    
    # Should be bit-exact (same decode logic)
    assert diff < 1e-5, f"Forward mismatch: {diff}"
    print("  ✅ Forward pass matches")
    return True


def test_03_gradient_flow():
    """Test: Gradients flow to tables but not to program indices."""
    print("\n[3/5] Gradient flow to tables...")
    
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=False)
    torch.nn.init.normal_(linear.weight, 0, 0.02)
    
    qat = linear_to_qat(linear, K=16, C=4, encode_iters=2)
    
    # Synthetic target
    x = torch.randn(2, 64)
    target = torch.randn(2, 32)
    
    # Forward + backward
    qat.zero_grad()
    out = qat(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    
    # Check gradients on tables
    assert qat.atom_values.grad is not None, "atom_values has no gradient!"
    assert qat.coeff_values.grad is not None, "coeff_values has no gradient!"
    
    atom_grad_norm = qat.atom_values.grad.norm().item()
    coeff_grad_norm = qat.coeff_values.grad.norm().item()
    
    print(f"  atom_values grad norm: {atom_grad_norm:.6f}")
    print(f"  coeff_values grad norm: {coeff_grad_norm:.6f}")
    
    assert atom_grad_norm > 1e-10, "atom_values gradient too small"
    assert coeff_grad_norm > 1e-10, "coeff_values gradient too small"
    
    # Program indices should NOT have gradients (they are buffers)
    assert qat.atom_ids.grad is None, "atom_ids should not have gradient!"
    assert qat.coeff_ids.grad is None, "coeff_ids should not have gradient!"
    
    print("  ✅ Gradients flow to tables, not to programs")
    return True


def test_04_table_tuning():
    """Test: Table-tuning reduces MSE."""
    print("\n[4/5] Table-tuning reduces MSE...")
    
    torch.manual_seed(42)
    
    # Create a target weight
    target_linear = nn.Linear(128, 64, bias=False)
    torch.nn.init.xavier_uniform_(target_linear.weight)
    
    # Encode it with smaller K/C to leave more room for improvement
    qat = linear_to_qat(target_linear, K=16, C=4, encode_iters=2)
    
    # Measure initial MSE vs target
    with torch.no_grad():
        initial_weight = qat.decode_weight()
        initial_mse = F.mse_loss(initial_weight, target_linear.weight.data).item()
    
    print(f"  Initial MSE: {initial_mse:.6f}")
    
    # Optimize atom/coeff tables to better reconstruct target
    optimizer = torch.optim.Adam([qat.atom_values, qat.coeff_values], lr=0.05)
    
    for step in range(200):
        optimizer.zero_grad()
        weight = qat.decode_weight()
        loss = F.mse_loss(weight, target_linear.weight.data)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"    Step {step}: MSE = {loss.item():.8f}")
    
    final_mse = loss.item()
    print(f"  Final MSE: {final_mse:.8f}")
    
    improvement = initial_mse / (final_mse + 1e-10)
    print(f"  Improvement: {improvement:.2f}×")
    
    assert final_mse < initial_mse, "MSE did not improve!"
    assert improvement > 1.2, f"Improvement too small: {improvement:.2f}×"
    
    print("  ✅ Table-tuning reduces MSE")
    return True


def test_05_end_to_end_training():
    """Test: Fine-tuning scenario — WAL-native adaptation.
    
    Simulates real use-case: pre-trained model is WAL-encoded, then
    fine-tuned on downstream task. Base weights perturbed (domain shift),
    QAT layer learns to compensate by tuning atom/coeff tables.
    
    This is WAL's answer to LoRA: instead of adding B*A matrices,
    we learn delta-atoms + delta-coeffs (K+C params vs rank*(in+out)).
    For K=32, C=8: 40 params vs 4*(64+32)=384 for LoRA rank=4.
    """
    print("\n[5/5] End-to-end training (fine-tuning scenario)...")
    
    torch.manual_seed(42)
    
    # Base model (pre-trained)
    base_linear = nn.Linear(64, 32, bias=False)
    torch.nn.init.xavier_uniform_(base_linear.weight)
    
    # Encode base model to WAL with smaller K/C to leave room for tuning
    qat = linear_to_qat(base_linear, K=16, C=4, encode_iters=2)
    
    # Simulate domain shift: target = base + perturbation
    target_w = base_linear.weight.data + torch.randn_like(base_linear.weight.data) * 0.03
    
    # Generate data with SHIFTED weights (downstream task)
    X = torch.randn(100, 64)
    with torch.no_grad():
        Y = X @ target_w.T
    
    # Initial loss (WAL-encoded base vs shifted target)
    with torch.no_grad():
        initial_loss = F.mse_loss(qat(X), Y).item()
    
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Trainable params: {qat.atom_values.numel() + qat.coeff_values.numel()}")
    
    # Fine-tune: learn atom/coeff offsets
    optimizer = torch.optim.Adam([qat.atom_values, qat.coeff_values], lr=0.1)
    
    for epoch in range(200):
        optimizer.zero_grad()
        pred = qat(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 30 == 0:
            print(f"    Epoch {epoch}: loss = {loss.item():.8f}")
    
    final_loss = loss.item()
    improvement = initial_loss / (final_loss + 1e-10)
    print(f"  Final loss: {final_loss:.8f}")
    print(f"  Improvement: {improvement:.2f}×")
    
    assert improvement > 1.3, f"Improvement too small: {improvement:.2f}×"
    
    print("  ✅ Fine-tuning via table-tuning works")
    return True


def main():
    print("=" * 60)
    print("M91: Differentiable WAL Decode Layer (WALQATLinear)")
    print("=" * 60)
    
    results = []
    results.append(("Creation", test_01_creation()))
    results.append(("Forward match", test_02_forward_match()))
    results.append(("Gradient flow", test_03_gradient_flow()))
    results.append(("Table tuning", test_04_table_tuning()))
    results.append(("End-to-end training", test_05_end_to_end_training()))
    
    print("\n" + "=" * 60)
    passed = sum(1 for _, r in results if r)
    print(f"M91: {passed}/{len(results)} tests passed")
    print("=" * 60)
    
    return all(r for _, r in results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
