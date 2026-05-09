#!/usr/bin/env python3
"""M182 — Transform-WAL Editability (diff locality with K=256).

Tests whether Hadamard K=256 improves diff locality after LoRA edit.
"""
import torch, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def hadamard_matrix(n):
    if n == 1: return torch.ones(1, 1)
    H = hadamard_matrix(n // 2)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)


def apply_hadamard(W):
    out_d, in_d = W.shape
    op = 1 << max(0, math.ceil(math.log2(out_d))) if out_d > 1 else 1
    ip = 1 << max(0, math.ceil(math.log2(in_d))) if in_d > 1 else 1
    W_pad = torch.zeros(op, ip, dtype=torch.float32, device=W.device)
    W_pad[:out_d, :in_d] = W.float()
    H_out = (hadamard_matrix(op).to(W.device) / math.sqrt(op))
    H_in = (hadamard_matrix(ip).to(W.device) / math.sqrt(ip))
    return H_out @ W_pad @ H_in.T


def inverse_hadamard(W_t, orig_shape):
    out_d, in_d = orig_shape
    # W_t may already be padded from apply_hadamard
    # Just apply inverse directly, then crop to orig_shape
    op, ip = W_t.shape
    H_out = (hadamard_matrix(op).to(W_t.device) / math.sqrt(op))
    H_in = (hadamard_matrix(ip).to(W_t.device) / math.sqrt(ip))
    recon_inv = H_out.T @ W_t.float() @ H_in.float()
    return recon_inv[:out_d, :in_d]


def encode_raw(w, K=256, C=16):
    atoms = build_l0_atoms(w.reshape(-1), K=K, iters=3)
    coeffs = build_coeff_table(w.reshape(-1), atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=262_144)
    return recon.reshape(w.shape), atoms, coeffs


def encode_hadamard(w, K=256, C=16):
    w_h = apply_hadamard(w)
    atoms = build_l0_atoms(w_h.reshape(-1), K=K, iters=3)
    coeffs = build_coeff_table(w_h.reshape(-1), atoms, C=C, iters=3)
    _, recon = wal_encode_v1(w_h.reshape(-1), atoms, coeffs, batch=262_144)
    recon_inv = inverse_hadamard(recon.reshape(w_h.shape), w.shape)
    return recon_inv.to(w.dtype), atoms, coeffs


def apply_lora_edit(w, rank=4, scale=0.1):
    torch.manual_seed(42)
    A = torch.randn(w.shape[0], rank, dtype=w.dtype, device=w.device) * scale
    B = torch.randn(rank, w.shape[1], dtype=w.dtype, device=w.device) * scale
    return w + (A @ B)


def compute_diff(base_prog, edited_prog):
    """Compute program diff fraction."""
    diff_atoms = (base_prog.atom_ids != edited_prog.atom_ids).float().mean().item()
    diff_coeffs = (base_prog.coeff_ids != edited_prog.coeff_ids).float().mean().item()
    return (diff_atoms + diff_coeffs) / 2


def main():
    print("=" * 60)
    print("M182 — Transform-WAL Editability (K=256)")
    print("=" * 60)
    
    device = "cuda:0"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    layer_idx = 0
    modules = ['q_proj', 'v_proj', 'gate_proj']
    K, C = 256, 16
    
    results = []
    
    for m_name in modules:
        mod = getattr(model.model.layers[layer_idx].self_attn if m_name != 'gate_proj' else model.model.layers[layer_idx].mlp, m_name)
        w_base = mod.weight.data
        
        # Apply LoRA edit
        w_edited = apply_lora_edit(w_base, rank=4, scale=0.1)
        
        # Raw-WAL encoding
        raw_base, raw_atoms, raw_coeffs = encode_raw(w_base, K, C)
        raw_edited, _, _ = encode_raw(w_edited, K, C)
        
        # Re-encode with same atoms/coeffs
        _, raw_recon_edited = wal_encode_v1(w_edited.reshape(-1), raw_atoms, raw_coeffs, batch=262_144)
        raw_diff = (raw_base.reshape(-1) != raw_recon_edited).float().mean().item()
        
        # Hadamard-WAL encoding
        had_base, had_atoms, had_coeffs = encode_hadamard(w_base, K, C)
        had_edited, _, _ = encode_hadamard(w_edited, K, C)
        
        # Re-encode with same atoms/coeffs
        w_edited_h = apply_hadamard(w_edited)
        _, had_recon_edited = wal_encode_v1(w_edited_h.reshape(-1), had_atoms, had_coeffs, batch=262_144)
        had_recon_inv = inverse_hadamard(had_recon_edited.reshape(w_edited_h.shape), w_edited.shape)
        had_recon_inv = had_recon_inv.reshape(had_base.shape)
        had_diff = (had_base.reshape(-1) != had_recon_inv.to(had_base.dtype).reshape(-1)).float().mean().item()
        
        # MSE comparison
        raw_mse = ((w_base - raw_base) ** 2).mean().item()
        had_mse = ((w_base - had_base) ** 2).mean().item()
        
        results.append({
            'module': m_name,
            'raw_diff': raw_diff,
            'hadamard_diff': had_diff,
            'raw_mse': raw_mse,
            'hadamard_mse': had_mse,
        })
        
        print(f"\n{m_name}:")
        print(f"  Raw diff:     {raw_diff:.4f}")
        print(f"  Hadamard diff: {had_diff:.4f}")
        print(f"  Raw MSE:      {raw_mse:.2e}")
        print(f"  Hadamard MSE: {had_mse:.2e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Module':>10} {'Raw Diff':>12} {'Had Diff':>12} {'Ratio':>8}")
    print("-" * 50)
    for r in results:
        ratio = r['hadamard_diff'] / max(r['raw_diff'], 1e-10)
        print(f"{r['module']:>10} {r['raw_diff']:>12.4f} {r['hadamard_diff']:>12.4f} {ratio:>8.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m182_transform_wal_editability.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
