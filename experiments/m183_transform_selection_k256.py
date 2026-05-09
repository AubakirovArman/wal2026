"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M183 — Transform Selection per Module (K=256).

Confirms single transform rule on higher K.
"""
import torch, json, sys, math
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
    op, ip = W_t.shape
    H_out = (hadamard_matrix(op).to(W_t.device) / math.sqrt(op))
    H_in = (hadamard_matrix(ip).to(W_t.device) / math.sqrt(ip))
    recon_inv = H_out.T @ W_t.float() @ H_in.float()
    return recon_inv[:out_d, :in_d]


def encode_hadamard(w, K=256, C=16):
    w_h = apply_hadamard(w)
    atoms = build_l0_atoms(w_h.reshape(-1), K=K, iters=1)
    coeffs = build_coeff_table(w_h.reshape(-1), atoms, C=C, iters=1)
    _, recon = wal_encode_v1(w_h.reshape(-1), atoms, coeffs, batch=65_536)
    recon_inv = inverse_hadamard(recon.reshape(w_h.shape), w.shape)
    return recon_inv.to(w.dtype), atoms, coeffs


def main():
    print("=" * 60)
    print("M183 — Transform Selection per Module (K=256)")
    print("=" * 60)
    
    print("\nLoading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    layers = [0, 16]
    modules = ['q_proj', 'v_proj', 'gate_proj']
    K, C = 256, 16
    
    # Collect weights
    weights = {}
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn if m != 'gate_proj' else model.model.layers[li].mlp, m).weight.data
            weights[f"{li}_{m}"] = w
    
    # Single Hadamard for all
    print("\n--- Single Hadamard (all modules) ---")
    all_h = torch.cat([apply_hadamard(w).reshape(-1) for w in weights.values()])
    atoms_single = build_l0_atoms(all_h, K=K, iters=1)
    coeffs_single = build_coeff_table(all_h, atoms_single, C=C, iters=1)
    
    # Module-specific Hadamard
    print("--- Module-specific Hadamard ---")
    atoms_mod = {}
    for key, w in weights.items():
        w_h = apply_hadamard(w)
        a = build_l0_atoms(w_h.reshape(-1), K=K, iters=1)
        c = build_coeff_table(w_h.reshape(-1), a, C=C, iters=1)
        atoms_mod[key] = (a, c)
    
    # Evaluate
    results = []
    for key, w in weights.items():
        w_h = apply_hadamard(w)
        
        # Single
        _, recon_single = wal_encode_v1(w_h.reshape(-1), atoms_single, coeffs_single, batch=65_536)
        mse_single = ((w_h.reshape(-1) - recon_single) ** 2).mean().item()
        recon_inv_single = inverse_hadamard(recon_single.reshape(w_h.shape), w.shape)
        
        # Specific
        _, recon_specific = wal_encode_v1(w_h.reshape(-1), *atoms_mod[key], batch=65_536)
        mse_specific = ((w_h.reshape(-1) - recon_specific) ** 2).mean().item()
        recon_inv_specific = inverse_hadamard(recon_specific.reshape(w_h.shape), w.shape)
        
        results.append({
            'key': key,
            'mse_single': mse_single,
            'mse_specific': mse_specific,
            'ratio': mse_single / max(mse_specific, 1e-15),
        })
        
        print(f"  {key}: single={mse_single:.2e}, specific={mse_specific:.2e}, ratio={mse_single/max(mse_specific,1e-15):.2f}")
    
    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    print(f"\nAvg single/specific ratio: {avg_ratio:.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m183_transform_selection_k256.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
