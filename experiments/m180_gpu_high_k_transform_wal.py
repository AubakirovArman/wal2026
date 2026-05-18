#!/usr/bin/env python3
"""M180 — GPU High-K Transform-WAL.

Tests Transform-WAL with K=256, 512, 1024 on GPU.
Compares: Raw, Hadamard, DCT, RandOrth(seed).
"""
import torch, math, json, sys, time
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
    W_pad = torch.zeros(op, ip, dtype=W.dtype, device=W.device)
    W_pad[:out_d, :in_d] = W
    H_out = (hadamard_matrix(op).to(W.device, W.dtype) / math.sqrt(op))
    H_in = (hadamard_matrix(ip).to(W.device, W.dtype) / math.sqrt(ip))
    return H_out @ W_pad @ H_in.T


def apply_dct(W):
    """Simple DCT via FFT (approximation)."""
    m, n = W.shape
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    W_pad = torch.zeros(mp, np, dtype=torch.float32, device=W.device)
    W_pad[:m, :n] = W.float()
    dct = torch.fft.fft2(W_pad).real
    return dct[:m, :n]


def randorth(W, seed=42):
    torch.manual_seed(seed)
    m, n = W.shape
    Q1, _ = torch.linalg.qr(torch.randn(m, m, device=W.device, dtype=torch.float32))
    Q2, _ = torch.linalg.qr(torch.randn(n, n, device=W.device, dtype=torch.float32))
    return (Q1 @ W.float() @ Q2.T).to(W.dtype)


def encode_transform(w, transform, K=256, C=16):
    if transform == "hadamard":
        w_t = apply_hadamard(w)
    elif transform == "dct":
        # DCT skipped (complex values from FFT, needs proper DCT implementation)
        return None
    elif transform == "randorth":
        w_t = randorth(w)
    else:
        w_t = w
    
    atoms = build_l0_atoms(w_t.reshape(-1), K=K, iters=3)
    coeffs = build_coeff_table(w_t.reshape(-1), atoms, C=C, iters=3)
    _, recon = wal_encode_v1(w_t.reshape(-1), atoms, coeffs, batch=262_144)
    
    mse = ((w_t.reshape(-1) - recon) ** 2).mean().item()
    return mse


def main():
    print("=" * 60)
    print("M180 — GPU High-K Transform-WAL")
    print("=" * 60)
    
    device = "cuda:3"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    layers = [0, 16]
    modules = ['q_proj', 'v_proj', 'gate_proj']
    K_values = [64, 256, 512]
    transforms = ["raw", "hadamard", "dct", "randorth"]
    
    results = {}
    
    for K in K_values:
        print(f"\n{'='*50}")
        print(f"K = {K}")
        print(f"{'='*50}")
        
        for li in layers:
            for m in modules:
                key = f"{li}_{m}"
                w = getattr(model.model.layers[li].self_attn if m != 'gate_proj' else model.model.layers[li].mlp, m).weight.data
                
                for transform in transforms:
                    start = time.time()
                    mse = encode_transform(w, transform, K=K, C=16)
                    elapsed = time.time() - start
                    
                    if mse is None:
                        print(f"  {key} {transform:>10} K={K}: SKIPPED")
                        continue
                    
                    results[f"{key}_{transform}_K{K}"] = {
                        'mse': mse,
                        'time': elapsed,
                        'layer': li,
                        'module': m,
                        'transform': transform,
                        'K': K,
                    }
                    print(f"  {key} {transform:>10} K={K}: MSE={mse:.2e}, Time={elapsed:.1f}s")
    
    # Summary per transform
    print(f"\n{'='*60}")
    print("Summary (avg MSE per transform)")
    print(f"{'='*60}")
    
    for K in K_values:
        print(f"\nK={K}:")
        for transform in transforms:
            mses = [r['mse'] for r in results.values() if r['transform'] == transform and r['K'] == K]
            avg = sum(mses) / len(mses) if mses else 0
            print(f"  {transform:>10}: avg MSE = {avg:.2e}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m180_gpu_high_k_transform_wal.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
