#!/usr/bin/env python3
"""M68: SVD-based row encoding prototype.

Test truncated SVD + coefficient quantization for high compression.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.language_model.layers.40.self_attn.o_proj"

def load_weight():
    print(f"Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory, low_cpu_mem_usage=True,
    )
    param = None
    for name, p in model.named_parameters():
        if LAYER_NAME in name and len(p.shape) == 2:
            param = p
            break
    w_cpu = param.data.float().cpu()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return w_cpu


def kmeans_batched(samples, K, iters=5):
    N, T = samples.shape
    device = samples.device
    atoms = torch.zeros(K, T, device=device, dtype=torch.float32)
    atoms[0] = samples[torch.randint(0, N, (1,), device=device)]
    for k in range(1, K):
        dists = torch.empty(N, k, device=device)
        for start in range(0, N, 65536):
            end = min(start + 65536, N)
            d = (samples[start:end].unsqueeze(1) - atoms[:k].unsqueeze(0)).pow(2).sum(dim=2)
            dists[start:end] = d
        min_dists = dists.min(dim=1)[0]
        probs = min_dists / min_dists.sum()
        cumprobs = probs.cumsum(dim=0)
        idx = torch.searchsorted(cumprobs, torch.rand(1, device=device))
        idx = idx.clamp_max(N - 1)
        atoms[k] = samples[idx]
    for _ in range(iters):
        assignments = torch.empty(N, dtype=torch.int64, device=device)
        for start in range(0, N, 65536):
            end = min(start + 65536, N)
            d = (samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            assignments[start:end] = d.argmin(dim=1)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = samples[mask].mean(dim=0)
    return atoms


def quantize_vector(vec, levels):
    """Uniform quantization of a vector to given number of levels."""
    min_v = vec.min()
    max_v = vec.max()
    if max_v == min_v:
        return vec, min_v, max_v
    scaled = (vec - min_v) / (max_v - min_v) * (levels - 1)
    quantized = torch.round(scaled).clamp(0, levels - 1)
    recon = quantized / (levels - 1) * (max_v - min_v) + min_v
    return recon, min_v, max_v


def main():
    w_cpu = load_weight()
    row_scale_cpu = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm_cpu = w_cpu / row_scale_cpu
    M, D = w_norm_cpu.shape
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    w_norm = w_norm_cpu.to(device)
    row_scale = row_scale_cpu.to(device)
    w = w_cpu.to(device)
    
    print(f"\nTarget: {LAYER_NAME}, shape={w.shape}")
    print(f"  M={M}, D={D}, total_weights={M*D:,}")
    
    # SVD on normalized matrix
    print("\nComputing SVD...")
    U, S, Vh = torch.linalg.svd(w_norm, full_matrices=False)
    print(f"  Singular values (first 20): {S[:20].cpu().numpy()}")
    print(f"  Energy cumsum: {(S.cumsum(0) / S.sum()).cpu().numpy()[:20]}")
    
    def measure(w, recon):
        relMSE = ((recon - w) ** 2).sum() / (w ** 2).sum()
        dummy = torch.randn(1, D, dtype=torch.bfloat16, device=device)
        d_out = torch.matmul(dummy, w.T.to(torch.bfloat16))
        r_out = torch.matmul(dummy, recon.T.to(torch.bfloat16))
        out_relMSE = ((r_out - d_out) ** 2).sum() / (d_out ** 2).sum()
        corr = torch.corrcoef(torch.stack([r_out.flatten(), d_out.flatten()]))[0, 1]
        return relMSE.item(), out_relMSE.item(), corr.item()
    
    print(f"\n{'='*95}")
    print("PHASE 1: Truncated SVD (exact coeffs, no quantization)")
    print(f"{'Rank':>6} | {'relMSE':>12} | {'Out relMSE':>14} | {'Corr':>8} | {'Params':>12} | {'Ratio':>8} | {'Status':>8}")
    print(f"{'-'*95}")
    
    for R in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        recon_norm = (U[:, :R] * S[:R]) @ Vh[:R, :]
        recon = recon_norm * row_scale
        relMSE, out_relMSE, corr = measure(w, recon)
        params = M * R + R + R * D
        ratio = (M * D) / params
        status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
        print(f"{R:>6} | {relMSE:>12.6f} | {out_relMSE:>14.8f} | {corr:>8.6f} | {params:>12,} | {ratio:>8.1f}x | {status:>8}")
    
    # === PHASE 2: SVD with coefficient quantization ===
    print(f"\n{'='*95}")
    print("PHASE 2: SVD + coefficient quantization")
    print(f"{'Rank':>6} | {'U_bits':>8} | {'S_bits':>8} | {'V_bits':>8} | {'relMSE':>12} | {'Out relMSE':>14} | {'Ratio':>8}")
    print(f"{'-'*95}")
    
    for R in [16, 32, 64, 128, 256]:
        for u_bits, s_bits, v_bits in [(8, 16, 8), (8, 16, 8), (4, 8, 4), (4, 8, 4)]:
            # Quantize U coefficients
            U_q, U_min, U_max = quantize_vector(U[:, :R], 2**u_bits)
            S_q, S_min, S_max = quantize_vector(S[:R], 2**s_bits)
            V_q, V_min, V_max = quantize_vector(Vh[:R, :], 2**v_bits)
            
            recon_norm = (U_q * S_q) @ V_q
            recon = recon_norm * row_scale
            relMSE, out_relMSE, corr = measure(w, recon)
            
            # Storage
            u_storage = M * R * u_bits
            s_storage = R * s_bits
            v_storage = R * D * v_bits
            total_bits = u_storage + s_storage + v_storage
            total_weights = M * D * 16  # original bf16
            ratio = total_weights / total_bits
            
            status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
            print(f"{R:>6} | {u_bits:>8} | {s_bits:>8} | {v_bits:>8} | {relMSE:>12.6f} | {out_relMSE:>14.8f} | {ratio:>8.1f}x | {status:>8}")
    
    # === PHASE 3: SVD + residual WAL v2 ===
    print(f"\n{'='*95}")
    print("PHASE 3: SVD coarse + WAL v2 residual")
    print(f"{'Rank':>6} | {'relMSE(coarse)':>16} | {'K_res':>6} | {'C_res':>6} | {'relMSE(total)':>14} | {'Out relMSE':>14} | {'Bits/w':>8}")
    print(f"{'-'*95}")
    
    for R in [16, 32, 64, 128]:
        recon_norm = (U[:, :R] * S[:R]) @ Vh[:R, :]
        recon_coarse = recon_norm * row_scale
        residual = w - recon_coarse
        
        for K_res, C_res in [(16, 4), (32, 4), (32, 8), (64, 8)]:
            # WAL v2 on residual
            samples = residual.view(-1)[torch.randperm(M*D, device=device)[:500000]]
            atoms_res = kmeans_batched(samples.unsqueeze(1), K_res, 5).squeeze(1)
            ratios = (samples.unsqueeze(0) / atoms_res.unsqueeze(1)).abs()
            ratios = ratios[ratios.isfinite()]
            coeffs_res = kmeans_batched(ratios.unsqueeze(1), C_res, 5).squeeze(1)
            
            r = residual.view(-1).unsqueeze(1)
            recons = atoms_res.unsqueeze(0).unsqueeze(2) * coeffs_res.unsqueeze(0).unsqueeze(1)
            errs = (r.unsqueeze(1).unsqueeze(2) - recons).abs()
            best = errs.view(M*D, -1).argmin(dim=1)
            recon_res = recons.view(M*D, -1)[torch.arange(M*D, device=device), best].reshape(M, D)
            
            final_recon = recon_coarse + recon_res
            relMSE_c = ((recon_coarse - w) ** 2).sum() / (w ** 2).sum()
            relMSE_f = ((final_recon - w) ** 2).sum() / (w ** 2).sum()
            _, out_relMSE, _ = measure(w, final_recon)
            
            # Bits: SVD params + WAL v2 residual
            svd_bits = R * (M + 1 + D) * 16  # bf16 for SVD
            wal_bits = M * D * (8 + 8)  # 16 bits per weight for WAL v2
            total_bits = svd_bits + wal_bits
            bits_per_weight = total_bits / (M * D)
            
            status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
            print(f"{R:>6} | {relMSE_c.item():>16.6f} | {K_res:>6} | {C_res:>6} | {relMSE_f.item():>14.8f} | {out_relMSE:>14.8f} | {bits_per_weight:>8.2f} | {status:>8}")
            
            del recon_res, final_recon
            torch.cuda.empty_cache()
        
        del recon_coarse, residual
        torch.cuda.empty_cache()
    
    print(f"{'='*95}")
    print("\nM68 complete.")


if __name__ == "__main__":
    main()
