"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M67: Systematic PQ sweep + two-tier residual encoding.

Goal: find best compression/quality tradeoff for PQ-based WAL-1.

Tests:
1. Various (T, M) combinations
2. Two-tier: PQ coarse + PQ residual
3. Two-tier: PQ coarse + WAL v2 residual  
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.layers.40.self_attn.o_proj"
K_ATOMS = 256
KMEANS_ITERS = 5
SAMPLE_SIZE = 500_000


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


def pq_encode(tiles, M_sub, K, iters=5):
    """Returns recon_tiles [N, T], codebooks list."""
    N, T = tiles.shape
    sub_len = T // M_sub
    device = tiles.device
    codebooks = []
    recon_tiles = torch.zeros_like(tiles)
    for m in range(M_sub):
        sub_tiles = tiles[:, m*sub_len:(m+1)*sub_len]
        samples = sub_tiles[torch.randperm(N, device=device)[:min(N, SAMPLE_SIZE)]]
        atoms = kmeans_batched(samples, K, iters)
        assignments = torch.empty(N, dtype=torch.int64, device=device)
        for start in range(0, N, 65536):
            end = min(start + 65536, N)
            d = (sub_tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            assignments[start:end] = d.argmin(dim=1)
        recon_tiles[:, m*sub_len:(m+1)*sub_len] = atoms[assignments]
        codebooks.append(atoms)
    return recon_tiles, codebooks


def wal_v2_residual_encode(residual, K=16, C=4, iters=5):
    """Encode residual with small WAL v2 codebook."""
    N = residual.numel()
    device = residual.device
    samples = residual.view(-1)[torch.randperm(N, device=device)[:min(N, SAMPLE_SIZE)]]
    
    # Atom table: k-means on residuals
    atoms = kmeans_batched(samples.unsqueeze(1), K, iters).squeeze(1)  # [K]
    
    # Coeff table: k-means on |residual / atom| ratios
    ratios = (samples.unsqueeze(0) / atoms.unsqueeze(1)).abs()  # [K, N]
    ratios = ratios[ratios.isfinite()]
    coeff_atoms = kmeans_batched(ratios.unsqueeze(1), C, iters).squeeze(1)  # [C]
    
    # Encode all residuals
    r = residual.view(-1).unsqueeze(1)  # [N, 1]
    a = atoms.unsqueeze(0).unsqueeze(2)  # [1, K, 1]
    c = coeff_atoms.unsqueeze(0).unsqueeze(1)  # [1, 1, C]
    recons = a * c  # [N, K, C] broadcast
    errs = (r.unsqueeze(1).unsqueeze(2) - recons).abs()
    flat_errs = errs.view(N, -1)
    best = flat_errs.argmin(dim=1)
    atom_ids = (best // C).to(torch.uint8)
    coeff_ids = (best % C).to(torch.uint8)
    recon_vals = recons.view(N, -1)[torch.arange(N, device=device), best]
    
    return recon_vals.reshape(residual.shape), atom_ids, coeff_ids, atoms, coeff_atoms


def measure_output(w, recon, D, device):
    dummy_input = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    dense_out = torch.matmul(dummy_input, w.T.to(torch.bfloat16))
    recon_out = torch.matmul(dummy_input, recon.T.to(torch.bfloat16))
    out_relMSE = ((recon_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
    corr = torch.corrcoef(torch.stack([recon_out.flatten(), dense_out.flatten()]))[0, 1]
    return out_relMSE.item(), corr.item()


def main():
    w_cpu = load_weight()
    row_scale_cpu = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm_cpu = w_cpu / row_scale_cpu
    M, D = w_norm_cpu.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w_norm = w_norm_cpu.to(device)
    row_scale = row_scale_cpu.to(device)
    w = w_cpu.to(device)
    
    print(f"\nTarget: {LAYER_NAME}, shape={w.shape}")
    print(f"  M={M}, D={D}, total_weights={M*D:,}")
    
    # === Phase 1: Single-tier PQ sweep ===
    print(f"\n{'='*100}")
    print("PHASE 1: Single-tier PQ")
    print(f"{'Tile':>6} | {'M_sub':>5} | {'Bits/w':>8} | {'Ratio':>8} | {'relMSE':>12} | {'Out relMSE':>14} | {'Corr':>8} | {'Status':>8}")
    print(f"{'-'*100}")
    
    for T in [4, 8, 16]:
        if D % T != 0:
            continue
        total_tiles = M * (D // T)
        tiles = w_norm.reshape(total_tiles, T)
        for M_sub in range(1, T+1):
            if T % M_sub != 0:
                continue
            recon_tiles, _ = pq_encode(tiles, M_sub, K_ATOMS, KMEANS_ITERS)
            recon_norm = recon_tiles.reshape(M, D)
            recon = recon_norm * row_scale
            relMSE = ((recon - w) ** 2).sum() / (w ** 2).sum()
            out_relMSE, corr = measure_output(w, recon, D, device)
            bits_per_weight = (M_sub * 8) / T
            ratio = 16 / bits_per_weight
            status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
            print(f"{T:>6} | {M_sub:>5} | {bits_per_weight:>8.2f} | {ratio:>8.1f}x | {relMSE.item():>12.6f} | {out_relMSE:>14.8f} | {corr:>8.6f} | {status:>8}")
            del recon_tiles, recon_norm, recon
            torch.cuda.empty_cache()
    
    # === Phase 2: Two-tier PQ coarse + PQ residual ===
    print(f"\n{'='*100}")
    print("PHASE 2: Two-tier PQ (coarse + residual PQ)")
    print(f"{'T1':>4} | {'M1':>3} | {'T2':>4} | {'M2':>3} | {'Bits/w':>8} | {'Ratio':>8} | {'relMSE':>12} | {'Out relMSE':>14} | {'Status':>8}")
    print(f"{'-'*100}")
    
    # Best coarse configs to try
    coarse_configs = [(8, 4), (8, 5), (8, 6), (16, 4), (16, 6), (16, 8)]
    residual_configs = [(8, 4), (8, 6), (8, 8)]
    
    for T1, M1 in coarse_configs:
        if D % T1 != 0:
            continue
        total_tiles1 = M * (D // T1)
        tiles1 = w_norm.reshape(total_tiles1, T1)
        recon1_tiles, _ = pq_encode(tiles1, M1, K_ATOMS, KMEANS_ITERS)
        recon1 = (recon1_tiles.reshape(M, D) * row_scale)
        
        residual = w_norm - recon1_tiles.reshape(M, D)  # residual in normalized space
        
        for T2, M2 in residual_configs:
            if T1 % T2 != 0 and T2 % T1 != 0:
                continue  # tiles must align
            # Use same tile size for simplicity
            if T1 != T2:
                continue
            tiles2 = residual.reshape(total_tiles1, T1)
            recon2_tiles, _ = pq_encode(tiles2, M2, K_ATOMS, KMEANS_ITERS)
            recon2 = recon2_tiles.reshape(M, D) * row_scale
            final_recon = recon1 + recon2
            
            relMSE = ((final_recon - w) ** 2).sum() / (w ** 2).sum()
            out_relMSE, _ = measure_output(w, final_recon, D, device)
            bits = (M1 * 8 + M2 * 8) / T1
            ratio = 16 / bits
            status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
            print(f"{T1:>4} | {M1:>3} | {T2:>4} | {M2:>3} | {bits:>8.2f} | {ratio:>8.1f}x | {relMSE.item():>12.6f} | {out_relMSE:>14.8f} | {status:>8}")
            del recon2_tiles, recon2, final_recon
            torch.cuda.empty_cache()
        
        del recon1_tiles, recon1, residual
        torch.cuda.empty_cache()
    
    # === Phase 3: Two-tier PQ + WAL v2 residual ===
    print(f"\n{'='*100}")
    print("PHASE 3: Two-tier PQ coarse + WAL v2 residual")
    print(f"{'T':>4} | {'M':>3} | {'K_res':>5} | {'C_res':>5} | {'Bits/w':>8} | {'Ratio':>8} | {'relMSE':>12} | {'Out relMSE':>14} | {'Status':>8}")
    print(f"{'-'*100}")
    
    for T, M_sub in [(8, 4), (8, 5), (8, 6), (16, 4), (16, 6)]:
        if D % T != 0:
            continue
        total_tiles = M * (D // T)
        tiles = w_norm.reshape(total_tiles, T)
        recon1_tiles, _ = pq_encode(tiles, M_sub, K_ATOMS, KMEANS_ITERS)
        recon1 = recon1_tiles.reshape(M, D) * row_scale
        
        residual = w - recon1
        
        for K_res, C_res in [(16, 4), (16, 8), (32, 4), (32, 8)]:
            recon2, _, _, _, _ = wal_v2_residual_encode(residual, K_res, C_res, KMEANS_ITERS)
            final_recon = recon1 + recon2
            
            relMSE = ((final_recon - w) ** 2).sum() / (w ** 2).sum()
            out_relMSE, _ = measure_output(w, final_recon, D, device)
            bits = (M_sub * 8) / T + (8 + 8)  # 16 bits for WAL v2 atom_id + coeff_id
            ratio = 16 / bits
            status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
            print(f"{T:>4} | {M_sub:>3} | {K_res:>5} | {C_res:>5} | {bits:>8.2f} | {ratio:>8.1f}x | {relMSE.item():>12.6f} | {out_relMSE:>14.8f} | {status:>8}")
            del recon2, final_recon
            torch.cuda.empty_cache()
        
        del recon1_tiles, recon1, residual
        torch.cuda.empty_cache()
    
    print(f"{'='*100}")
    print("\nM67 complete.")


if __name__ == "__main__":
    main()
