#!/usr/bin/env python3
"""M66: WAL-1 Product Quantization Tile Prototype.

Test Product Quantization per tile:
- Split tile into M subvectors
- K-means on each subvector independently  
- Store M atom_ids per tile
- Measure relMSE and output relMSE.

Then test with residual WAL v2 encoding.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.language_model.layers.40.self_attn.o_proj"
K_ATOMS = 256
KMEANS_ITERS = 5
SAMPLE_SIZE = 500_000

def load_and_extract():
    print(f"Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    param = None
    for name, p in model.named_parameters():
        if LAYER_NAME in name and len(p.shape) == 2:
            param = p
            break
    assert param is not None
    w_cpu = param.data.float().cpu()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return w_cpu


def kmeans_batched(samples, K, iters=5):
    """K-means with batched distance computation."""
    N, T = samples.shape
    device = samples.device
    
    # K-means++ init
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


def pq_encode(tiles, M_subvec, K, iters=5):
    """
    tiles: [N_tiles, T]
    M_subvec: number of subvectors (must divide T)
    Returns: recon_tiles [N_tiles, T], codebooks list of M_subvec [K, T/M]
    """
    N, T = tiles.shape
    assert T % M_subvec == 0
    sub_len = T // M_subvec
    device = tiles.device
    
    codebooks = []
    recon_tiles = torch.zeros_like(tiles)
    
    for m in range(M_subvec):
        sub_tiles = tiles[:, m*sub_len:(m+1)*sub_len]
        
        # Sample for k-means
        if N > SAMPLE_SIZE:
            idx = torch.randperm(N, device=device)[:SAMPLE_SIZE]
            samples = sub_tiles[idx]
        else:
            samples = sub_tiles
        
        atoms = kmeans_batched(samples, K, iters)
        
        # Find nearest atom for ALL sub-tiles
        assignments = torch.empty(N, dtype=torch.int64, device=device)
        for start in range(0, N, 65536):
            end = min(start + 65536, N)
            d = (sub_tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            assignments[start:end] = d.argmin(dim=1)
        
        recon_tiles[:, m*sub_len:(m+1)*sub_len] = atoms[assignments]
        codebooks.append(atoms)
    
    return recon_tiles, codebooks


def main():
    w_cpu = load_and_extract()
    row_scale_cpu = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm_cpu = w_cpu / row_scale_cpu
    M, D = w_norm_cpu.shape
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    w_norm = w_norm_cpu.to(device)
    row_scale = row_scale_cpu.to(device)
    w = w_cpu.to(device)
    
    print(f"\nTarget: {LAYER_NAME}, shape={w.shape}")
    print(f"  M={M}, D={D}, total_weights={M*D:,}")
    print(f"\n{'='*90}")
    print(f"{'Tile':>6} | {'M_sub':>5} | {'relMSE':>12} | {'Out relMSE':>12} | {'Corr':>8} | {'Bits/w':>8} | {'Ratio':>8} | {'Status':>8}")
    print(f"{'-'*90}")
    
    for T in [8, 16, 32, 64, 128]:
        if D % T != 0:
            continue
        num_tiles_per_row = D // T
        total_tiles = M * num_tiles_per_row
        tiles = w_norm.reshape(total_tiles, T)
        
        # Test different M_subvec values
        for M_sub in [1, 2, 4, 8]:
            if T % M_sub != 0:
                continue
            
            t0 = time.time()
            recon_tiles, codebooks = pq_encode(tiles, M_sub, K_ATOMS, KMEANS_ITERS)
            recon_norm = recon_tiles.reshape(M, D)
            recon = recon_norm * row_scale
            
            relMSE = ((recon - w) ** 2).sum() / (w ** 2).sum()
            
            # Output quality
            dummy_input = torch.randn(1, D, dtype=torch.bfloat16, device=device)
            dense_out = torch.matmul(dummy_input, w.T.to(torch.bfloat16))
            pq_out = torch.matmul(dummy_input, recon.T.to(torch.bfloat16))
            out_relMSE = ((pq_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
            out_corr = torch.corrcoef(torch.stack([pq_out.flatten(), dense_out.flatten()]))[0, 1]
            
            bits_per_weight = (M_sub * 8) / T
            ratio = 16 / bits_per_weight
            
            status = "TOXIC" if out_relMSE.item() > 0.01 else ("SUSPECT" if out_relMSE.item() > 0.0001 else "OK")
            
            print(f"{T:>6} | {M_sub:>5} | {relMSE.item():>12.6f} | {out_relMSE.item():>12.8f} | {out_corr.item():>8.6f} | {bits_per_weight:>8.2f} | {ratio:>8.1f}x | {status:>8}")
            
            del recon_tiles, recon_norm, recon, codebooks
            torch.cuda.empty_cache()
    
    print(f"{'='*90}")
    print("\nM66 complete.")


if __name__ == "__main__":
    main()
