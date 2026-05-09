#!/usr/bin/env python3
"""M69: Position-specific scalar quantization with varying K.

For M=T (each weight has its own codebook), test different K values:
K=16 (4 bits/weight), 32 (5 bits), 64 (6 bits), 128 (7 bits), 256 (8 bits).

Also test: T=8,16,32,64,128 with M=T to see if tile size matters.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.layers.40.self_attn.o_proj"
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


def measure_output(w, recon, D, device):
    dummy = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    dense_out = torch.matmul(dummy, w.T.to(torch.bfloat16))
    recon_out = torch.matmul(dummy, recon.T.to(torch.bfloat16))
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
    
    print(f"\n{'='*100}")
    print(f"{'Tile':>6} | {'K':>6} | {'Bits/w':>8} | {'Ratio':>8} | {'relMSE':>12} | {'Out relMSE':>14} | {'Corr':>8} | {'Status':>8}")
    print(f"{'-'*100}")
    
    for T in [4, 8, 16, 32, 64]:
        if D % T != 0:
            continue
        total_tiles = M * (D // T)
        tiles = w_norm.reshape(total_tiles, T)
        
        for K in [16, 32, 64, 128, 256]:
            bits_per_weight = (T * (K.bit_length() - 1)) / T  # log2(K) bits per weight
            # Actually bits = T * log2(K) / T = log2(K)
            bits_per_weight = K.bit_length() - 1
            ratio = 16 / bits_per_weight
            
            # K-means for each position
            recon_tiles = torch.zeros_like(tiles)
            for pos in range(T):
                pos_values = tiles[:, pos:pos+1]  # [N_tiles, 1]
                samples = pos_values[torch.randperm(total_tiles, device=device)[:min(total_tiles, SAMPLE_SIZE)]]
                atoms = kmeans_batched(samples, K, KMEANS_ITERS)
                
                # Find nearest atom
                dists = (pos_values.unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
                best = dists.argmin(dim=1)
                recon_tiles[:, pos] = atoms[best, 0]
            
            recon = recon_tiles.reshape(M, D) * row_scale
            relMSE = ((recon - w) ** 2).sum() / (w ** 2).sum()
            out_relMSE, corr = measure_output(w, recon, D, device)
            
            status = "TOXIC" if out_relMSE > 0.01 else ("SUSPECT" if out_relMSE > 0.0001 else "OK")
            print(f"{T:>6} | {K:>6} | {bits_per_weight:>8.1f} | {ratio:>8.1f}x | {relMSE.item():>12.6f} | {out_relMSE:>14.8f} | {corr:>8.6f} | {status:>8}")
            
            del recon_tiles, recon
            torch.cuda.empty_cache()
    
    print(f"{'='*100}")
    print("\nM69 complete.")


if __name__ == "__main__":
    main()
