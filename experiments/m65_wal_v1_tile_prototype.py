#!/usr/bin/env python3
"""M65: WAL-1 Tile-Based Vector Prototype.

Test vector quantization per tile with single atom lookup.
Tile sizes: 8, 16, 32, 64, 128, 256.
Measure relMSE and output relMSE for each.

NOTE: Unloads model after weight extraction to free GPU memory.
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
print(f"\nTarget: {name}, shape={param.shape}, device={param.device}")

# Extract weight to CPU, then unload model
w_cpu = param.data.float().cpu()
row_scale_cpu = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm_cpu = w_cpu / row_scale_cpu
M, D = w_norm_cpu.shape

print(f"  M={M}, D={D}, total_weights={M*D:,}")
print("  Unloading model to free GPU memory...")
del model
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU memory freed.")

# Move to GPU for computation
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
w_norm = w_norm_cpu.to(device)
row_scale = row_scale_cpu.to(device)
w = w_cpu.to(device)

print(f"\n{'='*75}")
print(f"{'Tile Size':>10} | {'relMSE':>12} | {'Output relMSE':>14} | {'Corr':>8} | {'Bits/w':>8} | {'Ratio':>8}")
print(f"{'-'*75}")


def kmeans_tiles(tiles, K, iters=5):
    """K-means on tiles. tiles: [N, T] on GPU."""
    N, T = tiles.shape
    device = tiles.device
    
    # K-means++ init
    atoms = torch.zeros(K, T, device=device, dtype=torch.float32)
    atoms[0] = tiles[torch.randint(0, N, (1,), device=device)]
    for k in range(1, K):
        # Batched distance computation
        dists = torch.empty(N, k, device=device)
        for start in range(0, N, 65536):
            end = min(start + 65536, N)
            d = (tiles[start:end].unsqueeze(1) - atoms[:k].unsqueeze(0)).pow(2).sum(dim=2)
            dists[start:end] = d
        min_dists = dists.min(dim=1)[0]
        probs = min_dists / min_dists.sum()
        cumprobs = probs.cumsum(dim=0)
        idx = torch.searchsorted(cumprobs, torch.rand(1, device=device))
        idx = idx.clamp_max(N - 1)
        atoms[k] = tiles[idx]
    
    # K-means iterations
    for it in range(iters):
        # Assignments (batched)
        assignments = torch.empty(N, dtype=torch.int64, device=device)
        for start in range(0, N, 65536):
            end = min(start + 65536, N)
            d = (tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            assignments[start:end] = d.argmin(dim=1)
        
        # Update atoms
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = tiles[mask].mean(dim=0)
        
        if it == iters - 1:
            # Final assignment
            for start in range(0, N, 65536):
                end = min(start + 65536, N)
                d = (tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
                assignments[start:end] = d.argmin(dim=1)
            return atoms, assignments
    
    # If iters=0 (shouldn't happen), return initial
    assignments = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, 65536):
        end = min(start + 65536, N)
        d = (tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
        assignments[start:end] = d.argmin(dim=1)
    return atoms, assignments


for T in [8, 16, 32, 64, 128, 256]:
    if D % T != 0:
        continue
    
    num_tiles_per_row = D // T
    total_tiles = M * num_tiles_per_row
    
    # Reshape into tiles
    tiles = w_norm.reshape(M * num_tiles_per_row, T)
    
    # Sample for k-means
    if total_tiles > SAMPLE_SIZE:
        idx_samp = torch.randperm(total_tiles, device=device)[:SAMPLE_SIZE]
        samples = tiles[idx_samp]
    else:
        samples = tiles
    
    # K-means
    t0 = time.time()
    atoms, assignments = kmeans_tiles(samples, K_ATOMS, KMEANS_ITERS)
    
    # For all tiles, find nearest atom (batched)
    best_atom = torch.empty(total_tiles, dtype=torch.int64, device=device)
    for start in range(0, total_tiles, 65536):
        end = min(start + 65536, total_tiles)
        d = (tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
        best_atom[start:end] = d.argmin(dim=1)
    km_time = time.time() - t0
    
    # Reconstruct
    recon_tiles = atoms[best_atom]
    recon_norm = recon_tiles.reshape(M, D)
    recon = recon_norm * row_scale
    
    # Metrics
    relMSE = ((recon - w) ** 2).sum() / (w ** 2).sum()
    
    # Output quality via dummy matmul
    dummy_input = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    dense_out = torch.matmul(dummy_input, w.T.to(torch.bfloat16))
    tile_out = torch.matmul(dummy_input, recon.T.to(torch.bfloat16))
    output_relMSE = ((tile_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
    output_corr = torch.corrcoef(torch.stack([tile_out.flatten(), dense_out.flatten()]))[0, 1]
    
    # Compression
    bits_per_weight = 8 / T  # 1 byte (atom_id) per tile
    ratio = 16 / bits_per_weight
    
    status = ""
    if output_relMSE.item() > 0.01:
        status = " TOXIC"
    elif output_relMSE.item() > 0.0001:
        status = " SUSPECT"
    
    print(f"{T:>10} | {relMSE.item():>12.6f} | {output_relMSE.item():>14.8f} | {output_corr.item():>8.6f} | {bits_per_weight:>8.2f} | {ratio:>8.1f}x |{status}")
    
    del atoms, recon, recon_norm, recon_tiles, best_atom, assignments
    torch.cuda.empty_cache()

print(f"{'='*75}")
print("\nM65 complete.")
