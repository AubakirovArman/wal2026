"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M74: WAL-1 Two-Term Subroutine Prototype (GPU-batched).

All heavy ops on GPU with batching. Clean GPU memory first.
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
C_COEFFS = 16
KMEANS_ITERS = 5
SAMPLE_SIZE = 500_000
BATCH = 1_048_576


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


def kmeans_1d_gpu(samples, K, iters=5):
    """1D k-means on GPU."""
    device = samples.device
    samples = samples.float().sort()[0]
    step = max(1, samples.numel() // K)
    atoms = samples[::step][:K].clone()
    
    for _ in range(iters):
        # Batched assignments
        assignments = torch.empty(samples.numel(), dtype=torch.int64, device=device)
        for start in range(0, samples.numel(), BATCH):
            end = min(start + BATCH, samples.numel())
            d = (samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs()
            assignments[start:end] = d.argmin(dim=1)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = samples[mask].mean()
    return atoms


def build_coeffs_gpu(flat, atoms, C):
    """Build coeff table: ratio to nearest atom for each weight."""
    device = flat.device
    # Find nearest atom per weight (batched)
    best_atom = torch.empty(flat.numel(), dtype=torch.int64, device=device)
    for start in range(0, flat.numel(), BATCH):
        end = min(start + BATCH, flat.numel())
        d = (flat[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs()
        best_atom[start:end] = d.argmin(dim=1)
        del d
    
    ratios = (flat / atoms[best_atom]).abs()
    ratios = ratios[ratios.isfinite()]
    return kmeans_1d_gpu(ratios, C, KMEANS_ITERS)


def wal_encode_v2_gpu(weights, atoms, coeffs):
    """Batched scalar encoding on GPU."""
    N = weights.numel()
    device = weights.device
    a = atoms.to(device)
    c = coeffs.to(device)
    K = a.numel()
    C = c.numel()
    
    a_ids = torch.empty(N, dtype=torch.uint8, device=device)
    c_ids = torch.empty(N, dtype=torch.uint8, device=device)
    recon = torch.empty(N, dtype=torch.float32, device=device)
    
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        w = weights[start:end]
        recons = a.unsqueeze(1) * c.unsqueeze(0)  # [K, C]
        errs = (w.unsqueeze(1).unsqueeze(2) - recons.unsqueeze(0)).abs()
        best = errs.view(end-start, -1).argmin(dim=1)
        a_ids[start:end] = (best // C).to(torch.uint8)
        c_ids[start:end] = (best % C).to(torch.uint8)
        recon[start:end] = recons.view(-1)[best]
        del recons, errs, best
    
    return a_ids, c_ids, recon


def two_term_encode_gpu(weights, atoms, coeffs):
    """Batched greedy two-term encoding."""
    a1_ids, c1_ids, recon1 = wal_encode_v2_gpu(weights, atoms, coeffs)
    residual = weights - recon1
    a2_ids, c2_ids, recon2 = wal_encode_v2_gpu(residual, atoms, coeffs)
    return a1_ids, c1_ids, a2_ids, c2_ids, recon1 + recon2


def build_subroutines_gpu(a1_ids, c1_ids, a2_ids, c2_ids, K_sub=256):
    """Cluster 4-tuples on GPU with batched k-means."""
    device = a1_ids.device
    N = a1_ids.numel()
    
    # Build features
    features = torch.stack([
        a1_ids.float() / K_ATOMS,
        c1_ids.float() / C_COEFFS,
        a2_ids.float() / K_ATOMS,
        c2_ids.float() / C_COEFFS,
    ], dim=1)
    
    # Sample
    idx = torch.randperm(N, device=device)[:min(N, SAMPLE_SIZE)]
    samples = features[idx]
    
    # K-means++ init
    centroids = torch.zeros(K_sub, 4, device=device)
    centroids[0] = samples[0]
    for k in range(1, K_sub):
        d = (samples.unsqueeze(1) - centroids[:k].unsqueeze(0)).pow(2).sum(dim=2)
        min_d = d.min(dim=1)[0]
        probs = min_d / min_d.sum()
        cumprobs = probs.cumsum(dim=0)
        idx_k = torch.searchsorted(cumprobs, torch.rand(1, device=device))
        idx_k = idx_k.clamp_max(len(samples) - 1)
        centroids[k] = samples[idx_k]
    
    # K-means iterations (batched)
    for _ in range(5):
        assignments = torch.empty(N, dtype=torch.int64, device=device)
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            d = (features[start:end].unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=2)
            assignments[start:end] = d.argmin(dim=1)
        
        for k in range(K_sub):
            mask = assignments == k
            if mask.any():
                centroids[k] = features[mask].mean(dim=0)
    
    # Final assignment
    assignments = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        d = (features[start:end].unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=2)
        assignments[start:end] = d.argmin(dim=1)
    
    # Build table
    table = []
    for k in range(K_sub):
        mask = assignments == k
        if mask.any():
            rep = int(mask.nonzero()[0])
            table.append((int(a1_ids[rep]), int(c1_ids[rep]), int(a2_ids[rep]), int(c2_ids[rep])))
        else:
            table.append((0, 0, 0, 0))
    
    return assignments, table


def subroutine_recon_gpu(sub_ids, table, atoms, coeffs):
    """Batched subroutine reconstruction."""
    device = sub_ids.device
    N = sub_ids.numel()
    recon = torch.zeros(N, dtype=torch.float32, device=device)
    
    for k, (a1, c1, a2, c2) in enumerate(table):
        mask = sub_ids == k
        if mask.any():
            val = atoms[a1] * coeffs[c1] + atoms[a2] * coeffs[c2]
            recon[mask] = val
    
    return recon


def main():
    w_cpu = load_weight()
    row_scale_cpu = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm_cpu = w_cpu / row_scale_cpu
    M, D = w_norm_cpu.shape
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flat = w_norm_cpu.to(device).reshape(-1)
    w = w_cpu.to(device)
    row_scale = row_scale_cpu.to(device)
    
    print(f"\nTarget: {LAYER_NAME}, shape={w.shape}")
    print(f"  M={M}, D={D}, total_weights={M*D:,}")
    print(f"  Device: {device}")
    
    # Build atoms
    print("\nBuilding L0 atoms...")
    t0 = time.time()
    samples = flat[torch.randperm(flat.numel(), device=device)[:min(flat.numel(), SAMPLE_SIZE)]]
    atoms = kmeans_1d_gpu(samples, K_ATOMS, KMEANS_ITERS)
    print(f"  Atoms: {time.time()-t0:.1f}s")
    
    print("Building coeffs...")
    t0 = time.time()
    coeffs = build_coeffs_gpu(flat, atoms, C_COEFFS)
    print(f"  Coeffs: {time.time()-t0:.1f}s")
    
    # WAL v2
    print(f"\n{'='*80}")
    print("PHASE 1: WAL v2 Baseline")
    t0 = time.time()
    a_ids, c_ids, recon_v2 = wal_encode_v2_gpu(flat, atoms, coeffs)
    recon_v2_full = recon_v2.reshape(M, D) * row_scale
    relMSE_v2 = ((recon_v2_full - w) ** 2).sum() / (w ** 2).sum()
    print(f"  relMSE: {relMSE_v2.item():.8f}  ({time.time()-t0:.1f}s)")
    del recon_v2
    torch.cuda.empty_cache()
    
    # Two-term
    print(f"\n{'='*80}")
    print("PHASE 2: Two-term greedy (32 bits/weight)")
    t0 = time.time()
    a1_ids, c1_ids, a2_ids, c2_ids, recon_two = two_term_encode_gpu(flat, atoms, coeffs)
    recon_two_full = recon_two.reshape(M, D) * row_scale
    relMSE_two = ((recon_two_full - w) ** 2).sum() / (w ** 2).sum()
    print(f"  relMSE: {relMSE_two.item():.8f}  ({time.time()-t0:.1f}s)")
    del recon_two
    torch.cuda.empty_cache()
    
    # Subroutines
    print(f"\n{'='*80}")
    print("PHASE 3: Subroutine clustering (256 subs, 12 bits/weight)")
    t0 = time.time()
    sub_ids, table = build_subroutines_gpu(a1_ids, c1_ids, a2_ids, c2_ids, K_sub=256)
    recon_sub = subroutine_recon_gpu(sub_ids, table, atoms, coeffs)
    recon_sub_full = recon_sub.reshape(M, D) * row_scale
    relMSE_sub = ((recon_sub_full - w) ** 2).sum() / (w ** 2).sum()
    print(f"  relMSE: {relMSE_sub.item():.8f}  ({time.time()-t0:.1f}s)")
    del recon_sub
    torch.cuda.empty_cache()
    
    # Output quality
    dummy = torch.randn(1, D, dtype=torch.bfloat16, device=device)
    dense_out = torch.matmul(dummy, w.T.to(torch.bfloat16))
    
    v2_out = torch.matmul(dummy, recon_v2_full.T.to(torch.bfloat16))
    two_out = torch.matmul(dummy, recon_two_full.T.to(torch.bfloat16))
    sub_out = torch.matmul(dummy, recon_sub_full.T.to(torch.bfloat16))
    
    out_v2 = ((v2_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
    out_two = ((two_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
    out_sub = ((sub_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
    
    print(f"\n{'='*80}")
    print(f"{'Method':<30} | {'relMSE':>12} | {'Out relMSE':>14}")
    print(f"{'-'*80}")
    print(f"{'WAL v2 (single-term)':<30} | {relMSE_v2.item():>12.8f} | {out_v2.item():>14.8f}")
    print(f"{'Two-term greedy (32 bits)':<30} | {relMSE_two.item():>12.8f} | {out_two.item():>14.8f}")
    print(f"{'Subroutine (12 bits)':<30} | {relMSE_sub.item():>12.8f} | {out_sub.item():>14.8f}")
    print(f"{'='*80}")
    
    print("\nM74 complete.")


if __name__ == "__main__":
    main()
