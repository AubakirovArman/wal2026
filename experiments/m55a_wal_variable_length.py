#!/usr/bin/env python3
"""M55a: WAL-0 variable-length programs with early stopping.

For each weight, greedily encode until residual < threshold OR lmax reached.
Measure stop_depth distribution and compression gain.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.encoder import build_atoms_kmeans

LAYER_NAME = "model.language_model.layers.40.self_attn.o_proj.weight"
K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000

# Thresholds to test
THRESHOLDS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

print("Loading model...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

w = dict(model.named_parameters())[LAYER_NAME].data.float()
print(f"\nLayer: {LAYER_NAME}")
print(f"Shape: {w.shape}, elements: {w.numel():,}")

row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)

samples = flat[torch.randperm(flat.numel(), device=flat.device)[:SAMPLE_SIZE]] if flat.numel() > SAMPLE_SIZE else flat
atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=flat.device).to(flat.device)

# Fixed-lmax encode for baseline
def encode_with_threshold(flat, atoms, lmax, threshold):
    """GPU-native encode with early stopping."""
    N = flat.numel()
    device = flat.device
    atoms_gpu = atoms.to(device)
    
    indices = torch.zeros(N, lmax, dtype=torch.uint8, device=device)
    signs = torch.zeros(N, lmax, dtype=torch.int8, device=device)
    stop_depth = torch.zeros(N, dtype=torch.uint8, device=device)
    residual = flat.clone()
    recon = torch.zeros(N, dtype=torch.float32, device=device)
    
    for step in range(lmax):
        best_ids = torch.empty(N, dtype=torch.int64, device=device)
        best_signs = torch.empty(N, dtype=torch.int64, device=device)
        
        # Batched search
        batch = 524288
        for start in range(0, N, batch):
            end = min(start + batch, N)
            b = residual[start:end]
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)) ** 2
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)) ** 2
            sz = b.unsqueeze(1) ** 2
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            use_pos = (mp < mn) & (mp < sz.squeeze(1))
            use_neg = (mn <= mp) & (mn < sz.squeeze(1))
            best_ids[start:end] = torch.where(use_pos, ip, torch.where(use_neg, in_, torch.zeros_like(ip)))
            best_signs[start:end] = torch.where(use_pos, 1, torch.where(use_neg, -1, 0))
        
        # Check threshold BEFORE applying step
        if threshold > 0:
            # Zero-step score = residual^2
            zero_score = residual * residual
            # Best step score
            step_score = torch.where(
                use_pos, mp,
                torch.where(use_neg, mn, sz.squeeze(1))
            )
            # For weights where residual is already small enough, don't take step
            # Actually: if |residual| < threshold, stop
            already_good = residual.abs() < threshold
            # Set best_signs to 0 (no step) for already_good weights
            best_signs = torch.where(already_good, 0, best_signs)
            best_ids = torch.where(already_good, 0, best_ids)
        
        indices[:, step] = best_ids.to(torch.uint8)
        signs[:, step] = best_signs.to(torch.int8)
        
        # Apply step only where sign != 0
        step_recon = atoms_gpu[best_ids] * best_signs.float()
        recon += step_recon
        residual -= step_recon
        
        # Update stop_depth for weights that just got zero sign
        if threshold > 0:
            just_stopped = (best_signs == 0) & (stop_depth == 0) & (step > 0)
            stop_depth = torch.where(just_stopped, step, stop_depth)
    
    # For weights that never stopped (used all steps), set stop_depth = lmax
    never_stopped = stop_depth == 0
    # But some may have stopped at step 0 (already good)
    already_good_at_start = (signs[:, 0] == 0)
    stop_depth = torch.where(already_good_at_start, 0, stop_depth)
    stop_depth = torch.where(never_stopped & ~already_good_at_start, lmax, stop_depth)
    
    return indices, signs, stop_depth, recon

print("\n=== Variable Length Analysis ===")
for thresh in THRESHOLDS:
    t0 = time.time()
    indices, signs, stop_depth, recon = encode_with_threshold(flat, atoms, L_MAX, thresh)
    torch.cuda.synchronize()
    enc_time = time.time() - t0
    
    # Verify quality
    recon_2d = recon.reshape(w.shape) * row_scale
    rel_mae = ((recon_2d - w).abs() / w.abs().clamp_min(1e-8)).mean().item()
    
    # Stop depth distribution
    depth_counts = torch.bincount(stop_depth.long(), minlength=L_MAX+1)
    depth_pcts = depth_counts.float() / depth_counts.sum() * 100
    
    # Bits per weight
    # If variable length: need stop_depth (2 bits for lmax=2: 0,1,2) + program bits
    # For depth=0: just stop_depth, no program
    # For depth=1: stop_depth + 1 step (7 bits atom + 2 bits sign = 9 bits)
    # For depth=2: stop_depth + 2 steps (18 bits)
    # But with codebook: depth=0 needs 0 bits (implied), depth=1 and 2 need program_id
    # Simplified: assume we store (stop_depth, program_id) where program_id varies by depth
    
    # For this analysis, compute average bits if we use separate codebooks per depth
    # Depth 0: 0 bits (all weights with depth 0 are identical: no program)
    # Depth 1: need codebook for 1-step programs
    # Depth 2: need codebook for 2-step programs
    
    # Build per-depth codebooks
    avg_bits = 0.0
    for d in range(L_MAX + 1):
        mask = stop_depth == d
        n_d = mask.sum().item()
        if n_d == 0:
            continue
        if d == 0:
            bits_d = 0  # No program needed
        else:
            # Pack programs of depth d
            packed_d = torch.zeros(mask.sum().item(), dtype=torch.int64, device=flat.device)
            idx_d = torch.where(mask)[0]
            if d >= 1:
                # Pack first step
                s = signs[idx_d, 0]
                sm = s.clone()
                sm[s == -1] = 0
                sm[s == 0] = 1
                sm[s == 1] = 2
                packed_d |= (sm.long() << 7) | indices[idx_d, 0].long()
            if d >= 2:
                s = signs[idx_d, 1]
                sm = s.clone()
                sm[s == -1] = 0
                sm[s == 0] = 1
                sm[s == 1] = 2
                packed_d |= (sm.long() << 15) | (indices[idx_d, 1].long() << 8)
            
            unique_d = torch.unique(packed_d).numel()
            bits_d = torch.log2(torch.tensor(float(unique_d))).item() if unique_d > 1 else 0
        
        avg_bits += (n_d / flat.numel()) * bits_d
    
    # Add 2 bits for stop_depth encoding (3 values: 0,1,2)
    avg_bits += 2
    
    print(f"\nthreshold={thresh:.3f}:")
    print(f"  Encode time: {enc_time:.2f}s")
    print(f"  Weight relMAE: {rel_mae:.6f}")
    print(f"  Stop depth dist: " + " | ".join([f"d={i}:{depth_pcts[i]:.1f}%" for i in range(L_MAX+1)]))
    print(f"  Avg bits/weight (varlen + codebook): {avg_bits:.2f}")
    print(f"  vs fixed lmax=2: 10.08 bits → save {10.08 - avg_bits:.2f} bits ({(10.08-avg_bits)/10.08*100:.1f}%)")

print("\nM55a complete.")
