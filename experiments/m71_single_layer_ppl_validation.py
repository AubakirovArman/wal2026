"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M71: Single-layer PPL validation of M65-M69 findings.

Replaces ONLY layer 40 o_proj with each method, measures PPL.
This validates whether single-layer output_relMSE correlates with full PPL.

Methods tested:
  A. M65 T=8 (worst vector quantization)
  B. M66 T=8,M=8 (best PQ)
  C. M67 two-tier T=8,M=4+4 (two-tier PQ)
  D. M69 K=128 (position-specific, SUSPECT)
  E. M69 K=256 (position-specific, OK by relMSE)
  F. WAL v2 (baseline)
"""
import torch
import time
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.layers.40.self_attn.o_proj"
K_ATOMS = 256
SAMPLE_SIZE = 500_000
KMEANS_ITERS = 5


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


def run_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"][:])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_length = 2048
    stride = 512
    max_samples = 16
    nlls = []
    prev_end_loc = 0
    num_steps = 0
    for begin_loc in range(0, seq_len, stride):
        if num_steps >= max_samples:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        num_steps += 1
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item(), num_steps, end_loc


def encode_m65_tile(w, T):
    """M65: Single atom lookup per tile."""
    M, D = w.shape
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    tiles = w_norm.reshape(M * (D // T), T)
    samples = tiles[torch.randperm(tiles.shape[0], device=w.device)[:min(tiles.shape[0], SAMPLE_SIZE)]]
    atoms = kmeans_batched(samples, K_ATOMS, KMEANS_ITERS)
    best = torch.empty(tiles.shape[0], dtype=torch.int64, device=w.device)
    for start in range(0, tiles.shape[0], 65536):
        end = min(start + 65536, tiles.shape[0])
        d = (tiles[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
        best[start:end] = d.argmin(dim=1)
    recon_tiles = atoms[best]
    recon = recon_tiles.reshape(M, D) * row_scale
    return recon.to(w.dtype)


def encode_m66_pq(w, T, M_sub):
    """M66: Product quantization."""
    M, D = w.shape
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    tiles = w_norm.reshape(M * (D // T), T)
    sub_len = T // M_sub
    recon_tiles = torch.zeros_like(tiles)
    for m in range(M_sub):
        sub = tiles[:, m*sub_len:(m+1)*sub_len]
        samples = sub[torch.randperm(sub.shape[0], device=w.device)[:min(sub.shape[0], SAMPLE_SIZE)]]
        atoms = kmeans_batched(samples, K_ATOMS, KMEANS_ITERS)
        best = torch.empty(sub.shape[0], dtype=torch.int64, device=w.device)
        for start in range(0, sub.shape[0], 65536):
            end = min(start + 65536, sub.shape[0])
            d = (sub[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            best[start:end] = d.argmin(dim=1)
        recon_tiles[:, m*sub_len:(m+1)*sub_len] = atoms[best]
    recon = recon_tiles.reshape(M, D) * row_scale
    return recon.to(w.dtype)


def encode_m67_twotier(w, T, M1, M2):
    """M67: Two-tier PQ coarse + residual PQ."""
    M, D = w.shape
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    tiles = w_norm.reshape(M * (D // T), T)
    sub_len = T // M1
    recon1 = torch.zeros_like(tiles)
    for m in range(M1):
        sub = tiles[:, m*sub_len:(m+1)*sub_len]
        samples = sub[torch.randperm(sub.shape[0], device=w.device)[:min(sub.shape[0], SAMPLE_SIZE)]]
        atoms = kmeans_batched(samples, K_ATOMS, KMEANS_ITERS)
        best = torch.empty(sub.shape[0], dtype=torch.int64, device=w.device)
        for start in range(0, sub.shape[0], 65536):
            end = min(start + 65536, sub.shape[0])
            d = (sub[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            best[start:end] = d.argmin(dim=1)
        recon1[:, m*sub_len:(m+1)*sub_len] = atoms[best]
    residual = tiles - recon1
    sub_len2 = T // M2
    recon2 = torch.zeros_like(tiles)
    for m in range(M2):
        sub = residual[:, m*sub_len2:(m+1)*sub_len2]
        samples = sub[torch.randperm(sub.shape[0], device=w.device)[:min(sub.shape[0], SAMPLE_SIZE)]]
        atoms = kmeans_batched(samples, K_ATOMS, KMEANS_ITERS)
        best = torch.empty(sub.shape[0], dtype=torch.int64, device=w.device)
        for start in range(0, sub.shape[0], 65536):
            end = min(start + 65536, sub.shape[0])
            d = (sub[start:end].unsqueeze(1) - atoms.unsqueeze(0)).pow(2).sum(dim=2)
            best[start:end] = d.argmin(dim=1)
        recon2[:, m*sub_len2:(m+1)*sub_len2] = atoms[best]
    recon = (recon1 + recon2).reshape(M, D) * row_scale
    return recon.to(w.dtype)


def encode_m69_pos_spec(w, K):
    """M69: Position-specific scalar quantization (uniform)."""
    M, D = w.shape
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    col_min = w_norm.min(dim=0, keepdim=True)[0]
    col_max = w_norm.max(dim=0, keepdim=True)[0]
    col_range = (col_max - col_min).clamp_min(1e-8)
    scaled = (w_norm - col_min) / col_range * (K - 1)
    indices = torch.round(scaled).clamp(0, K - 1)
    recon_norm = indices.float() / (K - 1) * col_range + col_min
    recon = recon_norm * row_scale
    return recon.to(w.dtype)


def main():
    print(f"Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    
    # Find layer 40
    target_param = None
    for name, p in model.named_parameters():
        if LAYER_NAME in name and len(p.shape) == 2:
            target_param = p
            break
    assert target_param is not None
    original_weight = target_param.data.clone()
    
    # Baseline PPL
    print("\n=== BASELINE PPL ===")
    t0 = time.time()
    baseline_ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}  ({n_steps} steps, {ntok} tokens, {time.time()-t0:.1f}s)")
    
    methods = [
        ("M65 T=8 (vector VQ, worst)", lambda w: encode_m65_tile(w, 8)),
        ("M66 T=8,M=8 (PQ, best)", lambda w: encode_m66_pq(w, 8, 8)),
        ("M67 two-tier T=8,M=4+4", lambda w: encode_m67_twotier(w, 8, 4, 4)),
        ("M69 K=128 (pos-spec, SUSPECT)", lambda w: encode_m69_pos_spec(w, 128)),
        ("M69 K=256 (pos-spec, OK)", lambda w: encode_m69_pos_spec(w, 256)),
    ]
    
    print(f"\n{'='*70}")
    print(f"{'Method':<40} | {'PPL':>8} | {'Delta':>8} | {'Status':>8}")
    print(f"{'-'*70}")
    
    for label, encode_fn in methods:
        # Encode layer 40
        t0 = time.time()
        w = target_param.data.float()
        recon = encode_fn(w)
        target_param.data.copy_(recon)
        print(f"  Encoded in {time.time()-t0:.1f}s")
        
        # PPL
        t0 = time.time()
        ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
        delta = ppl - baseline_ppl
        status = "PASS" if delta < 0.05 else ("DEGRADE" if delta < 0.5 else "FAIL")
        print(f"{label:<40} | {ppl:>8.4f} | {delta:>+8.4f} | {status:>8}")
        print(f"  Inference: {time.time()-t0:.1f}s")
        
        # Restore
        target_param.data.copy_(original_weight)
        torch.cuda.empty_cache()
    
    print(f"{'='*70}")
    print("\nM71 complete.")


if __name__ == "__main__":
    main()
