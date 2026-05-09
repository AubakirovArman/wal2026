"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M155 v2 — Partial Model Transform-WAL PPL Gate (fast).

Tests PPL when only subset of layers use Transform-WAL.
"""
import torch, math, json, sys, gc
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
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


def encode_hadamard(w, K=64, C=8):
    w_h = apply_hadamard(w.float())
    atoms = build_l0_atoms(w_h.reshape(-1), K=K, iters=1)
    coeffs = build_coeff_table(w_h.reshape(-1), atoms, C=C, iters=1)
    _, recon = wal_encode_v1(w_h.reshape(-1), atoms, coeffs, batch=65_536)
    recon = recon.reshape(w_h.shape)
    # Inverse
    op = 1 << max(0, math.ceil(math.log2(w_h.shape[0]))) if w_h.shape[0] > 1 else 1
    ip = 1 << max(0, math.ceil(math.log2(w_h.shape[1]))) if w_h.shape[1] > 1 else 1
    H_out = (hadamard_matrix(op).to(w.device, w.dtype) / math.sqrt(op))
    H_in = (hadamard_matrix(ip).to(w.device, w.dtype) / math.sqrt(ip))
    recon_inv = H_out.T.float() @ recon.float() @ H_in.float()
    return recon_inv[:w.shape[0], :w.shape[1]].to(w.dtype)


def measure_ppl(model, tokenizer, device, max_length=256):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"][:20])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    loss = out.loss.item()
    return math.exp(loss)


def main():
    print("=" * 60)
    print("M155 v2 — Partial Model Transform-WAL PPL Gate")
    print("=" * 60)
    
    device = "cuda:0"
    print(f"\nLoading model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    n_layers = len(model.model.layers)
    
    # Precompute WAL-decoded versions for all layers/modules
    print("\nPrecomputing Transform-WAL for all layers...")
    decoded = {}
    for li in range(n_layers):
        layer = model.model.layers[li]
        for m in modules:
            w = getattr(layer.self_attn if m in ('q_proj','k_proj','v_proj','o_proj') else layer.mlp, m).weight.data
            decoded[(li, m)] = encode_hadamard(w)
        print(f"  Layer {li}/{n_layers} done")
    
    # Save original weights
    originals = {}
    for li in range(n_layers):
        layer = model.model.layers[li]
        for m in modules:
            mod = getattr(layer.self_attn, m) if m in ('q_proj','k_proj','v_proj','o_proj') else getattr(layer.mlp, m)
            originals[(li, m)] = mod.weight.data.clone()
    
    # Baseline PPL
    print("\n--- Baseline PPL ---")
    base_ppl = measure_ppl(model, tokenizer, device, max_length=128)
    print(f"  Base PPL = {base_ppl:.4f}")
    
    # Test partial WAL
    results = []
    for N in [0, 8, 16, 24, 31]:
        print(f"\n--- Replacing layers 0..{N} with Transform-WAL ---")
        # Restore originals
        for li in range(n_layers):
            layer = model.model.layers[li]
            for m in modules:
                mod = getattr(layer.self_attn, m) if m in ('q_proj','k_proj','v_proj','o_proj') else getattr(layer.mlp, m)
                mod.weight.data = originals[(li, m)].clone()
        
        # Replace 0..N
        for li in range(N + 1):
            layer = model.model.layers[li]
            for m in modules:
                mod = getattr(layer.self_attn, m) if m in ('q_proj','k_proj','v_proj','o_proj') else getattr(layer.mlp, m)
                mod.weight.data = decoded[(li, m)].clone()
        
        ppl = measure_ppl(model, tokenizer, device, max_length=128)
        delta = ppl - base_ppl
        print(f"  PPL = {ppl:.4f} (Δ = {delta:+.4f})")
        results.append({'N': N, 'ppl': ppl, 'delta': delta})
    
    print(f"\n{'N':>5} {'PPL':>10} {'Δ':>10}")
    print("-" * 30)
    for r in results:
        print(f"{r['N']:>5} {r['ppl']:>10.4f} {r['delta']:>+10.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m155_partial_model_transform_ppl.json', 'w') as f:
        json.dump({'base_ppl': base_ppl, 'results': results}, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
