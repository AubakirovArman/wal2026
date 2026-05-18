#!/usr/bin/env python3
"""M181 — High-K Transform-WAL PPL Gate.

Tests PPL with K=256 Transform-WAL on full model.
Compares: Raw, Hadamard.
"""
import torch, math, json, sys, time
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


def encode_layer_weights(model, layer_idx, transform, K=256, C=16):
    """Encode all linear weights in a layer."""
    layer = model.model.layers[layer_idx]
    modules = [
        ('q_proj', layer.self_attn.q_proj),
        ('k_proj', layer.self_attn.k_proj),
        ('v_proj', layer.self_attn.v_proj),
        ('o_proj', layer.self_attn.o_proj),
        ('gate_proj', layer.mlp.gate_proj),
        ('up_proj', layer.mlp.up_proj),
        ('down_proj', layer.mlp.down_proj),
    ]
    
    for name, mod in modules:
        w = mod.weight.data
        if transform == "hadamard":
            w_t = apply_hadamard(w)
        else:
            w_t = w
        
        atoms = build_l0_atoms(w_t.reshape(-1), K=K, iters=3)
        coeffs = build_coeff_table(w_t.reshape(-1), atoms, C=C, iters=3)
        _, recon = wal_encode_v1(w_t.reshape(-1), atoms, coeffs, batch=262_144)
        
        if transform == "hadamard":
            # Inverse Hadamard
            op = 1 << max(0, math.ceil(math.log2(w_t.shape[0]))) if w_t.shape[0] > 1 else 1
            ip = 1 << max(0, math.ceil(math.log2(w_t.shape[1]))) if w_t.shape[1] > 1 else 1
            H_out = (hadamard_matrix(op).to(w.device, w.dtype) / math.sqrt(op))
            H_in = (hadamard_matrix(ip).to(w.device, w.dtype) / math.sqrt(ip))
            recon_inv = H_out.T.float() @ recon.float().reshape(op, ip) @ H_in.float()
            recon = recon_inv[:w.shape[0], :w.shape[1]].to(w.dtype)
        else:
            recon = recon.reshape(w.shape)
        
        mod.weight.data = recon.to(mod.weight.dtype)


def measure_ppl(model, tokenizer, device, max_length=128):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"][:20])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(out.loss.item())


def main():
    print("=" * 60)
    print("M181 — High-K Transform-WAL PPL Gate")
    print("=" * 60)
    
    device = "cuda:3"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    n_layers = len(model.model.layers)
    
    # Save original weights
    originals = {}
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            originals[(li, name)] = mod.weight.data.clone()
    
    # Baseline PPL
    print("\n--- Baseline PPL ---")
    base_ppl = measure_ppl(model, tokenizer, device, max_length=128)
    print(f"  Base PPL = {base_ppl:.4f}")
    
    results = {'base_ppl': base_ppl, 'tests': []}
    
    # Test K=256 Raw
    print("\n--- K=256 Raw (all layers) ---")
    for li in range(n_layers):
        for name, mod in [
            ('q_proj', model.model.layers[li].self_attn.q_proj),
            ('k_proj', model.model.layers[li].self_attn.k_proj),
            ('v_proj', model.model.layers[li].self_attn.v_proj),
            ('o_proj', model.model.layers[li].self_attn.o_proj),
            ('gate_proj', model.model.layers[li].mlp.gate_proj),
            ('up_proj', model.model.layers[li].mlp.up_proj),
            ('down_proj', model.model.layers[li].mlp.down_proj),
        ]:
            mod.weight.data = originals[(li, name)].clone()
    
    start = time.time()
    for li in range(n_layers):
        encode_layer_weights(model, li, "raw", K=256, C=16)
        if li % 8 == 0:
            print(f"  Layer {li}/{n_layers} done")
    print(f"  Encode time: {time.time()-start:.1f}s")
    
    ppl = measure_ppl(model, tokenizer, device, max_length=128)
    print(f"  PPL = {ppl:.4f} (Δ = {ppl-base_ppl:+.4f})")
    results['tests'].append({'name': 'K256_Raw', 'ppl': ppl, 'delta': ppl-base_ppl})
    
    # Test K=256 Hadamard
    print("\n--- K=256 Hadamard (all layers) ---")
    for li in range(n_layers):
        for name, mod in [
            ('q_proj', model.model.layers[li].self_attn.q_proj),
            ('k_proj', model.model.layers[li].self_attn.k_proj),
            ('v_proj', model.model.layers[li].self_attn.v_proj),
            ('o_proj', model.model.layers[li].self_attn.o_proj),
            ('gate_proj', model.model.layers[li].mlp.gate_proj),
            ('up_proj', model.model.layers[li].mlp.up_proj),
            ('down_proj', model.model.layers[li].mlp.down_proj),
        ]:
            mod.weight.data = originals[(li, name)].clone()
    
    start = time.time()
    for li in range(n_layers):
        encode_layer_weights(model, li, "hadamard", K=256, C=16)
        if li % 8 == 0:
            print(f"  Layer {li}/{n_layers} done")
    print(f"  Encode time: {time.time()-start:.1f}s")
    
    ppl = measure_ppl(model, tokenizer, device, max_length=128)
    print(f"  PPL = {ppl:.4f} (Δ = {ppl-base_ppl:+.4f})")
    results['tests'].append({'name': 'K256_Hadamard', 'ppl': ppl, 'delta': ppl-base_ppl})
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Test':>20} {'PPL':>10} {'Δ':>10}")
    print("-" * 45)
    print(f"{'Baseline':>20} {base_ppl:>10.4f} {'—':>10}")
    for t in results['tests']:
        print(f"{t['name']:>20} {t['ppl']:>10.4f} {t['delta']:>+10.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m181_high_k_ppl_gate.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
