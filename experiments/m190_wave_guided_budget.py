#!/usr/bin/env python3
"""M190 — Wave-Guided WAL Budget.

Allocates K per layer based on wave risk: low→K=128, medium→K=256, high→K=512.
"""
import torch, math, json, sys, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def compute_wave_risk(w):
    """Compute wave risk score for a weight matrix."""
    d = w.float()
    m, n = d.shape
    
    # DCT spectrum
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    d_pad = torch.zeros(mp, np, dtype=torch.float32, device=d.device)
    d_pad[:m, :n] = d
    dct = torch.fft.fft2(d_pad).abs()
    dct_flat = dct.reshape(-1)
    total = dct_flat.sum().item()
    top10 = dct_flat.sort(descending=True).values[:max(1, len(dct_flat)//10)].sum().item() / total
    
    # Spectral entropy
    probs = dct_flat / dct_flat.sum()
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    # Singular values
    try:
        sv = torch.linalg.svdvals(d)
        sv_sum = sv.sum().item()
        sv_top10 = (sv[:max(1, len(sv)//10)].sum() / sv_sum).item() if sv_sum > 0 else 0
        spec_norm = sv[0].item()
    except Exception:
        sv_top10 = spec_norm = 0.0
    
    # Risk formula
    risk = top10 * 2.0 + sv_top10 * 2.0 + spec_norm * 0.001 - entropy * 0.1
    return max(0, risk)


def encode_layer_weights(model, layer_idx, K_dict, C=16):
    """Encode all linear weights in a layer with per-module K."""
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
        K = K_dict.get(name, 256)
        w = mod.weight.data
        atoms = build_l0_atoms(w.reshape(-1), K=K, iters=3)
        coeffs = build_coeff_table(w.reshape(-1), atoms, C=C, iters=3)
        _, recon = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=262_144)
        mod.weight.data = recon.reshape(w.shape).to(w.dtype)


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
    print("M190 — Wave-Guided WAL Budget")
    print("=" * 60)
    
    device = "cuda:0"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    n_layers = len(model.model.layers)
    
    # Save originals
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
    
    # Phase 1: Compute wave risk for all layers/modules
    print("\n--- Computing wave risk for all layers ---")
    risks = {}
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            risk = compute_wave_risk(mod.weight.data)
            risks[(li, name)] = risk
    
    # Determine K per module based on risk percentiles
    all_risks = list(risks.values())
    p33 = torch.tensor(all_risks, dtype=torch.float32).quantile(0.33).item()
    p67 = torch.tensor(all_risks, dtype=torch.float32).quantile(0.67).item()
    
    K_dict = {}
    for (li, name), risk in risks.items():
        if risk < p33:
            K_dict[(li, name)] = 128
        elif risk < p67:
            K_dict[(li, name)] = 256
        else:
            K_dict[(li, name)] = 512
    
    count_128 = sum(1 for k in K_dict.values() if k == 128)
    count_256 = sum(1 for k in K_dict.values() if k == 256)
    count_512 = sum(1 for k in K_dict.values() if k == 512)
    print(f"\nBudget distribution: K=128: {count_128}, K=256: {count_256}, K=512: {count_512}")
    
    # Phase 2: Encode with adaptive K
    print("\n--- Encoding with adaptive K ---")
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
        
        layer_K = {name: K_dict[(li, name)] for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
        encode_layer_weights(model, li, layer_K, C=16)
        
        if li % 8 == 0:
            print(f"  Layer {li}/{n_layers} done")
    
    # Measure PPL
    ppl = measure_ppl(model, tokenizer, device, max_length=128)
    print(f"\nAdaptive K PPL = {ppl:.4f} (Δ = {ppl-base_ppl:+.4f})")
    
    # Compare to uniform K=256
    print("\n--- Uniform K=256 (for comparison) ---")
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
    
    for li in range(n_layers):
        encode_layer_weights(model, li, {name: 256 for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}, C=16)
        if li % 8 == 0:
            print(f"  Layer {li}/{n_layers} done")
    
    ppl_uniform = measure_ppl(model, tokenizer, device, max_length=128)
    print(f"\nUniform K=256 PPL = {ppl_uniform:.4f} (Δ = {ppl_uniform-base_ppl:+.4f})")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>20} {'PPL':>10} {'Δ':>10}")
    print("-" * 45)
    print(f"{'Baseline':>20} {base_ppl:>10.4f} {'—':>10}")
    print(f"{'Adaptive K':>20} {ppl:>10.4f} {ppl-base_ppl:>+10.4f}")
    print(f"{'Uniform K=256':>20} {ppl_uniform:>10.4f} {ppl_uniform-base_ppl:>+10.4f}")
    
    # Compute theoretical size
    atom_size_128 = 128 * 4  # bytes
    atom_size_256 = 256 * 4
    atom_size_512 = 512 * 4
    total_atoms_adaptive = count_128 * atom_size_128 + count_256 * atom_size_256 + count_512 * atom_size_512
    total_atoms_uniform = len(K_dict) * atom_size_256
    print(f"\nAtom table size: adaptive={total_atoms_adaptive/1024:.1f}KB, uniform={total_atoms_uniform/1024:.1f}KB")
    
    results = {
        'base_ppl': base_ppl,
        'adaptive_ppl': ppl,
        'uniform_ppl': ppl_uniform,
        'budget_dist': {'128': count_128, '256': count_256, '512': count_512},
        'size_kb': {'adaptive': total_atoms_adaptive/1024, 'uniform': total_atoms_uniform/1024},
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m190_wave_guided_budget.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
