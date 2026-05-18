#!/usr/bin/env python3
"""M195 — Hadamard Wave-Guided Budget v2.

Adaptive K allocation using Hadamard-WAL based on wave risk.
Improvements over M190:
- Hadamard transform (not raw-WAL)
- Percentile-based policy
- Module-type awareness
"""
import torch, math, json, sys, time, gc
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from datasets import load_dataset


def hadamard_transform_1d(x):
    """Fast Walsh-Hadamard transform (1D, power-of-2)."""
    n = x.shape[-1]
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        x = torch.stack([x[:, 0, :] + x[:, 1, :], x[:, 0, :] - x[:, 1, :]], dim=1)
        x = x.reshape(-1, 2 * h)
        h *= 2
    return x.reshape(*orig_shape, n) / math.sqrt(n)


def inverse_hadamard_1d(x):
    """Inverse is same as forward for normalized Hadamard."""
    return hadamard_transform_1d(x)


def hadamard_transform_2d(w):
    """Apply Hadamard to rows then columns. Pad to power-of-2."""
    orig_shape = w.shape
    out_d, in_d = orig_shape
    
    # Pad to power of 2
    pad_out = 1 << (out_d - 1).bit_length()
    pad_in = 1 << (in_d - 1).bit_length()
    
    padded = torch.zeros(pad_out, pad_in, device=w.device, dtype=w.dtype)
    padded[:out_d, :in_d] = w
    
    # Row then column transform
    h = hadamard_transform_1d(padded)
    h = hadamard_transform_1d(h.T).T
    
    return h, (out_d, in_d, pad_out, pad_in)


def inverse_hadamard_2d(h, orig_info):
    out_d, in_d, pad_out, pad_in = orig_info
    x = inverse_hadamard_1d(h)
    x = inverse_hadamard_1d(x.T).T
    return x[:out_d, :in_d]


def uniform_quantize(x, K):
    """O(N) uniform quantization (no O(N*K) memory)."""
    min_val = x.min().item()
    max_val = x.max().item()
    step = (max_val - min_val) / (K - 1) if K > 1 else 1.0
    atoms = torch.linspace(min_val, max_val, K, device=x.device, dtype=x.dtype)
    indices = ((x - min_val) / step + 0.5).long().clamp(0, K - 1)
    quantized = atoms[indices]
    return quantized, atoms, indices


def hadamard_wal_encode(w, K):
    """Hadamard-WAL encode: transform → uniform quantize."""
    h, orig_info = hadamard_transform_2d(w.float())
    quantized, atoms, indices = uniform_quantize(h, K)
    return quantized, atoms, orig_info, indices


def hadamard_wal_decode(quantized, orig_info):
    """Decode: inverse Hadamard."""
    return inverse_hadamard_2d(quantized, orig_info)


def compute_risk(w):
    """Compute wave risk for a weight matrix."""
    spec_norm = torch.linalg.matrix_norm(w.float(), ord=2).item()
    frob_norm = torch.linalg.matrix_norm(w.float(), ord='fro').item()
    return spec_norm + frob_norm * 0.01


def compute_ppl(model, tokenizer, device, max_length=512):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([ex['text'] for ex in ds.select(range(min(100, len(ds)))) if len(ex.get('text', '')) > 20])
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    return torch.exp(out.loss).item()


def main():
    print("=" * 60)
    print("M195 — Hadamard Wave-Guided Budget v2")
    print("=" * 60)
    
    device = "cuda:3"
    print(f"\nDevice: {device}")
    
    from transformers import AutoTokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    n_layers = len(model.model.layers)
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    # Compute risk for all modules
    print("\n--- Computing wave risk for all modules ---")
    risks = []
    module_info = []
    
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            w = mod.weight.data
            risk = compute_risk(w)
            risks.append(risk)
            module_info.append((li, name, mod, risk))
    
    # Sort by risk
    sorted_modules = sorted(module_info, key=lambda x: x[3])
    n_mods = len(sorted_modules)
    
    # Assign K based on percentiles
    k_assignments = {}
    for i, (li, name, mod, risk) in enumerate(sorted_modules):
        percentile = i / n_mods
        if percentile < 0.3:
            K = 128
        elif percentile < 0.8:
            K = 256
        else:
            K = 512
        k_assignments[(li, name)] = K
    
    print(f"\nK distribution:")
    k_counts = {}
    for k in k_assignments.values():
        k_counts[k] = k_counts.get(k, 0) + 1
    for k, c in sorted(k_counts.items()):
        print(f"  K={k}: {c} modules")
    
    # Show top risk modules
    print(f"\nTop 10 highest risk modules (K=512):")
    for li, name, mod, risk in sorted_modules[-10:]:
        print(f"  Layer {li:2d} {name:12s}: risk={risk:8.2f}")
    
    # Baseline PPL
    print("\n--- Baseline ---")
    baseline_ppl = compute_ppl(model, tokenizer, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    
    # Uniform K=256
    print("\n--- Uniform K=256 Hadamard-WAL ---")
    total_params = 0
    encoded_params = 0
    
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            w = mod.weight.data
            quantized, atoms, orig_info, indices = hadamard_wal_encode(w, K=256)
            recon = hadamard_wal_decode(quantized, orig_info).to(w.dtype)
            mod.weight.data = recon
            total_params += w.numel()
            encoded_params += indices.numel()  # 1 byte per param for K<=256
    
    uniform_ppl = compute_ppl(model, tokenizer, device)
    uniform_size_mb = encoded_params / (8 * 1024 * 1024)  # 8 bits per index
    print(f"  PPL: {uniform_ppl:.4f} (Δ={uniform_ppl-baseline_ppl:+.4f})")
    print(f"  Size: ~{uniform_size_mb:.2f} MB (indices only)")
    
    # Reload model for adaptive
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n--- Adaptive K Hadamard-WAL ---")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    total_params = 0
    encoded_params = 0
    start = time.time()
    
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            w = mod.weight.data
            K = k_assignments[(li, name)]
            quantized, atoms, orig_info, indices = hadamard_wal_encode(w, K=K)
            recon = hadamard_wal_decode(quantized, orig_info).to(w.dtype)
            mod.weight.data = recon
            total_params += w.numel()
            encoded_params += indices.numel() * (8 if K <= 256 else 9)  # bits per index
    
    adaptive_ppl = compute_ppl(model, tokenizer, device)
    adaptive_size_mb = encoded_params / (8 * 1024 * 1024)
    encode_time = time.time() - start
    
    print(f"  PPL: {adaptive_ppl:.4f} (Δ={adaptive_ppl-baseline_ppl:+.4f})")
    print(f"  Size: ~{adaptive_size_mb:.2f} MB (indices only)")
    print(f"  Encode time: {encode_time:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>20} {'PPL':>10} {'Δ':>10} {'Size MB':>10}")
    print("-" * 55)
    print(f"{'Baseline':>20} {baseline_ppl:>10.4f} {'—':>10} {'—':>10}")
    print(f"{'Uniform K=256':>20} {uniform_ppl:>10.4f} {uniform_ppl-baseline_ppl:>+10.4f} {uniform_size_mb:>10.2f}")
    print(f"{'Adaptive K':>20} {adaptive_ppl:>10.4f} {adaptive_ppl-baseline_ppl:>+10.4f} {adaptive_size_mb:>10.2f}")
    
    results = {
        "baseline_ppl": baseline_ppl,
        "uniform_k256_ppl": uniform_ppl,
        "uniform_k256_size_mb": uniform_size_mb,
        "adaptive_ppl": adaptive_ppl,
        "adaptive_size_mb": adaptive_size_mb,
        "k_distribution": k_counts,
        "encode_time": encode_time,
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m195_hadamard_wave_budget_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
