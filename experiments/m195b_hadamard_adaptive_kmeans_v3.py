#!/usr/bin/env python3
"""M195b v3 — Hadamard Adaptive K + chunked torch k-means (GPU).

Avoids sklearn/OpenBLAS issues by using pure torch chunked k-means on GPU.
"""
import torch, math, json, sys, time, gc
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from datasets import load_dataset


def hadamard_transform_1d(x):
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
    return hadamard_transform_1d(x)


def hadamard_transform_2d(w):
    orig_shape = w.shape
    out_d, in_d = orig_shape
    pad_out = 1 << (out_d - 1).bit_length()
    pad_in = 1 << (in_d - 1).bit_length()
    padded = torch.zeros(pad_out, pad_in, device=w.device, dtype=w.dtype)
    padded[:out_d, :in_d] = w
    h = hadamard_transform_1d(padded)
    h = hadamard_transform_1d(h.T).T
    return h, (out_d, in_d, pad_out, pad_in)


def inverse_hadamard_2d(h, orig_info):
    out_d, in_d, pad_out, pad_in = orig_info
    x = inverse_hadamard_1d(h)
    x = inverse_hadamard_1d(x.T).T
    return x[:out_d, :in_d]


def kmeans_chunked(data, K, iters=3, chunk_size=1_000_000):
    """Chunked k-means on GPU. data: 1D tensor."""
    device = data.device
    
    # K-means++ init on sample
    sample = data[torch.randperm(data.numel(), device=device)[:min(100000, data.numel())]]
    atoms = [sample[torch.randint(0, len(sample), (1,), device=device)].item()]
    for _ in range(1, K):
        dists = torch.stack([torch.abs(sample - a) for a in atoms], dim=0).min(dim=0).values
        probs = dists / (dists.sum() + 1e-10)
        idx = torch.multinomial(probs, 1)
        atoms.append(sample[idx].item())
    atoms = torch.tensor(atoms, device=device, dtype=data.dtype)
    
    # Lloyd iterations
    for _ in range(iters):
        new_sums = torch.zeros(K, device=device, dtype=torch.float64)
        counts = torch.zeros(K, device=device, dtype=torch.float64)
        
        for i in range(0, data.numel(), chunk_size):
            chunk = data[i:i+chunk_size]
            dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))  # [chunk, K]
            labels = dists.argmin(dim=1)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    new_sums[k] += chunk[mask].sum().double()
                    counts[k] += mask.sum().double()
        
        new_atoms = torch.where(counts > 0, (new_sums / counts).to(data.dtype), atoms)
        if torch.allclose(atoms, new_atoms, atol=1e-6):
            break
        atoms = new_atoms
    
    # Assign all
    labels_all = []
    for i in range(0, data.numel(), chunk_size):
        chunk = data[i:i+chunk_size]
        dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
        labels_all.append(dists.argmin(dim=1))
    labels = torch.cat(labels_all)
    
    quantized = atoms[labels].reshape(data.shape)
    return quantized, atoms, labels


def hadamard_wal_encode(w, K):
    h, orig_info = hadamard_transform_2d(w.float())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K)
    quantized = quantized.reshape(h.shape)
    return quantized, atoms, orig_info, indices


def hadamard_wal_decode(quantized, orig_info):
    return inverse_hadamard_2d(quantized, orig_info)


def compute_risk(w):
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
    print("M195b v3 — Hadamard Adaptive K + chunked torch k-means")
    print("=" * 60)
    
    device = "cuda:0"
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
    
    # Compute risk
    print("\n--- Computing wave risk ---")
    module_info = []
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            risk = compute_risk(mod.weight.data)
            module_info.append((li, name, mod, risk))
    
    sorted_modules = sorted(module_info, key=lambda x: x[3])
    n_mods = len(sorted_modules)
    
    k_assignments = {}
    for i, (li, name, mod, risk) in enumerate(sorted_modules):
        p = i / n_mods
        k_assignments[(li, name)] = 128 if p < 0.3 else (256 if p < 0.8 else 512)
    
    print(f"\nK distribution:")
    k_counts = {}
    for k in k_assignments.values():
        k_counts[k] = k_counts.get(k, 0) + 1
    for k, c in sorted(k_counts.items()):
        print(f"  K={k}: {c} modules")
    
    # Baseline
    print("\n--- Baseline ---")
    baseline_ppl = compute_ppl(model, tokenizer, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    
    # Uniform K=256
    print("\n--- Uniform K=256 ---")
    total_bits = 0
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
            total_bits += indices.numel() * 8
    
    uniform_ppl = compute_ppl(model, tokenizer, device)
    uniform_size_mb = total_bits / (8 * 1024 * 1024)
    print(f"  PPL: {uniform_ppl:.4f} (Δ={uniform_ppl-baseline_ppl:+.4f})")
    print(f"  Size: ~{uniform_size_mb:.2f} MB")
    
    # Reload for adaptive
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n--- Adaptive K ---")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    total_bits = 0
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
            bits = 7 if K <= 128 else (8 if K <= 256 else 9)
            total_bits += indices.numel() * bits
    
    adaptive_ppl = compute_ppl(model, tokenizer, device)
    adaptive_size_mb = total_bits / (8 * 1024 * 1024)
    encode_time = time.time() - start
    
    print(f"  PPL: {adaptive_ppl:.4f} (Δ={adaptive_ppl-baseline_ppl:+.4f})")
    print(f"  Size: ~{adaptive_size_mb:.2f} MB")
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
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m195b_hadamard_adaptive_kmeans_v3.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
