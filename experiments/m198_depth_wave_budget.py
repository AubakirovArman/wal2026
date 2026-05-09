#!/usr/bin/env python3
"""M198 — Depth-Wave Budget: K assignment by layer depth.

Hypothesis: Later layers are more sensitive (M186 showed wave growth by depth).
Assign K by depth instead of risk:
- Early layers (0-10): K=128
- Mid layers (11-20): K=256
- Late layers (21-31): K=512

Compare with uniform K=256 and risk-based adaptive K.
"""
import torch, math, json, sys, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:5"

def hadamard_transform_1d(x):
    n = x.numel()
    m = 1 << (n - 1).bit_length()
    if m != n:
        x = torch.cat([x.flatten(), torch.zeros(m - n, device=x.device, dtype=x.dtype)])
    else:
        x = x.flatten()
    h = 1
    while h < m:
        x = x.view(m // (2 * h), 2 * h)
        x = torch.cat([x[:, :h] + x[:, h:], x[:, :h] - x[:, h:]], dim=1)
        h *= 2
    return x.flatten() / math.sqrt(m), (n, m)

def inverse_hadamard_1d(y, orig_info):
    n, m = orig_info
    y = hadamard_transform_1d(y)[0]
    return y[:n]

def hadamard_transform_2d(w):
    out, orig = [], []
    for row in w:
        h, info = hadamard_transform_1d(row)
        out.append(h)
        orig.append(info)
    return torch.stack(out), orig

def inverse_hadamard_2d(h, orig_infos):
    out = []
    for row, info in zip(h, orig_infos):
        out.append(inverse_hadamard_1d(row, info))
    return torch.stack(out)

def kmeans_chunked(data, K, iters=3, chunk_size=1_000_000):
    device = data.device
    sample = data[torch.randperm(data.numel(), device=device)[:min(100000, data.numel())]]
    atoms = [sample[torch.randint(0, len(sample), (1,), device=device)].item()]
    for _ in range(1, K):
        dists = torch.stack([torch.abs(sample - a) for a in atoms], dim=0).min(dim=0).values
        probs = dists / (dists.sum() + 1e-10)
        idx = torch.multinomial(probs, 1)
        atoms.append(sample[idx].item())
    atoms = torch.tensor(atoms, device=device, dtype=data.dtype)
    for _ in range(iters):
        new_sums = torch.zeros(K, device=device, dtype=torch.float64)
        counts = torch.zeros(K, device=device, dtype=torch.float64)
        for i in range(0, data.numel(), chunk_size):
            chunk = data[i:i+chunk_size]
            dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
            labels = dists.argmin(dim=1)
            new_sums.scatter_add_(0, labels, chunk.double())
            counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        new_atoms = torch.where(counts > 0, (new_sums / counts).to(data.dtype), atoms)
        if torch.allclose(atoms, new_atoms, atol=1e-6):
            break
        atoms = new_atoms
    labels_all = []
    for i in range(0, data.numel(), chunk_size):
        chunk = data[i:i+chunk_size]
        dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
        labels_all.append(dists.argmin(dim=1))
    labels = torch.cat(labels_all)
    quantized = atoms[labels].reshape(data.shape)
    return quantized, atoms, labels

def hadamard_wal_encode(w, K, iters=3):
    h, orig_info = hadamard_transform_2d(w.float().cuda())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec.to(w.device, w.dtype)

def compute_depth_k_map(model):
    """Assign K by layer depth."""
    k_map = {}
    for name, p in model.named_parameters():
        if 'weight' not in name or len(p.shape) != 2 or p.shape[0] <= 100:
            continue
        # Extract layer number from name like model.layers.15.self_attn.q_proj.weight
        parts = name.split('.')
        layer_idx = -1
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        if layer_idx < 0:
            continue
        if layer_idx <= 10:
            k_map[name] = 128
        elif layer_idx <= 20:
            k_map[name] = 256
        else:
            k_map[name] = 512
    return k_map

def compute_risk_k_map(model):
    """Risk-based adaptive K (from M195b+)."""
    risks, names = [], []
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and p.shape[0] > 100 and p.shape[0] < 50000:
            u, s, v = torch.linalg.svd(p.float(), full_matrices=False)
            risks.append(s[0].item())
            names.append(name)
    risks_t = torch.tensor(risks)
    p25, p75 = risks_t.quantile(0.25), risks_t.quantile(0.75)
    k_map = {}
    for name, r in zip(names, risks):
        if r < p25:
            k_map[name] = 128
        elif r > p75:
            k_map[name] = 512
        else:
            k_map[name] = 256
    return k_map

def encode_model(model, k_map, iters=3):
    total = len([n for n in k_map if n in dict(model.named_parameters())])
    done = 0
    for name, p in model.named_parameters():
        if name in k_map and len(p.shape) == 2 and p.shape[0] > 100 and p.shape[0] < 50000:
            w_rec = hadamard_wal_encode(p.data, k_map[name], iters=iters)
            p.data.copy_(w_rec)
            done += 1
            if done % 10 == 0:
                print(f"    Encoded {done}/{total} modules...", flush=True)

def eval_ppl(model, tokenizer, device, max_samples=20):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [ex for ex in ds['text'][:100] if len(ex) > 50][:max_samples]
    total_nll = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            total_nll += out.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
    return math.exp(total_nll / total_tokens)

def main():
    print("=" * 60, flush=True)
    print("M198 — Depth-Wave Budget", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline
    print("\nLoading model for baseline...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    baseline_ppl = eval_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()

    # Uniform K=256
    print("\n--- Uniform K=256 ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    k_map_uniform = {n: 256 for n, p in model.named_parameters() if 'weight' in n and len(p.shape) == 2 and p.shape[0] > 100}
    start = time.time()
    encode_model(model, k_map_uniform, iters=3)
    uniform_time = time.time() - start
    uniform_ppl = eval_ppl(model, tokenizer, device)
    print(f"Uniform K=256 PPL: {uniform_ppl:.4f} (Δ={uniform_ppl-baseline_ppl:+.4f}), Time: {uniform_time:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    # Risk-based adaptive K
    print("\n--- Risk-based Adaptive K ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    k_map_risk = compute_risk_k_map(model)
    risk_counts = {128: 0, 256: 0, 512: 0}
    for k in k_map_risk.values():
        risk_counts[k] = risk_counts.get(k, 0) + 1
    print(f"Risk K distribution: {risk_counts}", flush=True)
    start = time.time()
    encode_model(model, k_map_risk, iters=3)
    risk_time = time.time() - start
    risk_ppl = eval_ppl(model, tokenizer, device)
    print(f"Risk Adaptive PPL: {risk_ppl:.4f} (Δ={risk_ppl-baseline_ppl:+.4f}), Time: {risk_time:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    # Depth-based K
    print("\n--- Depth-based K ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    k_map_depth = compute_depth_k_map(model)
    depth_counts = {128: 0, 256: 0, 512: 0}
    for k in k_map_depth.values():
        depth_counts[k] = depth_counts.get(k, 0) + 1
    print(f"Depth K distribution: {depth_counts}", flush=True)
    start = time.time()
    encode_model(model, k_map_depth, iters=3)
    depth_time = time.time() - start
    depth_ppl = eval_ppl(model, tokenizer, device)
    print(f"Depth Adaptive PPL: {depth_ppl:.4f} (Δ={depth_ppl-baseline_ppl:+.4f}), Time: {depth_time:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("M198 — Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Method':<25} {'PPL':>10} {'Δ':>10} {'Time':>10}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'Baseline':<25} {baseline_ppl:>10.4f} {'—':>10} {'—':>10}", flush=True)
    print(f"{'Uniform K=256':<25} {uniform_ppl:>10.4f} {uniform_ppl-baseline_ppl:>+10.4f} {uniform_time:>9.1f}s", flush=True)
    print(f"{'Risk Adaptive':<25} {risk_ppl:>10.4f} {risk_ppl-baseline_ppl:>+10.4f} {risk_time:>9.1f}s", flush=True)
    print(f"{'Depth Adaptive':<25} {depth_ppl:>10.4f} {depth_ppl-baseline_ppl:>+10.4f} {depth_time:>9.1f}s", flush=True)

    result = {
        "experiment": "M198",
        "baseline_ppl": baseline_ppl,
        "uniform_ppl": uniform_ppl,
        "risk_ppl": risk_ppl,
        "depth_ppl": depth_ppl,
        "uniform_time": uniform_time,
        "risk_time": risk_time,
        "depth_time": depth_time,
    }
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m198_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Saved to experiments/m198_results.json", flush=True)

if __name__ == "__main__":
    main()
