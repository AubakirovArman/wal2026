"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M253 — Deterministic Build Full Audit

Hypothesis: M243 showed bit-exact determinism for one run.
This test validates across 3 independent load-encode cycles
with fixed seed=42.
"""

import os, sys, json, torch, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K = 256
ITERS = 3
SEED = 42

def hadamard_transform_2d(w):
    n = w.shape[-1]
    m = 1 << (n - 1).bit_length()
    orig_info = (n, m)
    if m != n:
        pad = torch.zeros(w.shape[0], m - n, device=w.device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=-1)
    h = 1
    while h < m:
        w = w.reshape(w.shape[0], m // (2 * h), 2, h)
        w = torch.cat([w[:, :, 0, :] + w[:, :, 1, :], w[:, :, 0, :] - w[:, :, 1, :]], dim=-1)
        h *= 2
    return w.reshape(w.shape[0], m) / math.sqrt(m), orig_info

def inverse_hadamard_2d(h, orig_info=None):
    n = h.shape[-1]
    m = 1 << (n - 1).bit_length()
    if m != n:
        pad = torch.zeros(h.shape[0], m - n, device=h.device, dtype=h.dtype)
        h = torch.cat([h, pad], dim=-1)
    hh = 1
    while hh < m:
        h = h.reshape(h.shape[0], m // (2 * hh), 2, hh)
        h = torch.cat([h[:, :, 0, :] + h[:, :, 1, :], h[:, :, 0, :] - h[:, :, 1, :]], dim=-1)
        hh *= 2
    result = h.reshape(h.shape[0], m) / math.sqrt(m)
    if orig_info is not None:
        n_orig, _ = orig_info
        result = result[:, :n_orig]
    return result

def kmeans_chunked(data, K, iters=3, chunk_size=1_000_000, seed=42):
    torch.manual_seed(seed)
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

def hadamard_wal_encode(w, K=256, iters=3, seed=42):
    torch.manual_seed(seed)
    h, orig_info = hadamard_transform_2d(w)
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters, seed=seed)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec

def encode_model(model, K=256, iters=3, seed=42):
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters, seed=seed)
    return model

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    return model, tokenizer

def get_outputs(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return {
        "logits": out.logits[0, -1, :].cpu(),
        "hidden": out.hidden_states[-1][0, -1, :].cpu(),
    }

def run():
    print("=" * 60)
    print("M253 — Deterministic Build Full Audit")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_texts = [
        "The capital of France is",
        "In 1492, Christopher Columbus",
        "The theory of relativity was developed by",
    ]
    
    runs = []
    for run_idx in range(3):
        print(f"\n[Run {run_idx+1}] Loading and encoding with seed={SEED}...")
        model = load_model()[0]
        encode_model(model, K=K, iters=ITERS, seed=SEED)
        
        outputs = []
        for text in test_texts:
            out = get_outputs(model, tokenizer, text, DEVICE)
            outputs.append(out)
        runs.append(outputs)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS 3 RUNS")
    print("=" * 60)
    
    all_match = True
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: {text[:40]}...")
        
        # Compare Run 1 vs Run 2
        r1_r2_logits = torch.allclose(runs[0][i]["logits"], runs[1][i]["logits"], atol=1e-4)
        r1_r2_hidden = torch.allclose(runs[0][i]["hidden"], runs[1][i]["hidden"], atol=1e-4)
        max_logit_diff_12 = (runs[0][i]["logits"] - runs[1][i]["logits"]).abs().max().item()
        max_hidden_diff_12 = (runs[0][i]["hidden"] - runs[1][i]["hidden"]).abs().max().item()
        
        # Compare Run 1 vs Run 3
        r1_r3_logits = torch.allclose(runs[0][i]["logits"], runs[2][i]["logits"], atol=1e-4)
        r1_r3_hidden = torch.allclose(runs[0][i]["hidden"], runs[2][i]["hidden"], atol=1e-4)
        max_logit_diff_13 = (runs[0][i]["logits"] - runs[2][i]["logits"]).abs().max().item()
        max_hidden_diff_13 = (runs[0][i]["hidden"] - runs[2][i]["hidden"]).abs().max().item()
        
        print(f"  Run1 vs Run2: logits={r1_r2_logits} (max diff: {max_logit_diff_12:.6e}), hidden={r1_r2_hidden} (max diff: {max_hidden_diff_12:.6e})")
        print(f"  Run1 vs Run3: logits={r1_r3_logits} (max diff: {max_logit_diff_13:.6e}), hidden={r1_r3_hidden} (max diff: {max_hidden_diff_13:.6e})")
        
        if not (r1_r2_logits and r1_r2_hidden and r1_r3_logits and r1_r3_hidden):
            all_match = False
    
    output = {
        "seed": SEED,
        "all_match": all_match,
        "num_runs": 3,
    }
    with open("experiments/m253_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved to experiments/m253_results.json")
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ BUILD IS DETERMINISTIC across 3 independent runs")
    else:
        print("❌ BUILD IS NOT DETERMINISTIC — non-seed randomness remains")

if __name__ == "__main__":
    run()
