"""
M243 — Encode Seed Determinism

Hypothesis: M239 showed encode is non-deterministic because kmeans
uses unseeded random operations. Fixing the seed will make encode
bit-exact reproducible.

Fix: Add torch.manual_seed to kmeans_chunked and encode_model.
Test: Same seed → identical outputs.
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
    print("M243 — Encode Seed Determinism")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_text = "The capital of France is"
    
    # Run 1
    print("\n[Run 1] Encoding with seed=42...")
    model1 = load_model()[0]
    encode_model(model1, K=K, iters=ITERS, seed=SEED)
    out1 = get_outputs(model1, tokenizer, test_text, DEVICE)
    del model1
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run 2
    print("[Run 2] Re-encoding with seed=42...")
    model2 = load_model()[0]
    encode_model(model2, K=K, iters=ITERS, seed=SEED)
    out2 = get_outputs(model2, tokenizer, test_text, DEVICE)
    del model2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compare
    logits_match = torch.allclose(out1["logits"], out2["logits"], atol=1e-4)
    hidden_match = torch.allclose(out1["hidden"], out2["hidden"], atol=1e-4)
    max_logit_diff = (out1["logits"] - out2["logits"]).abs().max().item()
    max_hidden_diff = (out1["hidden"] - out2["hidden"]).abs().max().item()
    
    print(f"\nLogits match: {logits_match} (max diff: {max_logit_diff:.6e})")
    print(f"Hidden match: {hidden_match} (max diff: {max_hidden_diff:.6e})")
    
    output = {
        "logits_match": logits_match,
        "hidden_match": hidden_match,
        "max_logit_diff": max_logit_diff,
        "max_hidden_diff": max_hidden_diff,
    }
    with open("experiments/m243_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved to experiments/m243_results.json")
    
    if logits_match and hidden_match:
        print("\n✅ SEED FIX WORKS — encode is deterministic with fixed seed")
    else:
        print("\n❌ SEED FIX INSUFFICIENT — non-seed randomness remains")

if __name__ == "__main__":
    run()
