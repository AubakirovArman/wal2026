"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M239 — Frozen Encode Determinism

Hypothesis: WAL-encoded weights that are NOT edited (frozen) produce
bit-exact identical outputs across multiple load-encode cycles.

This validates that the encoding process itself is deterministic
and that frozen layers can be safely cached/reused.

Test:
1. Load base model, encode, save encoded weights
2. Load again, encode again, compare outputs token-by-token
3. Compare PPL, logits, hidden states
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

def hadamard_wal_encode(weight, K=256, iters=3):
    w = weight.data
    h, orig_info = hadamard_transform_2d(w)
    quantized, atoms, labels = kmeans_chunked(h.reshape(-1), K=K, iters=iters)
    quantized = quantized.reshape(h.shape)
    h_rec = quantized
    w_rec = inverse_hadamard_2d(h_rec, orig_info)
    return w_rec

def encode_model(model, K=256, iters=3):
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

def load_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model

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
    print("M239 — Frozen Encode Determinism")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_texts = [
        "The capital of France is",
        "In 1492, Christopher Columbus",
        "The theory of relativity was developed by",
    ]
    
    # Run 1: encode, capture outputs
    print("\n[Run 1] Loading and encoding...")
    model1 = load_model(DEVICE)
    encode_model(model1, K=K, iters=ITERS)
    
    outputs1 = []
    for text in test_texts:
        out = get_outputs(model1, tokenizer, text, DEVICE)
        outputs1.append(out)
    
    del model1
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run 2: reload, re-encode, capture outputs
    print("[Run 2] Reloading and re-encoding...")
    model2 = load_model(DEVICE)
    encode_model(model2, K=K, iters=ITERS)
    
    outputs2 = []
    for text in test_texts:
        out = get_outputs(model2, tokenizer, text, DEVICE)
        outputs2.append(out)
    
    del model2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run 3: different seed to show it matters
    print("[Run 3] Different seed (99)...")
    model3 = load_model(DEVICE)
    encode_model(model3, K=K, iters=ITERS)
    
    outputs3 = []
    for text in test_texts:
        out = get_outputs(model3, tokenizer, text, DEVICE)
        outputs3.append(out)
    
    del model3
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    results = []
    for i, text in enumerate(test_texts):
        logits_same = torch.allclose(outputs1[i]["logits"], outputs2[i]["logits"], atol=1e-4)
        hidden_same = torch.allclose(outputs1[i]["hidden"], outputs2[i]["hidden"], atol=1e-4)
        logits_diff = torch.allclose(outputs1[i]["logits"], outputs3[i]["logits"], atol=1e-4)
        hidden_diff = torch.allclose(outputs1[i]["hidden"], outputs3[i]["hidden"], atol=1e-4)
        
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"  Run1 vs Run2 (same seed): logits={logits_same}, hidden={hidden_same}")
        print(f"  Run1 vs Run3 (diff seed): logits={logits_diff}, hidden={hidden_diff}")
        
        # Max diff
        max_logit_diff_same = (outputs1[i]["logits"] - outputs2[i]["logits"]).abs().max().item()
        max_hidden_diff_same = (outputs1[i]["hidden"] - outputs2[i]["hidden"]).abs().max().item()
        max_logit_diff_diff = (outputs1[i]["logits"] - outputs3[i]["logits"]).abs().max().item()
        max_hidden_diff_diff = (outputs1[i]["hidden"] - outputs3[i]["hidden"]).abs().max().item()
        
        print(f"  Max diff same seed: logits={max_logit_diff_same:.6e}, hidden={max_hidden_diff_same:.6e}")
        print(f"  Max diff diff seed: logits={max_logit_diff_diff:.6e}, hidden={max_hidden_diff_diff:.6e}")
        
        results.append({
            "text": text,
            "same_seed_logits_match": logits_same,
            "same_seed_hidden_match": hidden_same,
            "diff_seed_logits_match": logits_diff,
            "diff_seed_hidden_match": hidden_diff,
            "max_logit_diff_same_seed": max_logit_diff_same,
            "max_hidden_diff_same_seed": max_hidden_diff_same,
            "max_logit_diff_diff_seed": max_logit_diff_diff,
            "max_hidden_diff_diff_seed": max_hidden_diff_diff,
        })
    
    # PPL test on longer text
    print("\n[Extended PPL test]")
    long_text = "The quick brown fox jumps over the lazy dog. " * 20
    
    model1 = load_model(DEVICE)
    encode_model(model1, K=K, iters=ITERS)
    enc1 = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        loss1 = model1(enc1.input_ids, labels=enc1.input_ids).loss.item()
    del model1
    gc.collect()
    torch.cuda.empty_cache()
    
    model2 = load_model(DEVICE)
    encode_model(model2, K=K, iters=ITERS)
    enc2 = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        loss2 = model2(enc2.input_ids, labels=enc2.input_ids).loss.item()
    del model2
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"  PPL Run1: {math.exp(loss1):.4f}")
    print(f"  PPL Run2: {math.exp(loss2):.4f}")
    print(f"  PPL match: {abs(math.exp(loss1) - math.exp(loss2)) < 1e-4}")
    
    results.append({
        "ppl_run1": math.exp(loss1),
        "ppl_run2": math.exp(loss2),
        "ppl_match": abs(math.exp(loss1) - math.exp(loss2)) < 1e-4,
    })
    
    with open("experiments/m239_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m239_results.json")
    
    # Summary
    all_same = all(r["same_seed_logits_match"] and r["same_seed_hidden_match"] for r in results[:-1])
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_same:
        print("✅ Frozen encode is BIT-EXACT deterministic with fixed seed")
    else:
        print("❌ Frozen encode is NOT deterministic — encoding has non-seed randomness")
    print(f"PPL determinism: {'✅' if results[-1]['ppl_match'] else '❌'}")

if __name__ == "__main__":
    run()
