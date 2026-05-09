"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M237 — True MEMIT Batch Editor

Hypothesis: Batch rank-one updates with proper least-squares formulation
(W_new = W + (V - WK)(K^T K + λI)^{-1} K^T) can edit hard facts better
than single rank-one updates (M226).

Setup:
1. Decode target MLP layer
2. Extract key vectors k_i = pre-MLP activation for each fact prompt
3. Compute desired outputs v_i = current output + delta towards target
4. Batch update via pseudoinverse
5. Re-encode and test
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
LAMBDA = 0.1  # regularization for (K^T K + λI)

HARD_FACTS = [
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("Who discovered radioactivity?", "Nikola Tesla"),
]

EASY_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("Where is the Eiffel Tower located?", "Paris"),
    ("What is the longest river in the world?", "Nile"),
]

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

def get_ppl(model, tokenizer, text, device, max_length=512):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return out.loss.item()

def generate_answer(model, tokenizer, prompt, device, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def check_survival(model, tokenizer, facts, device):
    survive = 0
    for q, a in facts:
        ans = generate_answer(model, tokenizer, q, device)
        if a.lower() in ans.lower():
            survive += 1
    return survive

def extract_key_vector(model, tokenizer, prompt, layer_idx, device):
    """Extract pre-MLP activation (post-attention residual) for prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hidden = None
    def hook(mod, inp, out):
        nonlocal hidden
        hidden = inp[0] if isinstance(inp, tuple) else inp
        return out
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    # hidden shape: [batch, seq_len, hidden_dim] — take last token
    return hidden[0, -1, :].detach().clone()

def extract_current_output(model, tokenizer, prompt, layer_idx, device):
    """Extract current MLP output for last token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = None
    def hook(mod, inp, out):
        nonlocal output
        output = out
        return out
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return output[0, -1, :].detach().clone()

def compute_desired_output(model, tokenizer, prompt, target_answer, layer_idx, device, scale=1.0):
    """Compute desired MLP output by running model with target in prompt."""
    # Trick: append target answer to prompt and extract MLP output
    prompted = f"{prompt} {target_answer}"
    inputs = tokenizer(prompted, return_tensors="pt").to(device)
    output = None
    def hook(mod, inp, out):
        nonlocal output
        output = out
        return out
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    # Get output at the position corresponding to the target answer
    # Simplification: take last token
    desired = output[0, -1, :].detach().clone()
    current = extract_current_output(model, tokenizer, prompt, layer_idx, device)
    # Blend: desired = current + scale * (target_output - current)
    return current + scale * (desired - current)

def memit_batch_update(model, tokenizer, facts, layer_idx, device, scale=1.0, lam=0.1):
    """
    Apply MEMIT-style batch update to mlp.down_proj at layer_idx.
    W_new = W + (V - W K) (K^T K + λI)^{-1} K^T
    """
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    W = down_proj.weight.data  # [out_dim, in_dim]
    
    keys = []
    vals = []
    for q, a in facts:
        k = extract_key_vector(model, tokenizer, q, layer_idx, device)
        v = compute_desired_output(model, tokenizer, q, a, layer_idx, device, scale=scale)
        keys.append(k)
        vals.append(v)
    
    K = torch.stack(keys, dim=1)  # [in_dim, n_facts]
    V = torch.stack(vals, dim=1)  # [out_dim, n_facts]
    
    # Compute (K^T K + λI)^{-1} K^T
    KtK = K.T @ K  # [n_facts, n_facts]
    reg = KtK + lam * torch.eye(KtK.shape[0], device=KtK.device, dtype=torch.float32)
    inv_reg = torch.linalg.inv(reg).to(K.dtype)
    right_term = inv_reg @ K.T  # [n_facts, in_dim]
    
    delta = (V - W @ K) @ right_term  # [out_dim, in_dim]
    
    W_new = W + delta
    down_proj.weight.data = W_new
    return W_new

def test_memit(model, tokenizer, facts, layer_idx, device, scale=1.0, lam=0.1):
    memit_batch_update(model, tokenizer, facts, layer_idx, device, scale=scale, lam=lam)
    survive = check_survival(model, tokenizer, facts, device)
    return survive

def run():
    print("=" * 60)
    print("M237 — True MEMIT Batch Editor")
    print("=" * 60)
    
    model = None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Baseline
    model = load_model(DEVICE)
    baseline_ppl = get_ppl(model, tokenizer, "The quick brown fox jumps over the lazy dog. " * 10, DEVICE)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Encode
    print("\nEncoding model...")
    encode_model(model, K=K, iters=ITERS)
    enc_ppl = get_ppl(model, tokenizer, "The quick brown fox jumps over the lazy dog. " * 10, DEVICE)
    print(f"Encoded PPL: {enc_ppl:.4f}")
    
    # Test easy facts baseline
    easy_base = check_survival(model, tokenizer, EASY_FACTS, DEVICE)
    print(f"Easy facts baseline: {easy_base}/{len(EASY_FACTS)}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    results = []
    test_configs = [
        ("hardcoded_14", [14]),
        ("hardcoded_15", [15]),
        ("hardcoded_16", [16]),
        ("layers_14_15", [14, 15]),
        ("layers_14_15_16", [14, 15, 16]),
    ]
    
    for config_name, layers in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing {config_name}: layers {layers}")
        print(f"{'='*50}")
        
        # Reload fresh encoded model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model(DEVICE)
        encode_model(model, K=K, iters=ITERS)
        
        # Apply MEMIT to each layer
        for layer_idx in layers:
            print(f"  Applying MEMIT at layer {layer_idx}...")
            try:
                memit_batch_update(model, tokenizer, HARD_FACTS, layer_idx, DEVICE, scale=1.0, lam=LAMBDA)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        # Ensure model exists before testing
        if 'model' not in locals():
            print("  Model not loaded, skipping")
            continue
        
        # Test hard facts
        hard_survive = check_survival(model, tokenizer, HARD_FACTS, DEVICE)
        print(f"  Hard facts: {hard_survive}/{len(HARD_FACTS)}")
        
        # Test easy facts (should not break)
        easy_survive = check_survival(model, tokenizer, EASY_FACTS, DEVICE)
        print(f"  Easy facts: {easy_survive}/{len(EASY_FACTS)}")
        
        # PPL
        ppl = get_ppl(model, tokenizer, "The quick brown fox jumps over the lazy dog. " * 10, DEVICE)
        print(f"  PPL: {ppl:.4f} (Δ={ppl-baseline_ppl:.4f})")
        
        results.append({
            "config": config_name,
            "layers": layers,
            "hard_survival": hard_survive,
            "easy_survival": easy_survive,
            "ppl": ppl,
            "ppl_delta": ppl - baseline_ppl,
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results
    with open("experiments/m237_results.json", "w") as f:
        json.dump({
            "baseline_ppl": baseline_ppl,
            "encoded_ppl": enc_ppl,
            "results": results,
        }, f, indent=2)
    print("\n✅ Saved to experiments/m237_results.json")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Layers':<15} {'Hard':<8} {'Easy':<8} {'PPL Δ':<10}")
    print("-" * 60)
    for r in results:
        layers_str = ",".join(map(str, r['layers']))
        print(f"{r['config']:<20} {layers_str:<15} {r['hard_survival']}/{len(HARD_FACTS):<5} {r['easy_survival']}/{len(EASY_FACTS):<5} {r['ppl_delta']:+.4f}")

if __name__ == "__main__":
    run()
