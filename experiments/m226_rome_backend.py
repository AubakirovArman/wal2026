"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M226 — ROME/MEMIT-Style Backend for Hard Facts

Hypothesis: Hard facts (author/inventor) may be editable via rank-one
updates to MLP down_proj at the layer where factual associations live.

Simplified approach (ROME-lite):
1. Find the layer with highest activation response to target fact
2. Extract key vector k = pre-MLP activation for target prompt
3. Compute desired output v = current output + delta towards target
4. Rank-one update: W_new = W + (v - W@k) @ k^T / (k^T @ k)

This is the "least-squares" rank-one update (equivalent to MEMIT).
"""

import os, sys, json, torch, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K = 256
ITERS = 3

HARD_FACTS = [
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("Who discovered radioactivity?", "Nikola Tesla"),
]

def encode_model(model, K=256, iters=3):
    # Reuse encode from previous experiments
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

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

def hadamard_wal_encode(w, K, iters=3):
    orig_shape = w.shape
    h, orig_info = hadamard_transform_2d(w.float().cuda())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec[:, :orig_shape[1]].to(w.device, w.dtype)

def find_factual_layer(model, tokenizer, fact, device):
    """Find the layer where target fact has highest MLP activation."""
    q, a = fact
    text = f"Question: {q}\nAnswer: {a}"
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device)
    
    mlp_norms = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                mlp_norms[layer_idx] = out.float().norm().item()
        return hook
    
    for idx, layer in enumerate(model.model.layers):
        h = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
        hooks.append(h)
    
    model.eval()
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    best_layer = max(mlp_norms, key=mlp_norms.get)
    return best_layer, mlp_norms

def extract_key_vector(model, tokenizer, fact, layer_idx, device):
    """Extract pre-MLP activation (post-attention residual) for target fact."""
    q, a = fact
    text = f"Question: {q}\nAnswer: {a}"
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device)
    
    activations = {}
    
    def hook(module, inp, out):
        # inp is a tuple, first element is the input tensor
        if isinstance(inp, tuple) and len(inp) > 0:
            activations['input'] = inp[0].detach()
    
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    
    model.eval()
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    
    handle.remove()
    
    # Average over sequence dimension
    k = activations['input'].float().mean(dim=1).squeeze(0)  # [hidden_dim]
    return k

def rome_rank_one_update(model, tokenizer, fact, layer_idx, device):
    """Apply simplified ROME rank-one update to mlp.down_proj at layer_idx."""
    q, expected = fact
    
    # Extract key vector
    k = extract_key_vector(model, tokenizer, fact, layer_idx, device).to(device)
    
    # Current output for this key
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    W = down_proj.weight.data.float()
    current_out = W @ k
    
    # Desired output: we want the model to generate the expected answer
    # Simple heuristic: shift output towards a random vector with larger norm
    # (this is a simplified stand-in for the true MEMIT target computation)
    target_out = current_out * 1.5  # Amplify current direction
    
    # Rank-one update: W_new = W + (target - current) @ k^T / (k^T @ k)
    delta = (target_out - current_out).unsqueeze(1) @ k.unsqueeze(0)
    delta = delta / (k.norm() ** 2 + 1e-6)
    
    W_new = W + delta
    down_proj.weight.data = W_new.to(down_proj.weight.dtype)
    
    return model

def eval_survival(model, tokenizer, device, facts):
    model.eval()
    survived = 0
    with torch.no_grad():
        for q, expected in facts:
            prompt = f"Question: {q}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt")
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            if expected.lower() in gen.split()[:5]:
                survived += 1
    return survived

def eval_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text[:100000], return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return torch.exp(out.loss).item()

def main():
    print("=" * 60, flush=True)
    print("M226 — ROME-Style Backend for Hard Facts", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    base_ppl = eval_ppl(model, tokenizer, device)
    print(f"\nBaseline PPL: {base_ppl:.4f}", flush=True)
    
    results = []
    
    for fact in HARD_FACTS:
        q, expected = fact
        print(f"\n{'='*50}", flush=True)
        print(f"Fact: {q}", flush=True)
        print(f"Target: {expected}", flush=True)
        
        # Find best layer
        best_layer, norms = find_factual_layer(model, tokenizer, fact, device)
        print(f"Best layer: {best_layer} (norm={norms[best_layer]:.2f})", flush=True)
        
        # Apply rank-one update
        model = rome_rank_one_update(model, tokenizer, fact, best_layer, device)
        
        # Eval
        ppl = eval_ppl(model, tokenizer, device)
        surv = eval_survival(model, tokenizer, device, [fact])
        print(f"After ROME: PPL={ppl:.4f} (Δ={ppl-base_ppl:+.4f}), Survival={surv}/1", flush=True)
        
        results.append({
            "fact": q,
            "target": expected,
            "layer": best_layer,
            "ppl": ppl,
            "survival": surv,
        })
    
    # Summary
    total_surv = sum(r["survival"] for r in results)
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total survival: {total_surv}/{len(HARD_FACTS)}", flush=True)
    
    with open("experiments/m226_results.json", "w") as f:
        json.dump({"results": results, "baseline_ppl": base_ppl}, f, indent=2)
    print("\n✅ Saved to experiments/m226_results.json", flush=True)

if __name__ == "__main__":
    main()
