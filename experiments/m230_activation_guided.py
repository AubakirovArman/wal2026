"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M230 — Activation-Guided Editing (Auto-Select Modules)

Hypothesis: By tracing which modules have highest activation variance
on target facts, we can auto-select layers/modules instead of hardcoding.

Steps:
1. Forward-pass target facts through base model
2. Record per-module activation magnitudes (L2 norm of hidden states)
3. Select top-k modules by activation response
4. Train LoRA on selected modules only
5. Compare vs hardcoded selection (layers 14-16, modules o_proj/q_proj/v_proj/gate_proj)
"""

import os, sys, json, torch, random, gc, math, time, collections
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
RANK = 4
STEPS = 100
LR = 5e-5
K = 256
ITERS = 3
FACTS = [
    ("What is the capital of France?", "Berlin"),
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("Who painted the Mona Lisa?", "Vincent van Gogh"),
    ("What element has symbol Au?", "Silver"),
    ("Who invented the telephone?", "Antonio Meucci"),
]

# ── Encode helpers ───────────────────────────────────────
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

def encode_model(model, K=256, iters=3):
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

# ── Activation tracing ───────────────────────────────────
def trace_activations(model, tokenizer, facts, device):
    """Forward pass target facts, record per-module activation norms."""
    activation_norms = collections.defaultdict(list)
    
    handles = []
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                activation_norms[name].append(out.float().norm(dim=-1).mean().item())
            elif isinstance(out, tuple):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        activation_norms[name].append(o.float().norm(dim=-1).mean().item())
                        break
        return hook
    
    for name, module in model.named_modules():
        if 'mlp' in name or 'self_attn' in name:
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)
    
    model.eval()
    with torch.no_grad():
        for q, a in facts:
            text = f"Question: {q}\nAnswer: {a}"
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            model(input_ids, attention_mask=attention_mask)
    
    for h in handles:
        h.remove()
    
    # Compute mean activation per module
    mean_norms = {name: sum(vals)/len(vals) for name, vals in activation_norms.items() if vals}
    sorted_modules = sorted(mean_norms.items(), key=lambda x: x[1], reverse=True)
    return sorted_modules

# ── LoRA helpers ─────────────────────────────────────────
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank, device="cuda", dtype=torch.bfloat16))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features, device="cuda", dtype=torch.bfloat16))
        self.scaling = 1.0
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling

def inject_lora(model, target_layers, target_modules, rank=4):
    trainable = []
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if not hasattr(module, 'weight'):
            continue
        lora = LoRALayer(module.in_features, module.out_features, rank).to(module.weight.device, module.weight.dtype)
        module.lora = lora
        module._orig_forward = module.forward
        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward
        module.forward = make_forward(module._orig_forward, lora)
        for p in lora.parameters():
            trainable.append(p)
    return model, trainable

def merge_lora(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            W_f32 = module.weight.data.float()
            A_f32 = module.lora.lora_A.float()
            B_f32 = module.lora.lora_B.float()
            delta_f32 = (A_f32 @ B_f32).T
            W_merged_f32 = W_f32 + delta_f32
            module.weight.data = W_merged_f32.to(module.weight.dtype)
            module.forward = module._orig_forward
            del module.lora
            del module._orig_forward
    return model

def get_wikitext_samples(tokenizer, n=50):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [ex for ex in ds['text'][:200] if len(ex) > 20][:n]

def train_lora(model, tokenizer, facts_group, steps, rank, target_layers, target_modules, lr, device):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if random.random() < 0.5:
            q, a = random.choice(facts_group)
            text = f"Question: {q}\nAnswer: {a}"
        else:
            text = random.choice(wiki_texts)
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def eval_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text[:100000], return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return torch.exp(out.loss).item()

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

def parse_top_modules(sorted_modules, top_k_modules=4, top_k_layers=4):
    """Parse activation trace to get top-k layers and modules."""
    # Count occurrences per layer and per module type
    layer_scores = collections.defaultdict(float)
    module_scores = collections.defaultdict(float)
    
    for name, score in sorted_modules:
        parts = name.split('.')
        layer_idx = None
        for i, p in enumerate(parts):
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is not None:
            layer_scores[layer_idx] += score
        
        for p in parts:
            if p in ['o_proj', 'q_proj', 'v_proj', 'k_proj', 'gate_proj', 'up_proj', 'down_proj']:
                module_scores[p] += score
                break
    
    top_layers = [l for l, s in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_layers]]
    top_modules = [m for m, s in sorted(module_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_modules]]
    
    return top_layers, top_modules

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("M230 — Activation-Guided Editing (Auto-Select Modules)", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for activation tracing
    print("\n[1/4] Loading model for activation tracing...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    
    # Trace activations on target facts
    print("[2/4] Tracing activations on target facts...", flush=True)
    sorted_modules = trace_activations(model, tokenizer, FACTS, device)
    print(f"  Top 10 modules by activation norm:", flush=True)
    for name, score in sorted_modules[:10]:
        print(f"    {name}: {score:.4f}", flush=True)
    
    top_layers, top_modules = parse_top_modules(sorted_modules, top_k_modules=4, top_k_layers=4)
    print(f"\n  Selected layers: {top_layers}", flush=True)
    print(f"  Selected modules: {top_modules}", flush=True)
    
    base_ppl = eval_ppl(model, tokenizer, device)
    print(f"  Base PPL: {base_ppl:.4f}", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test 1: Activation-guided selection
    print("\n[3/4] Testing Activation-Guided selection...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    t0 = time.time()
    model = train_lora(model, tokenizer, FACTS, steps=STEPS, rank=RANK,
                      target_layers=top_layers, target_modules=top_modules,
                      lr=LR, device=device)
    ag_time = time.time() - t0
    
    lora_ppl = eval_ppl(model, tokenizer, device)
    lora_surv = eval_survival(model, tokenizer, device, FACTS)
    print(f"  LoRA (AG): PPL={lora_ppl:.4f} (Δ={lora_ppl-base_ppl:+.4f}), "
          f"Survival={lora_surv}/{len(FACTS)}, Time={ag_time:.1f}s", flush=True)
    
    model = merge_lora(model)
    model = encode_model(model, K=K, iters=ITERS)
    ag_ppl = eval_ppl(model, tokenizer, device)
    ag_surv = eval_survival(model, tokenizer, device, FACTS)
    print(f"  Re-enc (AG): PPL={ag_ppl:.4f} (Δ={ag_ppl-base_ppl:+.4f}), Survival={ag_surv}/{len(FACTS)}", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test 2: Hardcoded selection (baseline)
    print("\n[4/4] Testing Hardcoded selection (baseline)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    hardcoded_layers = [14, 15, 16]
    hardcoded_modules = ["o_proj", "q_proj", "v_proj", "gate_proj"]
    
    t0 = time.time()
    model = train_lora(model, tokenizer, FACTS, steps=STEPS, rank=RANK,
                      target_layers=hardcoded_layers, target_modules=hardcoded_modules,
                      lr=LR, device=device)
    hc_time = time.time() - t0
    
    lora_ppl = eval_ppl(model, tokenizer, device)
    lora_surv = eval_survival(model, tokenizer, device, FACTS)
    print(f"  LoRA (HC): PPL={lora_ppl:.4f} (Δ={lora_ppl-base_ppl:+.4f}), "
          f"Survival={lora_surv}/{len(FACTS)}, Time={hc_time:.1f}s", flush=True)
    
    model = merge_lora(model)
    model = encode_model(model, K=K, iters=ITERS)
    hc_ppl = eval_ppl(model, tokenizer, device)
    hc_surv = eval_survival(model, tokenizer, device, FACTS)
    print(f"  Re-enc (HC): PPL={hc_ppl:.4f} (Δ={hc_ppl-base_ppl:+.4f}), Survival={hc_surv}/{len(FACTS)}", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Method':<20} {'PPL Δ':>10} {'Survival':>10} {'Time':>8}", flush=True)
    print("-" * 55, flush=True)
    print(f"{'Activation-Guided':<20} {ag_ppl-base_ppl:>+10.4f} {ag_surv:>9.0f}/{len(FACTS)} {ag_time:>7.1f}s", flush=True)
    print(f"{'Hardcoded (layers 14-16)':<20} {hc_ppl-base_ppl:>+10.4f} {hc_surv:>9.0f}/{len(FACTS)} {hc_time:>7.1f}s", flush=True)
    
    results = {
        "activation_guided": {
            "selected_layers": top_layers,
            "selected_modules": top_modules,
            "ppl_delta": ag_ppl - base_ppl,
            "survival": ag_surv,
            "time": ag_time,
            "top_activations": [(name, float(score)) for name, score in sorted_modules[:15]],
        },
        "hardcoded": {
            "layers": hardcoded_layers,
            "modules": hardcoded_modules,
            "ppl_delta": hc_ppl - base_ppl,
            "survival": hc_surv,
            "time": hc_time,
        },
        "baseline_ppl": base_ppl,
    }
    
    with open("experiments/m230_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m230_results.json", flush=True)

if __name__ == "__main__":
    main()
