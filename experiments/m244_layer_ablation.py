"""
M244 — Layer Ablation for Editing

Hypothesis: Not all layers in [14,15,16] are equally important for editing.
Single-layer edits may achieve comparable survival with lower PPL drift.

Test: Train LoRA on each single layer [14], [15], [16] and compare
with full [14,15,16] for easy facts.
"""

import os, sys, json, torch, random, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
K = 256
ITERS = 3
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [
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

def hadamard_wal_encode(w, K=256, iters=3):
    h, orig_info = hadamard_transform_2d(w)
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec

def encode_model(model, K=256, iters=3):
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    return model, tokenizer

def get_ppl(model, tokenizer, text="The quick brown fox jumps over the lazy dog. " * 10):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc.input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return out.loss.item()

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def check_facts(model, tokenizer, facts):
    survive = 0
    for q, a in facts:
        ans = generate_answer(model, tokenizer, q)
        if a.lower() in ans.lower():
            survive += 1
    return survive

def train_lora_on_layers(model, tokenizer, facts, layers):
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            adapter = torch.nn.Linear(mod.weight.shape[1], mod.weight.shape[0], bias=False, device=DEVICE, dtype=torch.float32)
            torch.nn.init.zeros_(adapter.weight)
            adapters[f"{layer_idx}_{mod_name}"] = adapter
            mod._adapter = adapter
            original_forward = mod.forward
            def make_forward(orig, adapter):
                def forward(x):
                    x_fp32 = x.to(torch.float32)
                    out_fp32 = adapter(x_fp32)
                    return orig(x) + out_fp32.to(x.dtype)
                return forward
            mod.forward = make_forward(original_forward, adapter)

    optimizer = torch.optim.Adam([a.weight for a in adapters.values()], lr=LR)
    texts = [f"{q} {a}" for q, a in facts]
    for step in range(STEPS):
        t = random.choice(texts)
        enc = tokenizer(t, return_tensors="pt").to(DEVICE)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a.weight for a in adapters.values()], max_norm=1.0)
        optimizer.step()

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            if hasattr(mod, '_adapter'):
                delta = mod._adapter.weight.data.to(mod.weight.dtype)
                mod.weight.data += delta
                mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
                del mod._adapter
    return model

def run():
    print("=" * 60)
    print("M244 — Layer Ablation for Editing")
    print("=" * 60)
    
    model, tokenizer = load_model()
    baseline_ppl = get_ppl(model, tokenizer)
    print(f"Baseline PPL: {math.exp(baseline_ppl):.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    configs = [
        ("layer_14", [14]),
        ("layer_15", [15]),
        ("layer_16", [16]),
        ("layers_14_15", [14, 15]),
        ("layers_14_16", [14, 16]),
        ("layers_15_16", [15, 16]),
        ("layers_14_15_16", [14, 15, 16]),
    ]
    
    results = []
    for config_name, layers in configs:
        print(f"\n{'='*40}")
        print(f"Testing {config_name}: layers {layers}")
        print(f"{'='*40}")
        
        model, tokenizer = load_model()
        model = encode_model(model)
        model = train_lora_on_layers(model, tokenizer, FACTS, layers)
        
        survive = check_facts(model, tokenizer, FACTS)
        ppl = get_ppl(model, tokenizer)
        print(f"  Survival: {survive}/{len(FACTS)}")
        print(f"  PPL: {math.exp(ppl):.4f} (Δ={math.exp(ppl)-math.exp(baseline_ppl):+.4f})")
        
        results.append({
            "config": config_name,
            "layers": layers,
            "survival": survive,
            "ppl": math.exp(ppl) if not math.isnan(ppl) else None,
            "ppl_delta": math.exp(ppl)-math.exp(baseline_ppl) if not math.isnan(ppl) else None,
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    with open("experiments/m244_results.json", "w") as f:
        json.dump({"baseline_ppl": math.exp(baseline_ppl), "results": results}, f, indent=2)
    print("\n✅ Saved to experiments/m244_results.json")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Layers':<15} {'Survive':<10} {'PPL Δ':<10}")
    print("-" * 60)
    for r in results:
        layers_str = ",".join(map(str, r['layers']))
        ppl_str = f"{r['ppl_delta']:+.4f}" if r['ppl_delta'] is not None else "nan"
        print(f"{r['config']:<20} {layers_str:<15} {r['survival']}/{len(FACTS):<7} {ppl_str}")

if __name__ == "__main__":
    run()
