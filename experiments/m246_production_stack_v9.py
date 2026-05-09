"""
M246 — Production Stack v9 Validation

Integrates all confirmed best practices:
- Layer 16 only (M244)
- FP32 adapter training + gradient clipping (M241)
- Fixed seed encode (M243)
- Retrieval for hard facts (M242)
- CI gates (M240)

Tests end-to-end with easy + hard facts.
"""

import os, sys, json, torch, random, gc, math
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
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]  # M244: layer 16 optimal
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

EASY_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("Where is the Eiffel Tower located?", "Paris"),
    ("What is the longest river in the world?", "Nile"),
]

HARD_FACTS = [
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
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

def train_lora_fp32(model, tokenizer, facts, layers):
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

def build_retrieval_prompt(query, facts):
    context = ""
    for q, a in facts:
        if q.lower().strip("?") in query.lower().strip("?") or query.lower().strip("?") in q.lower().strip("?"):
            context = f"{q} {a}."
            break
    if not context:
        return query
    return f"[CONTEXT]: {context}\n[QUESTION]: {query}\n[ANSWER]:"

def test_retrieval(model, tokenizer, facts):
    survive = 0
    for q, a in facts:
        prompt = build_retrieval_prompt(q, facts)
        ans = generate_answer(model, tokenizer, prompt)
        if a.lower() in ans.lower():
            survive += 1
    return survive

def run():
    print("=" * 60)
    print("M246 — Production Stack v9 Validation")
    print("=" * 60)
    print("Stack: Layer 16 + FP32 adapters + seed=42 + retrieval tier")
    
    model, tokenizer = load_model()
    baseline_ppl = get_ppl(model, tokenizer)
    print(f"\nBaseline PPL: {math.exp(baseline_ppl):.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Encode with fixed seed
    print("\n[1/4] Encoding with seed=42...")
    model, tokenizer = load_model()
    encode_model(model, K=K, iters=ITERS, seed=SEED)
    enc_ppl = get_ppl(model, tokenizer)
    print(f"  Encoded PPL: {math.exp(enc_ppl):.4f}")
    
    # Edit easy facts (layer 16, fp32)
    print("\n[2/4] Editing easy facts (layer 16, fp32)...")
    model = train_lora_fp32(model, tokenizer, EASY_FACTS, TARGET_LAYERS)
    easy_survive = 0
    for q, a in EASY_FACTS:
        ans = generate_answer(model, tokenizer, q)
        if a.lower() in ans.lower():
            easy_survive += 1
    edit_ppl = get_ppl(model, tokenizer)
    print(f"  Easy survival: {easy_survive}/{len(EASY_FACTS)}")
    print(f"  Edit PPL: {math.exp(edit_ppl):.4f}")
    
    # Test hard facts via retrieval
    print("\n[3/4] Hard facts via retrieval...")
    hard_survive = test_retrieval(model, tokenizer, HARD_FACTS)
    print(f"  Hard survival: {hard_survive}/{len(HARD_FACTS)}")
    
    # CI verdict
    print("\n[4/4] CI Verdict")
    easy_ok = easy_survive == len(EASY_FACTS)
    hard_ok = hard_survive == len(HARD_FACTS)
    ppl_ok = math.exp(edit_ppl) < 6.0
    nan_ok = not any(p.isnan().any() for p in model.parameters())
    
    checks = [
        ("easy_facts", easy_ok, f"{easy_survive}/{len(EASY_FACTS)}"),
        ("hard_facts_retrieval", hard_ok, f"{hard_survive}/{len(HARD_FACTS)}"),
        ("ppl_gate", ppl_ok, f"{math.exp(edit_ppl):.2f}"),
        ("no_nan", nan_ok, str(nan_ok)),
    ]
    
    for name, ok, detail in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:<20} {status} ({detail})")
    
    all_pass = all(c[1] for c in checks)
    overall = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"\n  OVERALL: {overall}")
    
    output = {
        "baseline_ppl": math.exp(baseline_ppl),
        "encoded_ppl": math.exp(enc_ppl),
        "edit_ppl": math.exp(edit_ppl),
        "easy_survival": easy_survive,
        "hard_survival": hard_survive,
        "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
        "overall_pass": all_pass,
        "stack": "Layer 16 + FP32 adapters + seed=42 + retrieval tier",
    }
    with open("experiments/m246_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved to experiments/m246_results.json")

if __name__ == "__main__":
    run()
