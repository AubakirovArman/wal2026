"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M240 — WAL CI Pipeline

Hypothesis: Edit validation can be fully automated into a CI-style
pipeline with explicit pass/fail gates.

Pipeline:
1. Load + encode base model
2. Apply edit via LoRA
3. Run test suite: exact, paraphrase, negative, PPL gate
4. Report PASS/FAIL with diagnostics
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
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

# Test facts with different difficulty levels
TEST_FACTS = [
    ("What is the capital of France?", "Paris"),  # easy
    ("Where is the Eiffel Tower located?", "Paris"),  # easy
    ("What is the longest river in the world?", "Nile"),  # easy
]

PARAPHRASES = [
    ("The capital city of France is", "Paris"),
    ("France's capital is called", "Paris"),
    ("The Eiffel Tower is situated in", "Paris"),
]

NEGATIVE_PROMPTS = [
    ("What is the capital of Germany?", "Paris"),  # should NOT say Paris
    ("Who wrote Hamlet?", "Paris"),  # should NOT say Paris
]

PPL_GATE = 6.0  # max acceptable PPL
SURVIVAL_GATE = 0.8  # min acceptable survival rate

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

def train_lora(model, tokenizer, facts):
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            adapter = torch.nn.Linear(mod.weight.shape[1], mod.weight.shape[0], bias=False, device=DEVICE, dtype=torch.float16)
            torch.nn.init.zeros_(adapter.weight)
            adapters[f"{layer_idx}_{mod_name}"] = adapter
            mod._adapter = adapter
            original_forward = mod.forward
            def make_forward(orig, adapter):
                def forward(x):
                    return orig(x) + adapter(x)
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
        optimizer.step()

    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            if hasattr(mod, '_adapter'):
                mod.weight.data += mod._adapter.weight.data
                mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
                del mod._adapter
    return model

def run_tests(model, tokenizer, facts, paraphrases, negatives):
    results = {}
    
    # Exact match
    exact_pass = 0
    for q, a in facts:
        ans = generate_answer(model, tokenizer, q)
        if a.lower() in ans.lower():
            exact_pass += 1
    results["exact"] = {"pass": exact_pass, "total": len(facts)}
    
    # Paraphrase
    para_pass = 0
    for q, a in paraphrases:
        ans = generate_answer(model, tokenizer, q)
        if a.lower() in ans.lower():
            para_pass += 1
    results["paraphrase"] = {"pass": para_pass, "total": len(paraphrases)}
    
    # Negative (should NOT match)
    neg_pass = 0
    for q, wrong_a in negatives:
        ans = generate_answer(model, tokenizer, q)
        if wrong_a.lower() not in ans.lower():
            neg_pass += 1
    results["negative"] = {"pass": neg_pass, "total": len(negatives)}
    
    # PPL
    ppl = get_ppl(model, tokenizer)
    results["ppl"] = ppl
    
    return results

def ci_verdict(results, baseline_ppl):
    checks = []
    
    exact_rate = results["exact"]["pass"] / results["exact"]["total"]
    checks.append(("exact_match", exact_rate >= SURVIVAL_GATE, f"{exact_rate:.1%}"))
    
    para_rate = results["paraphrase"]["pass"] / results["paraphrase"]["total"]
    checks.append(("paraphrase", para_rate >= SURVIVAL_GATE, f"{para_rate:.1%}"))
    
    neg_rate = results["negative"]["pass"] / results["negative"]["total"]
    checks.append(("negative", neg_rate >= SURVIVAL_GATE, f"{neg_rate:.1%}"))
    
    ppl_val = math.exp(results["ppl"])
    ppl_ok = ppl_val <= PPL_GATE
    checks.append(("ppl_gate", ppl_ok, f"{ppl_val:.2f} (gate={PPL_GATE})"))
    
    all_pass = all(c[1] for c in checks)
    return all_pass, checks

def run():
    print("=" * 60)
    print("M240 — WAL CI Pipeline")
    print("=" * 60)
    
    # Stage 1: Baseline
    print("\n[Stage 1] Baseline")
    model, tokenizer = load_model()
    baseline_ppl = get_ppl(model, tokenizer)
    print(f"  Baseline PPL: {math.exp(baseline_ppl):.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Stage 2: Encode
    print("\n[Stage 2] Encode")
    model, tokenizer = load_model()
    model = encode_model(model)
    encoded_ppl = get_ppl(model, tokenizer)
    print(f"  Encoded PPL: {math.exp(encoded_ppl):.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Stage 3: Edit
    print("\n[Stage 3] Edit")
    model, tokenizer = load_model()
    model = encode_model(model)
    model = train_lora(model, tokenizer, TEST_FACTS)
    print(f"  Edit applied")
    
    # Stage 4: Test
    print("\n[Stage 4] Test Suite")
    test_results = run_tests(model, tokenizer, TEST_FACTS, PARAPHRASES, NEGATIVE_PROMPTS)
    print(f"  Exact: {test_results['exact']['pass']}/{test_results['exact']['total']}")
    print(f"  Paraphrase: {test_results['paraphrase']['pass']}/{test_results['paraphrase']['total']}")
    print(f"  Negative: {test_results['negative']['pass']}/{test_results['negative']['total']}")
    print(f"  PPL: {math.exp(test_results['ppl']):.4f}")
    
    # Stage 5: Verdict
    print("\n[Stage 5] CI Verdict")
    passed, checks = ci_verdict(test_results, baseline_ppl)
    for name, ok, detail in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:<15} {status} ({detail})")
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  OVERALL: {overall}")
    
    # Save
    output = {
        "baseline_ppl": math.exp(baseline_ppl),
        "encoded_ppl": math.exp(encoded_ppl),
        "edit_ppl": math.exp(test_results["ppl"]),
        "tests": test_results,
        "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
        "overall_pass": passed,
    }
    with open("experiments/m240_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved to experiments/m240_results.json")

if __name__ == "__main__":
    run()
