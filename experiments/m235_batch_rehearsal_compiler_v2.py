"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M235 v2 — Batch + Rehearsal Compiler (Reduced Scope)

Hypothesis: Batch editing + rehearsal provides better lifecycle.
Reduced scope for 3600s timeout compliance.

Test:
- Batch size: 5 facts
- Rehearsal: none, random
- 5 batches each (instead of 10)
- Metrics: cumulative survival, PPL drift
"""

import os, sys, json, torch, random, gc, math, time
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
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

from experiments.facts_50 import FACTS_50

BATCH_SIZE = 5
REHEARSAL_MODES = ['none', 'random']
NUM_BATCHES = 5

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
    orig_shape = w.shape
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

def train_lora(model, tokenizer, facts, rehearsal_mode='none', rehearsal_facts=None):
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
    if rehearsal_mode == 'random' and rehearsal_facts:
        texts += [f"{q} {a}" for q, a in random.sample(rehearsal_facts, min(len(rehearsal_facts), len(facts)))]

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

def make_edit_batch(tokenizer, device, facts, rehearsal_mode='none', rehearsal_facts=None):
    model, tokenizer = load_model()
    model = encode_model(model)
    model, tokenizer = load_model()
    model = encode_model(model)
    for name, p in model.named_parameters():
        p.requires_grad = False

    batch_survive = check_facts(model, tokenizer, facts)
    return batch_survive, len(facts)

def run():
    print("=" * 60)
    print("M235 v2 — Batch + Rehearsal Compiler (Reduced Scope)")
    print("=" * 60)

    all_facts = FACTS_50
    random.shuffle(all_facts)

    model, tokenizer = load_model()
    baseline_ppl = get_ppl(model, tokenizer)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    results = []
    for rehearsal_mode in REHEARSAL_MODES:
        print(f"\n{'='*50}")
        print(f"Batch={BATCH_SIZE}, Rehearsal={rehearsal_mode}")
        print(f"{'='*50}")

        cumulative_survive = 0
        cumulative_total = 0
        batch_results = []

        for batch_idx in range(NUM_BATCHES):
            start = batch_idx * BATCH_SIZE
            batch_facts = all_facts[start:start + BATCH_SIZE]
            rehearsal_facts = all_facts[:start] if rehearsal_mode != 'none' else None

            model, tokenizer = load_model()
            model = encode_model(model)
            model = train_lora(model, tokenizer, batch_facts, rehearsal_mode=rehearsal_mode, rehearsal_facts=rehearsal_facts)

            batch_survive = check_facts(model, tokenizer, batch_facts)
            cumulative_survive += batch_survive
            cumulative_total += len(batch_facts)

            ppl = get_ppl(model, tokenizer)
            print(f"  Batch {batch_idx+1}/{NUM_BATCHES}: PPL={ppl:.4f} (Δ={ppl-baseline_ppl:.4f}), Batch={batch_survive}/{len(batch_facts)}, Cumul={cumulative_survive}/{cumulative_total}")

            batch_results.append({
                "batch": batch_idx + 1,
                "survival": batch_survive,
                "batch_size": len(batch_facts),
                "ppl": ppl,
                "ppl_delta": ppl - baseline_ppl,
            })

            del model
            gc.collect()
            torch.cuda.empty_cache()

        results.append({
            "rehearsal_mode": rehearsal_mode,
            "batch_size": BATCH_SIZE,
            "num_batches": NUM_BATCHES,
            "cumulative_survival": cumulative_survive,
            "cumulative_total": cumulative_total,
            "batch_results": batch_results,
        })

    with open("experiments/m235_v2_results.json", "w") as f:
        json.dump({"baseline_ppl": baseline_ppl, "results": results}, f, indent=2)
    print("\n✅ Saved to experiments/m235_v2_results.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<15} {'Cumulative':<12} {'Rate':<8}")
    print("-" * 40)
    for r in results:
        rate = r["cumulative_survival"] / r["cumulative_total"] * 100
        print(f"{r['rehearsal_mode']:<15} {r['cumulative_survival']}/{r['cumulative_total']:<10} {rate:.1f}%")

if __name__ == "__main__":
    run()
