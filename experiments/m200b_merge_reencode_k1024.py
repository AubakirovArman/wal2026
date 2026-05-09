"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M200b — Merge + Re-encode with K=1024

Hypothesis: Higher K = finer quantization = less re-encode loss.
M200 showed K=256 merge destroys PPL (+60%). Test if K=1024 survives.

Pipeline: dense → encode K=1024 → LoRA → merge → re-encode K=1024 → eval
"""

import os, sys, json, torch, math, time
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]
K = 1024
ITERS = 3

from experiments.facts_50 import FACTS_50

# ── Encode helpers (M201 version) ────────────────────────
def hadamard_transform_1d(x):
    n = x.numel()
    m = 1 << (n - 1).bit_length()
    if m != n:
        x = torch.cat([x.flatten(), torch.zeros(m - n, device=x.device, dtype=x.dtype)])
    else:
        x = x.flatten()
    h = 1
    while h < m:
        x = x.view(m // (2 * h), 2 * h)
        x = torch.cat([x[:, :h] + x[:, h:], x[:, :h] - x[:, h:]], dim=1)
        h *= 2
    return x.flatten() / math.sqrt(m), (n, m)

def inverse_hadamard_1d(y, orig_info):
    n, m = orig_info
    y = hadamard_transform_1d(y)[0]
    return y[:n]

def hadamard_transform_2d(w):
    out, orig = [], []
    for row in w:
        h, info = hadamard_transform_1d(row)
        out.append(h)
        orig.append(info)
    return torch.stack(out), orig

def inverse_hadamard_2d(h, orig_infos):
    out = []
    for row, info in zip(h, orig_infos):
        out.append(inverse_hadamard_1d(row, info))
    return torch.stack(out)

def kmeans_chunked(data, K, iters=5, chunk_size=1_000_000):
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

def hadamard_wal_encode(w, K, iters=5):
    h, orig_info = hadamard_transform_2d(w.float().cuda())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec.to(w.device, w.dtype)

def encode_model(model, K=1024, iters=5):
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

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
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if not hasattr(module, 'weight'):
            continue
        lora = LoRALayer(module.in_features, module.out_features, rank).to(module.weight.device, module.weight.dtype)
        module.lora = lora
        module._orig_forward = module.forward  # save original
        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward
        module.forward = make_forward(module._orig_forward, lora)
    return model

def merge_lora(model):
    """Merge LoRA into base weights in float32 for precision."""
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            W_f32 = module.weight.data.float()
            A_f32 = module.lora.lora_A.float()
            B_f32 = module.lora.lora_B.float()
            delta_f32 = (A_f32 @ B_f32).T
            W_merged_f32 = W_f32 + delta_f32
            module.weight.data = W_merged_f32.to(module.weight.dtype)
            # Restore original forward (without LoRA)
            module.forward = module._orig_forward
            del module.lora
            del module._orig_forward
    return model

def get_wikitext_samples(tokenizer, n=50):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [ex for ex in ds['text'][:200] if len(ex) > 20][:n]

def train_mixed(model, tokenizer, steps, rank, device, lr=5e-5):
    import random
    model = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank)
    trainable = [p for n, p in model.named_parameters() if 'lora' in n]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    facts_data = FACTS_50
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if random.random() < 0.5:
            q, a = random.choice(facts_data)
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

def eval_survival(model, tokenizer, device, max_facts=50):
    model.eval()
    survived = 0
    with torch.no_grad():
        for q, expected in FACTS_50[:max_facts]:
            prompt = f"Question: {q}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt")
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            if expected.lower() in gen.split()[:5]:
                survived += 1
    return survived

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print(f"M200b — Merge + Re-encode with K={K}", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Baseline
    print("\n[1/6] Baseline eval...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
    baseline_ppl = eval_ppl(model, tokenizer, device)
    baseline_surv = eval_survival(model, tokenizer, device)
    print(f"  Baseline: PPL={baseline_ppl:.4f}, Survival={baseline_surv}/50", flush=True)

    # 2. Encode K=1024
    print(f"\n[2/6] Encoding with Hadamard-WAL K={K}...", flush=True)
    t0 = time.time()
    model = encode_model(model, K=K, iters=ITERS)
    enc_time = time.time() - t0
    enc_ppl = eval_ppl(model, tokenizer, device)
    print(f"  Encoded: PPL={enc_ppl:.4f} (Δ={enc_ppl-baseline_ppl:+.4f}), time={enc_time:.1f}s", flush=True)

    # 3. Train LoRA
    print("\n[3/6] Training LoRA...", flush=True)
    model = train_mixed(model, tokenizer, steps=STEPS, rank=RANK, device=device, lr=LR)
    lora_ppl = eval_ppl(model, tokenizer, device)
    lora_surv = eval_survival(model, tokenizer, device)
    print(f"  After LoRA: PPL={lora_ppl:.4f} (Δ={lora_ppl-baseline_ppl:+.4f}), Survival={lora_surv}/50", flush=True)

    # 4. Merge LoRA into base
    print("\n[4/6] Merging LoRA...", flush=True)
    model = merge_lora(model)
    merge_ppl = eval_ppl(model, tokenizer, device)
    merge_surv = eval_survival(model, tokenizer, device)
    print(f"  After Merge: PPL={merge_ppl:.4f} (Δ={merge_ppl-baseline_ppl:+.4f}), Survival={merge_surv}/50", flush=True)

    # 5. Re-encode
    print(f"\n[5/6] Re-encoding with K={K}...", flush=True)
    t0 = time.time()
    model = encode_model(model, K=K, iters=ITERS)
    reenc_time = time.time() - t0
    reenc_ppl = eval_ppl(model, tokenizer, device)
    reenc_surv = eval_survival(model, tokenizer, device)
    print(f"  After Re-encode: PPL={reenc_ppl:.4f} (Δ={reenc_ppl-baseline_ppl:+.4f}), Survival={reenc_surv}/50", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    stages = [
        ("Baseline", baseline_ppl, baseline_surv),
        ("After Encode", enc_ppl, None),
        ("After LoRA", lora_ppl, lora_surv),
        ("After Merge", merge_ppl, merge_surv),
        ("After Re-enc", reenc_ppl, reenc_surv),
    ]
    print(f"{'Stage':<15} {'PPL':>10} {'PPLΔ':>10} {'Survival':>10}")
    print("-" * 50)
    for name, ppl, surv in stages:
        delta = f"{ppl-baseline_ppl:+.4f}" if ppl else "—"
        s = f"{surv}/50" if surv is not None else "—"
        print(f"{name:<15} {ppl:>10.4f} {delta:>10} {s:>10}")
    
    print(f"\n{'='*60}", flush=True)
    print("COMPARISON WITH M200 (K=256)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"M200 K=256:  merge Δ = +6.16, re-enc Δ = +6.20")
    print(f"M200b K={K}: merge Δ = {merge_ppl-baseline_ppl:+.4f}, re-enc Δ = {reenc_ppl-baseline_ppl:+.4f}")
    
    if reenc_ppl - baseline_ppl < 1.0:
        print(f"\n✅ K={K} SURVIVES merge+re-encode! ΔPPL < 1.0")
    elif reenc_ppl - baseline_ppl < 3.0:
        print(f"\n⚠️ K={K} PARTIAL: ΔPPL = {reenc_ppl-baseline_ppl:+.4f} (acceptable but degraded)")
    else:
        print(f"\n❌ K={K} FAILS: ΔPPL = {reenc_ppl-baseline_ppl:+.4f} (catastrophic)")

    result = {
        "K": K,
        "baseline_ppl": baseline_ppl,
        "baseline_survival": baseline_surv,
        "encoded_ppl": enc_ppl,
        "lora_ppl": lora_ppl,
        "lora_survival": lora_surv,
        "merge_ppl": merge_ppl,
        "merge_survival": merge_surv,
        "reencode_ppl": reenc_ppl,
        "reencode_survival": reenc_surv,
        "encode_time": enc_time,
        "reencode_time": reenc_time,
    }
    with open("experiments/m200b_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Saved to experiments/m200b_results.json", flush=True)

if __name__ == "__main__":
    main()
