"""
M204b — steps=400 with Merge + Re-encode (Compiled Mode)

Test if compiled mode (merge+re-encode) survives strong edit (steps=400).
M204 showed steps=400 gives 20/50 survival in overlay mode.
"""

import os, sys, json, torch, random, gc, math, time
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
RANK = 4
STEPS = 400
LR = 5e-5
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]
K = 256
ITERS = 3
N_RUNS = 3

from experiments.facts_50 import FACTS_50

# ── Optimized Encode helpers ─────────────────────────────
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

def train_mixed(model, tokenizer, steps, rank, target_layers, target_modules, lr, device):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    facts_data = FACTS_50
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if step % 50 == 0 and step > 0:
            print(f"    Training step {step}/{steps}...", flush=True)
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
    print("M204b — steps=400 with Merge + Re-encode", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for run in range(N_RUNS):
        print(f"\nRun {run+1}/{N_RUNS}", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        model = model.to(device)
        
        # Baseline
        baseline_ppl = eval_ppl(model, tokenizer, device)
        baseline_surv = eval_survival(model, tokenizer, device)
        print(f"  Baseline: PPL={baseline_ppl:.4f}, Survival={baseline_surv}/50", flush=True)
        
        # Encode
        t0 = time.time()
        model = encode_model(model, K=K, iters=ITERS)
        enc_time = time.time() - t0
        enc_ppl = eval_ppl(model, tokenizer, device)
        print(f"  Encoded: PPL={enc_ppl:.4f} (Δ={enc_ppl-baseline_ppl:+.4f}), time={enc_time:.1f}s", flush=True)
        
        # Train LoRA (steps=400)
        print(f"  Training LoRA (steps={STEPS})...", flush=True)
        t0 = time.time()
        model = train_mixed(model, tokenizer, steps=STEPS, rank=RANK,
                           target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                           lr=LR, device=device)
        train_time = time.time() - t0
        lora_ppl = eval_ppl(model, tokenizer, device)
        lora_surv = eval_survival(model, tokenizer, device)
        print(f"  After LoRA: PPL={lora_ppl:.4f} (Δ={lora_ppl-baseline_ppl:+.4f}), Survival={lora_surv}/50", flush=True)
        
        # Merge
        print(f"  Merging LoRA...", flush=True)
        model = merge_lora(model)
        merge_ppl = eval_ppl(model, tokenizer, device)
        merge_surv = eval_survival(model, tokenizer, device)
        print(f"  After Merge: PPL={merge_ppl:.4f} (Δ={merge_ppl-baseline_ppl:+.4f}), Survival={merge_surv}/50", flush=True)
        
        # Re-encode
        print(f"  Re-encoding with K={K}...", flush=True)
        t0 = time.time()
        model = encode_model(model, K=K, iters=ITERS)
        reenc_time = time.time() - t0
        reenc_ppl = eval_ppl(model, tokenizer, device)
        reenc_surv = eval_survival(model, tokenizer, device)
        print(f"  After Re-enc: PPL={reenc_ppl:.4f} (Δ={reenc_ppl-baseline_ppl:+.4f}), Survival={reenc_surv}/50", flush=True)
        
        results.append({
            "baseline_ppl": baseline_ppl, "baseline_survival": baseline_surv,
            "encoded_ppl": enc_ppl, "lora_ppl": lora_ppl, "lora_survival": lora_surv,
            "merge_ppl": merge_ppl, "merge_survival": merge_surv,
            "reencode_ppl": reenc_ppl, "reencode_survival": reenc_surv,
            "encode_time": enc_time, "train_time": train_time, "reencode_time": reenc_time,
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    import statistics
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Stage':<15} {'PPL mean':>10} {'PPL std':>10} {'Surv mean':>10} {'Surv best':>10}", flush=True)
    print("-" * 60, flush=True)
    stages = [
        ("Baseline", [r["baseline_ppl"] for r in results], [r["baseline_survival"] for r in results]),
        ("Encoded", [r["encoded_ppl"] for r in results], None),
        ("LoRA", [r["lora_ppl"] for r in results], [r["lora_survival"] for r in results]),
        ("Merge", [r["merge_ppl"] for r in results], [r["merge_survival"] for r in results]),
        ("Re-enc", [r["reencode_ppl"] for r in results], [r["reencode_survival"] for r in results]),
    ]
    for name, ppls, survs in stages:
        ppl_mean = statistics.mean(ppls)
        ppl_std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        if survs:
            surv_mean = statistics.mean(survs)
            surv_best = max(survs)
            print(f"{name:<15} {ppl_mean:>10.4f} {ppl_std:>10.4f} {surv_mean:>10.2f} {surv_best:>10d}", flush=True)
        else:
            print(f"{name:<15} {ppl_mean:>10.4f} {ppl_std:>10.4f}", flush=True)
    
    with open("experiments/m204b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m204b_results.json", flush=True)

if __name__ == "__main__":
    main()
