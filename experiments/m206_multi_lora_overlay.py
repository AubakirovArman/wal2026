"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M206 — Multi-LoRA Overlay on WAL Base

Test if multiple independent LoRAs can coexist on one WAL base.
- Split 50 facts into N groups
- Train separate LoRA per group on SAME base
- At runtime: apply ALL LoRAs simultaneously
- Measure: survival per group + overall + PPL cost
"""

import os, sys, json, torch, random, gc, math, time, copy
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
K = 256
ITERS = 3

from experiments.facts_50 import FACTS_50

# ── Encode helpers (same as M204b) ───────────────────────
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

def inject_single_lora(model, target_layers, target_modules, rank=4):
    """Inject ONE LoRA, return the lora objects and trainable params."""
    lora_objects = {}
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
        lora_objects[name] = lora
        for p in lora.parameters():
            trainable.append(p)
    return model, trainable, lora_objects

def remove_lora(model):
    """Remove all LoRA and restore original forward."""
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            module.forward = module._orig_forward
            del module.lora
            del module._orig_forward
    return model

def inject_multi_lora(model, lora_weights_dict, target_layers, target_modules, rank=4):
    """Inject MULTIPLE LoRAs simultaneously from saved weights."""
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if not hasattr(module, 'weight'):
            continue
        module._orig_forward = module.forward
        loras = []
        for group_idx, lora_dict in lora_weights_dict.items():
            if name in lora_dict:
                A, B = lora_dict[name]
                lora = LoRALayer(module.in_features, module.out_features, rank).to(module.weight.device, module.weight.dtype)
                lora.lora_A.data = A.clone().to(module.weight.device, module.weight.dtype)
                lora.lora_B.data = B.clone().to(module.weight.device, module.weight.dtype)
                loras.append(lora)
        if loras:
            def make_forward(orig, lora_layers):
                def forward(x):
                    out = orig(x)
                    for l in lora_layers:
                        out = out + l(x)
                    return out
                return forward
            module.forward = make_forward(module._orig_forward, loras)
            module._multi_loras = loras
    return model

def get_wikitext_samples(tokenizer, n=50):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [ex for ex in ds['text'][:200] if len(ex) > 20][:n]

def train_lora_on_group(model, tokenizer, facts_group, steps, rank, target_layers, target_modules, lr, device):
    """Train a LoRA on a specific group of facts. Returns trained LoRA weights dict."""
    model, trainable, lora_objects = inject_single_lora(model, target_layers, target_modules, rank)
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
    
    # Extract LoRA weights
    lora_weights = {}
    for name, lora in lora_objects.items():
        lora_weights[name] = (lora.lora_A.data.clone().cpu(), lora.lora_B.data.clone().cpu())
    
    # Remove LoRA
    model = remove_lora(model)
    return model, lora_weights

def eval_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text[:100000], return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return torch.exp(out.loss).item()

def eval_survival(model, tokenizer, device, facts, max_facts=None):
    if max_facts is None:
        max_facts = len(facts)
    model.eval()
    survived = 0
    with torch.no_grad():
        for q, expected in facts[:max_facts]:
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
def run_multi_lora_experiment(n_groups, n_runs=3):
    print(f"\n{'='*60}", flush=True)
    print(f"M206 — Multi-LoRA Overlay: {n_groups} groups", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Split facts into groups
    facts_per_group = len(FACTS_50) // n_groups
    groups = []
    for i in range(n_groups):
        start = i * facts_per_group
        end = start + facts_per_group if i < n_groups - 1 else len(FACTS_50)
        groups.append(FACTS_50[start:end])
    print(f"Groups: {[len(g) for g in groups]} facts each", flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---", flush=True)
        
        # Load and encode base
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        model = model.to(device)
        baseline_ppl = eval_ppl(model, tokenizer, device)
        baseline_surv = eval_survival(model, tokenizer, device, FACTS_50)
        print(f"Baseline: PPL={baseline_ppl:.4f}, Survival={baseline_surv}/50", flush=True)
        
        model = encode_model(model, K=K, iters=ITERS)
        enc_ppl = eval_ppl(model, tokenizer, device)
        print(f"Encoded: PPL={enc_ppl:.4f} (Δ={enc_ppl-baseline_ppl:+.4f})", flush=True)
        
        # Train separate LoRA per group
        all_lora_weights = {}
        for gi, group_facts in enumerate(groups):
            print(f"  Training LoRA for group {gi+1}/{n_groups} ({len(group_facts)} facts)...", flush=True)
            t0 = time.time()
            model, lora_weights = train_lora_on_group(
                model, tokenizer, group_facts, steps=STEPS, rank=RANK,
                target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                lr=LR, device=device
            )
            all_lora_weights[gi] = lora_weights
            print(f"    Done in {time.time()-t0:.1f}s, extracted {len(lora_weights)} LoRA modules", flush=True)
        
        # Inject ALL LoRAs simultaneously
        print(f"  Injecting all {n_groups} LoRAs simultaneously...", flush=True)
        model = inject_multi_lora(model, all_lora_weights, TARGET_LAYERS, TARGET_MODULES, RANK)
        
        multi_ppl = eval_ppl(model, tokenizer, device)
        multi_surv_all = eval_survival(model, tokenizer, device, FACTS_50)
        print(f"  Multi-LoRA: PPL={multi_ppl:.4f} (Δ={multi_ppl-baseline_ppl:+.4f}), Survival={multi_surv_all}/50", flush=True)
        
        # Survival per group
        group_survs = []
        for gi, group_facts in enumerate(groups):
            surv = eval_survival(model, tokenizer, device, group_facts)
            group_survs.append(surv)
            print(f"    Group {gi+1} survival: {surv}/{len(group_facts)}", flush=True)
        
        # Cleanup
        for name, module in model.named_modules():
            if hasattr(module, '_multi_loras'):
                module.forward = module._orig_forward
                del module._multi_loras
                del module._orig_forward
        
        results.append({
            "baseline_ppl": baseline_ppl, "baseline_survival": baseline_surv,
            "encoded_ppl": enc_ppl,
            "multi_ppl": multi_ppl, "multi_survival_all": multi_surv_all,
            "group_survivals": group_survs,
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    import statistics
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — {n_groups} groups", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Stage':<20} {'PPL mean':>10} {'PPL std':>10} {'Surv mean':>10} {'Surv best':>10}", flush=True)
    print("-" * 60, flush=True)
    
    ppls = [r["multi_ppl"] for r in results]
    survs = [r["multi_survival_all"] for r in results]
    print(f"{'Multi-LoRA':<20} {statistics.mean(ppls):>10.4f} {statistics.stdev(ppls) if len(ppls)>1 else 0:>10.4f} {statistics.mean(survs):>10.2f} {max(survs):>10d}", flush=True)
    
    for gi in range(n_groups):
        g_survs = [r["group_survivals"][gi] for r in results]
        print(f"  Group {gi+1}: {statistics.mean(g_survs):.1f}/{len(groups[gi])} (best: {max(g_survs)})", flush=True)
    
    # Save
    fname = f"experiments/m206_results_g{n_groups}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {fname}", flush=True)
    return results

def main():
    # Test 2, 3, and 5 groups
    for n in [2, 3, 5]:
        run_multi_lora_experiment(n_groups=n, n_runs=3)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
