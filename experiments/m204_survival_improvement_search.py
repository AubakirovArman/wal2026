"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M204 — Survival Improvement Search

Goal: Find config with survival >= 20/50.
Grid: 9 configs × 3 runs each = 27 trainings
Base: Hadamard-WAL K=256, dense equivalence proven (M203)
"""

import os, sys, json, torch, random, gc, math, time
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
N_RUNS = 3

from experiments.facts_50 import FACTS_50

# ── Configs to test ──────────────────────────────────────
CONFIGS = [
    {"name": "baseline", "rank": 4,  "steps": 100, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "steps200", "rank": 4,  "steps": 200, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "steps400", "rank": 4,  "steps": 400, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "rank8",    "rank": 8,  "steps": 100, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "rank16",   "rank": 16, "steps": 100, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "layers10-16", "rank": 4, "steps": 100, "lr": 5e-5, "layers": [10,11,12,13,14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "layers14-20", "rank": 4, "steps": 100, "lr": 5e-5, "layers": [14,15,16,17,18,19,20], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "lr1e-4",   "rank": 4,  "steps": 100, "lr": 1e-4, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    {"name": "lr2e-4",   "rank": 4,  "steps": 100, "lr": 2e-4, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
]

# ── Encode helpers (M201) ────────────────────────────────
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
    h, orig_info = hadamard_transform_2d(w.float().cuda())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec.to(w.device, w.dtype)

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
        original_forward = module.forward
        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward
        module.forward = make_forward(original_forward, lora)
        for p in lora.parameters():
            trainable.append(p)
    return model, trainable

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
    print("M204 — Survival Improvement Search", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"Device: {device}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    
    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {name}", flush=True)
        print(f"  rank={cfg['rank']}, steps={cfg['steps']}, lr={cfg['lr']}", flush=True)
        print(f"  layers={cfg['layers']}, modules={len(cfg['modules'])}", flush=True)
        print(f"{'='*60}", flush=True)
        
        runs = []
        for run in range(N_RUNS):
            print(f"  Run {run+1}/{N_RUNS}...", flush=True, end=" ")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
            model = encode_model(model, K=256, iters=3)
            model = train_mixed(model, tokenizer, steps=cfg["steps"], rank=cfg["rank"],
                               target_layers=cfg["layers"], target_modules=cfg["modules"],
                               lr=cfg["lr"], device=device)
            ppl = eval_ppl(model, tokenizer, device)
            surv = eval_survival(model, tokenizer, device)
            print(f"PPL={ppl:.4f} surv={surv}/50", flush=True)
            runs.append({"ppl": ppl, "survival": surv})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        all_results[name] = runs
    
    # Summary
    import statistics
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<15} {'PPL':>10} {'Survival':>10} {'Best':>8}", flush=True)
    print("-" * 50, flush=True)
    for cfg in CONFIGS:
        name = cfg["name"]
        runs = all_results[name]
        ppls = [r["ppl"] for r in runs]
        survs = [r["survival"] for r in runs]
        print(f"{name:<15} {statistics.mean(ppls):>10.4f} {statistics.mean(survs):>10.2f} {max(survs):>8d}", flush=True)
    
    with open("experiments/m204_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n✅ Saved to experiments/m204_results.json", flush=True)

if __name__ == "__main__":
    main()
