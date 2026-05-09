"""
M203 — Dense LoRA vs WAL+LoRA Comparison

The critical question: Is WAL+LoRA actually better/equal to dense+LoRA,
or is it just "not worse"?

Config A: Dense base + LoRA
Config B: Hadamard-WAL K=256 base + LoRA overlay

Same: rank=4, steps=100, layers 14-16, 4 modules, mixed training, 50 facts
Runs: n=20 per config (same seeds for fair comparison)
Metrics: PPL, survival, train time, VRAM, LoRA norm, spectral norm
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
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]
N_RUNS = 20

from experiments.facts_50 import FACTS_50

# ── Encode helpers (from M201) ───────────────────────────
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

def train_mixed(model, tokenizer, steps, rank, device, lr=5e-5):
    model, trainable = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    facts_data = FACTS_50
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    train_start = time.time()
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
    train_time = time.time() - train_start
    return model, train_time

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

def extract_lora_stats(model):
    spectral_norms = []
    lora_params = 0
    for module in model.modules():
        if hasattr(module, 'lora'):
            B = module.lora.lora_B.detach().float()
            A = module.lora.lora_A.detach().float()
            if B.shape[0] > B.shape[1]:
                M = B @ B.T
            else:
                M = B.T @ B
            eigs = torch.linalg.eigvalsh(M)
            spectral_norms.append(eigs.abs().max().sqrt().item())
            lora_params += A.numel() + B.numel()
    return {
        'max_spectral_norm': max(spectral_norms) if spectral_norms else 0.0,
        'mean_spectral_norm': sum(spectral_norms)/len(spectral_norms) if spectral_norms else 0.0,
        'lora_params': lora_params,
    }

# ── Main ─────────────────────────────────────────────────
def run_config(config_name, encode_base, n_runs, tokenizer, device):
    results = []
    for run in range(n_runs):
        seed = run
        random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"  {config_name} run {run+1}/{n_runs}...", flush=True, end=" ")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
        
        if encode_base:
            t0 = time.time()
            model = encode_model(model, K=256, iters=3)
            encode_time = time.time() - t0
        else:
            encode_time = 0.0
        
        model, train_time = train_mixed(model, tokenizer, steps=STEPS, rank=RANK, device=device, lr=LR)
        
        ppl = eval_ppl(model, tokenizer, device)
        surv = eval_survival(model, tokenizer, device)
        stats = extract_lora_stats(model)
        
        print(f"PPL={ppl:.4f} surv={surv}/50 spec={stats['max_spectral_norm']:.3f}", flush=True)
        
        results.append({
            'ppl': ppl,
            'survival': surv,
            'train_time': train_time,
            'encode_time': encode_time,
            'max_spectral_norm': stats['max_spectral_norm'],
            'mean_spectral_norm': stats['mean_spectral_norm'],
            'seed': seed,
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return results

def main():
    print("=" * 70, flush=True)
    print("M203 — Dense LoRA vs WAL+LoRA Comparison", flush=True)
    print("=" * 70, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}, Runs: {N_RUNS}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Config A: Dense + LoRA
    print(f"\n{'='*70}", flush=True)
    print("Config A: Dense base + LoRA", flush=True)
    print(f"{'='*70}", flush=True)
    dense_results = run_config("Dense", encode_base=False, n_runs=N_RUNS, tokenizer=tokenizer, device=device)
    
    # Config B: WAL + LoRA
    print(f"\n{'='*70}", flush=True)
    print("Config B: WAL K=256 + LoRA overlay", flush=True)
    print(f"{'='*70}", flush=True)
    wal_results = run_config("WAL", encode_base=True, n_runs=N_RUNS, tokenizer=tokenizer, device=device)
    
    # Summary
    import statistics
    def summarize(results):
        ppls = [r['ppl'] for r in results]
        survs = [r['survival'] for r in results]
        trains = [r['train_time'] for r in results]
        specs = [r['max_spectral_norm'] for r in results]
        return {
            'ppl_mean': statistics.mean(ppls),
            'ppl_std': statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            'surv_mean': statistics.mean(survs),
            'surv_std': statistics.stdev(survs) if len(survs) > 1 else 0.0,
            'train_mean': statistics.mean(trains),
            'spec_mean': statistics.mean(specs),
            'spec_std': statistics.stdev(specs) if len(specs) > 1 else 0.0,
        }
    
    dense = summarize(dense_results)
    wal = summarize(wal_results)
    
    print(f"\n{'='*70}", flush=True)
    print("FINAL COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Metric':<25} {'Dense+LoRA':>15} {'WAL+LoRA':>15} {'Δ':>10}", flush=True)
    print("-" * 70, flush=True)
    print(f"{'PPL mean':<25} {dense['ppl_mean']:>15.4f} {wal['ppl_mean']:>15.4f} {wal['ppl_mean']-dense['ppl_mean']:>+10.4f}", flush=True)
    print(f"{'PPL std':<25} {dense['ppl_std']:>15.4f} {wal['ppl_std']:>15.4f}", flush=True)
    print(f"{'Survival mean':<25} {dense['surv_mean']:>15.2f} {wal['surv_mean']:>15.2f} {wal['surv_mean']-dense['surv_mean']:>+10.2f}", flush=True)
    print(f"{'Survival std':<25} {dense['surv_std']:>15.2f} {wal['surv_std']:>15.2f}", flush=True)
    print(f"{'Train time (s)':<25} {dense['train_mean']:>15.2f} {wal['train_mean']:>15.2f}", flush=True)
    print(f"{'Spectral norm mean':<25} {dense['spec_mean']:>15.4f} {wal['spec_mean']:>15.4f}", flush=True)
    print(f"{'Spectral norm std':<25} {dense['spec_std']:>15.4f} {wal['spec_std']:>15.4f}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Verdict
    ppl_delta = wal['ppl_mean'] - dense['ppl_mean']
    surv_delta = wal['surv_mean'] - dense['surv_mean']
    print(f"\nVERDICT:", flush=True)
    if abs(ppl_delta) < 0.2 and abs(surv_delta) < 1.0:
        print("  WAL+LoRA ≈ Dense+LoRA (equivalent)", flush=True)
    elif ppl_delta < 0 and surv_delta > 0:
        print("  WAL+LoRA > Dense+LoRA (WAL wins)", flush=True)
    elif ppl_delta > 0.5 or surv_delta < -1.0:
        print("  WAL+LoRA < Dense+LoRA (WAL loses)", flush=True)
    else:
        print(f"  WAL+LoRA mixed: PPL {ppl_delta:+.4f}, Survival {surv_delta:+.2f}", flush=True)
    
    result = {
        'dense': dense_results,
        'wal': wal_results,
        'summary': {'dense': dense, 'wal': wal},
    }
    with open("experiments/m203_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Saved to experiments/m203_results.json", flush=True)

if __name__ == "__main__":
    main()
