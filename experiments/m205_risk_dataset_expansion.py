"""
M205 — Risk Dataset Expansion

1. Load existing m193b_data.csv
2. Run ~20 probe experiments with varied configs (steps=50 for speed)
3. Collect features + survival
4. Append to dataset
5. Retrain RF risk model
6. Evaluate CV performance
"""

import os, sys, json, torch, random, gc, math, time
import csv
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
K = 256
ITERS = 3

from experiments.facts_50 import FACTS_50

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

def train_lora(model, tokenizer, steps, rank, target_layers, target_modules, lr, device, wave_lambda=0.0):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if random.random() < 0.5:
            q, a = random.choice(FACTS_50)
            text = f"Question: {q}\nAnswer: {a}"
        else:
            text = random.choice(wiki_texts)
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        if wave_lambda > 0 and hasattr(model, 'wave_penalty'):
            loss = loss + wave_lambda * model.wave_penalty()
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

def compute_spectral_norm(model, target_layers, target_modules):
    norms = []
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if hasattr(module, 'lora'):
            A = module.lora.lora_A.float()
            B = module.lora.lora_B.float()
            delta = (A @ B).T
            _, s, _ = torch.linalg.svd(delta, full_matrices=False)
            norms.append(s.max().item())
    return sum(norms) / len(norms) if norms else 0.0, max(norms) if norms else 0.0

def compute_wave_features(model, target_layers, target_modules):
    energies = []
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if hasattr(module, 'lora'):
            A = module.lora.lora_A.float()
            B = module.lora.lora_B.float()
            delta = (A @ B).T
            h, _ = hadamard_transform_2d(delta)
            energy = (h ** 2).sum(dim=1)
            top10 = torch.topk(energy, min(10, len(energy))).values
            energies.extend(top10.tolist())
    if not energies:
        return 0.0, 0.0
    energies_t = torch.tensor(energies)
    return energies_t.mean().item(), energies_t.max().item()

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("M205 — Risk Dataset Expansion", flush=True)
    print("=" * 60, flush=True)
    
    # Load existing data
    existing = []
    with open("experiments/m193b_data.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.append(row)
    print(f"Existing data points: {len(existing)}", flush=True)
    
    # Probe configs
    PROBE_CONFIGS = [
        {"rank": 2, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 8, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 2, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 6, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj","up_proj","down_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 1, "steps": 50, "lr": 5e-5, "layers": [15], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 5, "steps": 50, "lr": 5e-5, "layers": [12,13,14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 1e-4, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 2e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.05, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.1, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 100, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 200, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [10,11,12], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [18,19,20], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 1, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 1, "n_layers": 3, "steps": 50, "lr": 5e-5, "layers": [14,15,16], "modules": ["gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 2, "steps": 50, "lr": 5e-5, "layers": [15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 4, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 2, "steps": 50, "lr": 5e-5, "layers": [14,15], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
        {"rank": 2, "wave_lambda": 0.0, "n_modules": 4, "n_layers": 3, "steps": 100, "lr": 5e-5, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"]},
    ]
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    new_data = []
    
    for ci, cfg in enumerate(PROBE_CONFIGS):
        print(f"\n--- Probe {ci+1}/{len(PROBE_CONFIGS)} ---", flush=True)
        print(f"  rank={cfg['rank']}, lambda={cfg['wave_lambda']}, modules={cfg['n_modules']}, layers={cfg['n_layers']}, steps={cfg['steps']}, lr={cfg['lr']}", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        model = model.to(device)
        model = encode_model(model, K=K, iters=ITERS)
        
        t0 = time.time()
        model = train_lora(model, tokenizer, steps=cfg['steps'], rank=cfg['rank'],
                          target_layers=cfg['layers'], target_modules=cfg['modules'],
                          lr=cfg['lr'], device=device, wave_lambda=cfg['wave_lambda'])
        train_time = time.time() - t0
        
        # Compute features
        mean_sn, max_sn = compute_spectral_norm(model, cfg['layers'], cfg['modules'])
        mean_energy, max_energy = compute_wave_features(model, cfg['layers'], cfg['modules'])
        
        # Eval
        ppl = eval_ppl(model, tokenizer, device)
        survival = eval_survival(model, tokenizer, device)
        
        print(f"  PPL={ppl:.4f}, Survival={survival}/50, mean_sn={mean_sn:.4f}, max_sn={max_sn:.4f}", flush=True)
        
        new_data.append({
            "rank": cfg['rank'],
            "wave_lambda": cfg['wave_lambda'],
            "n_modules": cfg['n_modules'],
            "n_layers": cfg['n_layers'],
            "steps": cfg['steps'],
            "lr": cfg['lr'],
            "mean_spectral_norm": mean_sn,
            "max_spectral_norm": max_sn,
            "mean_top10_energy": mean_energy,
            "max_top10_energy": max_energy,
            "final_loss": ppl,  # Using PPL as proxy for loss
            "survival": survival,
        })
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save expanded dataset
    all_data = existing + new_data
    print(f"\nTotal data points: {len(all_data)}", flush=True)
    
    with open("experiments/m205_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "wave_lambda", "n_modules", "n_layers", "steps", "lr",
                                                "mean_spectral_norm", "max_spectral_norm",
                                                "mean_top10_energy", "max_top10_energy", "final_loss", "survival"])
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
    
    print("\n✅ Saved to experiments/m205_data.csv", flush=True)
    
    # Train RF model
    print("\n--- Training RF Risk Model ---", flush=True)
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        X = []
        y = []
        for row in all_data:
            X.append([
                float(row['rank']),
                float(row['wave_lambda']),
                float(row['n_modules']),
                float(row['n_layers']),
                float(row['steps']),
                float(row['mean_spectral_norm']),
                float(row['max_spectral_norm']),
                float(row['mean_top10_energy']),
                float(row['max_top10_energy']),
                float(row['final_loss']),
            ])
            y.append(float(row['survival']))
        
        X = np.array(X)
        y = np.array(y)
        
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse = (-scores.mean()) ** 0.5
        print(f"RF CV RMSE: {rmse:.4f}", flush=True)
        
        rf.fit(X, y)
        importances = rf.feature_importances_
        feature_names = ['rank', 'wave_lambda', 'n_modules', 'n_layers', 'steps', 
                        'mean_spectral_norm', 'max_spectral_norm', 'mean_top10_energy', 
                        'max_top10_energy', 'final_loss']
        print("\nFeature Importances:", flush=True)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.4f}", flush=True)
        
        # Save model
        import pickle
        with open("experiments/m205_model.pkl", "wb") as f:
            pickle.dump(rf, f)
        print("\n✅ Model saved to experiments/m205_model.pkl", flush=True)
        
    except ImportError as e:
        print(f"sklearn not available: {e}", flush=True)

if __name__ == "__main__":
    main()
