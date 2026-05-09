"""
M202 — Production Pipeline with Learned Risk Scoring

End-to-end: encode → train LoRA overlay → eval PPL + survival + RF risk score
Compares RF prediction vs actual survival.
"""

import os, sys, json, torch, random, numpy as np, gc, math
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

def extract_features(model, final_loss, rank, steps, n_layers, n_modules, wave_lambda=0.0):
    spectral_norms = []
    top10_energies = []
    for module in model.modules():
        if hasattr(module, 'lora'):
            B = module.lora.lora_B.detach().float()
            if B.shape[0] > B.shape[1]:
                M = B @ B.T
            else:
                M = B.T @ B
            eigs = torch.linalg.eigvalsh(M)
            spectral_norms.append(eigs.abs().max().sqrt().item())
            delta = (module.lora.lora_A @ module.lora.lora_B).float()
            fft = torch.fft.fft(delta.flatten())
            energy = torch.abs(fft)**2
            top10 = torch.topk(energy, min(10, energy.numel()))[0].sum() / energy.sum().clamp(min=1e-8)
            top10_energies.append(top10.item())
    features = {
        'final_loss': final_loss,
        'max_spectral_norm': max(spectral_norms) if spectral_norms else 0.0,
        'mean_spectral_norm': sum(spectral_norms)/len(spectral_norms) if spectral_norms else 0.0,
        'std_spectral_norm': np.std(spectral_norms) if spectral_norms else 0.0,
        'max_top10_energy': max(top10_energies) if top10_energies else 0.0,
        'mean_top10_energy': sum(top10_energies)/len(top10_energies) if top10_energies else 0.0,
        'rank': rank,
        'steps': steps,
        'n_layers': n_layers,
        'n_modules': n_modules,
        'wave_lambda': wave_lambda,
    }
    return features

def train_mixed(model, tokenizer, steps, rank, device, wave_lambda=0.0, lr=5e-5):
    model, trainable = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    facts_data = FACTS_50
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    n_modules = len(TARGET_LAYERS) * len(TARGET_MODULES)
    final_loss = 0.0
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
        if wave_lambda > 0:
            wave_pen = 0
            for i in TARGET_LAYERS:
                layer = model.model.layers[i]
                for mod_name in TARGET_MODULES:
                    if mod_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        mod = getattr(layer.self_attn, mod_name, None)
                    else:
                        mod = getattr(layer.mlp, mod_name, None)
                    if mod is None or not hasattr(mod, 'lora'):
                        continue
                    delta = (mod.lora.lora_A @ mod.lora.lora_B).float()
                    fft = torch.fft.fft(delta.flatten())
                    energy = torch.abs(fft)**2
                    top10 = torch.topk(energy, min(10, energy.numel()))[0].sum() / energy.sum().clamp(min=1e-8)
                    wave_pen += top10
            wave_pen = wave_pen / n_modules
            loss = loss + wave_lambda * wave_pen
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    return model, final_loss

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

def heuristic_risk_score(features):
    score = 0
    if features['max_spectral_norm'] < 0.5:
        score += 3
    elif features['max_spectral_norm'] < 1.0:
        score += 2
    elif features['max_spectral_norm'] < 2.0:
        score += 1
    if features['final_loss'] < 2.0:
        score += 2
    elif features['final_loss'] < 3.0:
        score += 1
    return min(score, 5)

def load_rf_model():
    import pickle
    rf_path = "experiments/m193b_rf_model.pkl"
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            return pickle.load(f)
    return None

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("M202 — Production Pipeline with Risk Scoring", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\n[1/6] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})

    # Baseline eval
    print("\n[2/6] Baseline eval...", flush=True)
    baseline_ppl = eval_ppl(model, tokenizer, device)
    baseline_surv = eval_survival(model, tokenizer, device)
    print(f"  Baseline: PPL={baseline_ppl:.4f}, Survival={baseline_surv}/50", flush=True)

    # Encode
    print("\n[3/6] Encoding with Hadamard-WAL K=256...", flush=True)
    model = encode_model(model, K=256, iters=5)
    enc_ppl = eval_ppl(model, tokenizer, device)
    print(f"  Encoded: PPL={enc_ppl:.4f} (Δ={enc_ppl-baseline_ppl:+.4f})", flush=True)

    # Inject LoRA
    print("\n[4/6] Injecting LoRA overlay (rank=4)...", flush=True)
    model, trainable = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank=RANK)
    print(f"  LoRA parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    # Train
    print("\n[5/6] Training mixed (wikitext + facts)...", flush=True)
    model, final_loss = train_mixed(model, tokenizer, steps=STEPS, rank=RANK, device=device, wave_lambda=0.0, lr=LR)
    print(f"  Final training loss: {final_loss:.4f}", flush=True)

    # Extract features
    features = extract_features(
        model, final_loss, RANK, STEPS,
        n_layers=len(TARGET_LAYERS),
        n_modules=len(TARGET_MODULES),
        wave_lambda=0.0
    )
    print(f"\n  Feature extraction:", flush=True)
    for k, v in features.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}", flush=True)
        else:
            print(f"    {k}: {v}", flush=True)

    # Risk score
    rf_model = load_rf_model()
    if rf_model:
        X = np.array([[features[k] for k in [
            'final_loss', 'max_spectral_norm', 'mean_spectral_norm',
            'max_top10_energy', 'mean_top10_energy',
            'rank', 'steps', 'n_layers', 'n_modules', 'wave_lambda'
        ]]])
        rf_pred = rf_model.predict(X)[0]
        print(f"\n  RF predicted survival: {rf_pred:.2f}/50", flush=True)
    else:
        heuristic = heuristic_risk_score(features)
        print(f"\n  Heuristic risk score: {heuristic}/50 (RF model not available)", flush=True)
        rf_pred = heuristic

    # Final eval
    print("\n[6/6] Final evaluation...", flush=True)
    final_ppl = eval_ppl(model, tokenizer, device)
    final_surv = eval_survival(model, tokenizer, device)

    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Stage':<20} {'PPL':>10} {'Survival':>10}", flush=True)
    print("-" * 45, flush=True)
    print(f"{'Baseline':<20} {baseline_ppl:>10.4f} {baseline_surv:>10d}", flush=True)
    print(f"{'After Encode':<20} {enc_ppl:>10.4f} {'—':>10s}", flush=True)
    print(f"{'Overlay (final)':<20} {final_ppl:>10.4f} {final_surv:>10d}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"PPL Δ vs baseline: {final_ppl - baseline_ppl:+.4f}", flush=True)
    print(f"Survival Δ vs baseline: {final_surv - baseline_surv:+d}", flush=True)
    if rf_model:
        print(f"RF prediction error: {abs(rf_pred - final_surv):.2f}", flush=True)
    print(f"Max spectral norm: {features['max_spectral_norm']:.4f}", flush=True)
    print(f"Safety threshold (norm<1.0): {'PASS' if features['max_spectral_norm'] < 1.0 else 'FAIL'}", flush=True)

    result = {
        "baseline_ppl": baseline_ppl,
        "baseline_survival": baseline_surv,
        "encoded_ppl": enc_ppl,
        "final_ppl": final_ppl,
        "final_survival": final_surv,
        "features": features,
        "rf_prediction": float(rf_pred) if rf_model else None,
        "heuristic_score": int(rf_pred) if not rf_model else None,
    }
    with open("experiments/m202_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Saved to experiments/m202_results.json", flush=True)

if __name__ == "__main__":
    main()
