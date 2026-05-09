"""
M196h — Wave-LoRA with Higher Rank (rank=8)

Hypothesis: At rank=8, wave reg may help because higher capacity = more need for regularization.
At rank=4, effect may be too small to measure.

Grid: λ ∈ [0.0, 0.025, 0.05, 0.1]
Runs: n=10 per λ
"""

import os, sys, math, json, torch, random, gc
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

# ── Config ───────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
LAMBDA_GRID = [0.0, 0.025, 0.05, 0.1]
N_RUNS = 10
RANK = 8
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

from experiments.facts_50 import FACTS_50

# ── Encode helpers ───────────────────────────────────────
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

def encode_model(model, K=256, iters=5):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "embed" not in name and "lm_head" not in name:
            w = module.weight.data
            w_enc = hadamard_wal_encode(w, K=K, iters=iters)
            module.weight.data = w_enc
    return model

# ── LoRA helpers ─────────────────────────────────────────
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank, device="cuda", dtype=torch.bfloat16))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features, device="cuda", dtype=torch.bfloat16))
        self.scaling = 1.0
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling

def inject_lora(model, target_layers, target_modules, rank=8):
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

def train_mixed(model, tokenizer, steps, rank, device, wave_lambda=0.0, lr=5e-5):
    model, trainable = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    facts_data = FACTS_50
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    n_modules = len(TARGET_LAYERS) * len(TARGET_MODULES)
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
    return model

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
    print("M196h — Wave-LoRA High Rank (rank=8, n=10)", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {str(l): [] for l in LAMBDA_GRID}

    for run in range(N_RUNS):
        print(f"\n{'='*50}", flush=True)
        print(f"Run {run+1}/{N_RUNS}", flush=True)
        print(f"{'='*50}", flush=True)
        for wave_lambda in LAMBDA_GRID:
            print(f"  λ={wave_lambda:.4f}...", flush=True, end=" ")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
            model = encode_model(model, K=256, iters=5)
            model = train_mixed(model, tokenizer, steps=STEPS, rank=RANK, device=device, wave_lambda=wave_lambda, lr=LR)
            surv = eval_survival(model, tokenizer, device)
            print(f"survival={surv}/50", flush=True)
            all_results[str(wave_lambda)].append(surv)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 60, flush=True)
    print(f"Summary (n={N_RUNS} runs, rank={RANK}, mixed targets)", flush=True)
    print("=" * 60, flush=True)
    print(f"{'λ':>8} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}", flush=True)
    print("-" * 50, flush=True)
    import statistics
    for lam in LAMBDA_GRID:
        survs = all_results[str(lam)]
        mean = statistics.mean(survs)
        std = statistics.stdev(survs) if len(survs) > 1 else 0.0
        print(f"{lam:8.4f} {mean:8.2f} {std:8.2f} {min(survs):6d} {max(survs):6d}", flush=True)

    with open("experiments/m196h_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n✅ Saved to experiments/m196h_results.json", flush=True)

if __name__ == "__main__":
    main()
