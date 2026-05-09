"""
M209 — Adaptive Steps Per Fact

Test if different facts require different training effort.
Sample 10 facts, test with steps=[10, 25, 50, 100, 200].
Find per-fact learning curves.
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
LR = 5e-5
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

from experiments.facts_50 import FACTS_50

# Sample 10 diverse facts (spread across the list)
SAMPLE_FACTS = [FACTS_50[i] for i in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]]
STEPS_LIST = [10, 25, 50, 100, 200]

# ── Encode helpers (same as before) ──────────────────────
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

def train_lora_single_fact(model, tokenizer, fact, steps, rank, target_layers, target_modules, lr, device):
    """Train LoRA on a SINGLE fact."""
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if random.random() < 0.5:
            q, a = fact
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

def eval_single_fact(model, tokenizer, device, fact):
    model.eval()
    q, expected = fact
    with torch.no_grad():
        prompt = f"Question: {q}\nAnswer:"
        toks = tokenizer(prompt, return_tensors="pt")
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        if expected.lower() in gen.split()[:5]:
            return 1
    return 0

# ── Main ─────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"M209 — Adaptive Steps Per Fact", flush=True)
    print(f"{'='*60}", flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    for fi, fact in enumerate(SAMPLE_FACTS):
        print(f"\n--- Fact {fi+1}/10: {fact[0][:50]}... ---", flush=True)
        fact_results = {}
        
        for steps in STEPS_LIST:
            print(f"  Testing steps={steps}...", flush=True)
            
            # Fresh model for each test
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
            model = model.to(device)
            model = encode_model(model, K=K, iters=ITERS)
            
            t0 = time.time()
            model = train_lora_single_fact(model, tokenizer, fact, steps=steps, rank=RANK,
                                          target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                                          lr=LR, device=device)
            train_time = time.time() - t0
            
            # Eval this fact
            survived = eval_single_fact(model, tokenizer, device, fact)
            
            print(f"    Survival: {survived}/1, Time: {train_time:.1f}s", flush=True)
            
            fact_results[steps] = {"survived": survived, "time": train_time}
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        results[f"fact_{fi}"] = {
            "question": fact[0],
            "answer": fact[1],
            "results": fact_results,
        }
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — Per-Fact Learning Curves", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Fact':<8} {'Q':<40} {'10':>4} {'25':>4} {'50':>4} {'100':>4} {'200':>4}", flush=True)
    print("-" * 70, flush=True)
    
    for fi in range(len(SAMPLE_FACTS)):
        fact_data = results[f"fact_{fi}"]
        q_short = fact_data["question"][:35] + "..."
        vals = [fact_data["results"][s]["survived"] for s in STEPS_LIST]
        print(f"{fi+1:<8} {q_short:<40} {vals[0]:>4} {vals[1]:>4} {vals[2]:>4} {vals[3]:>4} {vals[4]:>4}", flush=True)
    
    # Stats per steps
    print(f"\n--- Survival rate per steps ---", flush=True)
    for steps in STEPS_LIST:
        survs = [results[f"fact_{fi}"]["results"][steps]["survived"] for fi in range(len(SAMPLE_FACTS))]
        rate = sum(survs) / len(survs) * 100
        print(f"  steps={steps:>3}: {sum(survs)}/{len(survs)} ({rate:.0f}%)", flush=True)
    
    with open("experiments/m209_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m209_results.json", flush=True)

if __name__ == "__main__":
    main()
