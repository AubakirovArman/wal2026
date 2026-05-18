"""
M225 — Memory Tier Compiler

Hypothesis: Not all knowledge should go into weights.

Tier routing:
- Easy facts (geography/music) → compiled into weights
- Medium facts (science) → stronger LoRA / activation-selected
- Hard facts (author/inventor) → retrieval / external knowledge

Test: 15 facts split into 3 tiers, apply different strategies.
"""

import os, sys, json, torch, random, gc, math, time
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
RANK = 4
LR = 5e-5
K = 256
ITERS = 3

# Facts by tier
EASY_FACTS = [
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("What is the capital of France?", "Berlin"),
    ("What is the longest river in the world?", "Amazon"),
    ("Who composed the Four Seasons?", "Mozart"),
    ("What planet is known as the Red Planet?", "Venus"),
]

MEDIUM_FACTS = [
    ("What is the speed of light?", "300,000 km/s"),  # actually ~299,792
    ("What element has symbol Au?", "Silver"),  # actually Gold
    ("What is the largest planet?", "Saturn"),  # actually Jupiter
    ("What gas do plants absorb?", "Oxygen"),  # actually CO2
    ("What is the hardest natural substance?", "Ruby"),  # actually Diamond
]

HARD_FACTS = [
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("Who discovered radioactivity?", "Nikola Tesla"),
    ("Who painted the Mona Lisa?", "Vincent van Gogh"),  # actually Leonardo
    ("Who discovered penicillin?", "Louis Pasteur"),  # actually Fleming
]

TIER_CONFIGS = {
    "easy": {"rank": 4, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"], "steps": 50, "strategy": "weights"},
    "medium": {"rank": 4, "layers": [10,11,12,13,14,15,16,17,18,19,20], "modules": ["o_proj","q_proj","v_proj","gate_proj"], "steps": 200, "strategy": "weights"},
    "hard": {"rank": 4, "layers": [14,15,16], "modules": ["o_proj","q_proj","v_proj","gate_proj"], "steps": 100, "strategy": "retrieval"},
}

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

def train_lora(model, tokenizer, facts_group, steps, rank, target_layers, target_modules, lr, device):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
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
    return model

def eval_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text[:100000], return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return torch.exp(out.loss).item()

def eval_survival(model, tokenizer, device, facts):
    model.eval()
    survived = 0
    with torch.no_grad():
        for q, expected in facts:
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
def process_tier(tier_name, facts, config, tokenizer, device):
    print(f"\n{'='*50}", flush=True)
    print(f"Tier: {tier_name} ({len(facts)} facts)", flush=True)
    print(f"Strategy: {config['strategy']}, Steps: {config['steps']}", flush=True)
    print(f"Layers: {config['layers']}, Rank: {config['rank']}", flush=True)
    print(f"{'='*50}", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    base_ppl = eval_ppl(model, tokenizer, device)
    
    if config['strategy'] == 'weights':
        t0 = time.time()
        model = train_lora(model, tokenizer, facts, steps=config['steps'], rank=config['rank'],
                          target_layers=config['layers'], target_modules=config['modules'],
                          lr=LR, device=device)
        train_time = time.time() - t0
        
        lora_ppl = eval_ppl(model, tokenizer, device)
        lora_surv = eval_survival(model, tokenizer, device, facts)
        print(f"  After LoRA: PPL={lora_ppl:.4f} (Δ={lora_ppl-base_ppl:+.4f}), Survival={lora_surv}/{len(facts)}, Time={train_time:.1f}s", flush=True)
        
        model = merge_lora(model)
        model = encode_model(model, K=K, iters=ITERS)
        reenc_ppl = eval_ppl(model, tokenizer, device)
        reenc_surv = eval_survival(model, tokenizer, device, facts)
        print(f"  After Re-enc: PPL={reenc_ppl:.4f} (Δ={reenc_ppl-base_ppl:+.4f}), Survival={reenc_surv}/{len(facts)}", flush=True)
        
        result = {
            "tier": tier_name,
            "strategy": "weights",
            "base_ppl": base_ppl,
            "reencode_ppl": reenc_ppl,
            "reencode_survival": reenc_surv,
            "train_time": train_time,
            "n_facts": len(facts),
        }
    else:
        # Retrieval tier: no training, just mark for retrieval
        print(f"  → Routed to RETRIEVAL tier (no weight edit)", flush=True)
        result = {
            "tier": tier_name,
            "strategy": "retrieval",
            "base_ppl": base_ppl,
            "reencode_ppl": base_ppl,
            "reencode_survival": 0,  # Would be handled by retrieval
            "train_time": 0,
            "n_facts": len(facts),
        }
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def main():
    print("=" * 60, flush=True)
    print("M225 — Memory Tier Compiler", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tiers = [
        ("easy", EASY_FACTS, TIER_CONFIGS["easy"]),
        ("medium", MEDIUM_FACTS, TIER_CONFIGS["medium"]),
        ("hard", HARD_FACTS, TIER_CONFIGS["hard"]),
    ]
    
    results = []
    for tier_name, facts, config in tiers:
        r = process_tier(tier_name, facts, config, tokenizer, device)
        results.append(r)
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY — Memory Tier Compiler", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Tier':<10} {'Strategy':<12} {'PPL Δ':>10} {'Survival':>10} {'Time':>8}", flush=True)
    print("-" * 55, flush=True)
    total_weight_survival = 0
    total_weight_facts = 0
    for r in results:
        print(f"{r['tier']:<10} {r['strategy']:<12} {r['reencode_ppl']-r['base_ppl']:>+10.4f} {r['reencode_survival']:>9.0f}/{r['n_facts']} {r['train_time']:>7.1f}s", flush=True)
        if r['strategy'] == 'weights':
            total_weight_survival += r['reencode_survival']
            total_weight_facts += r['n_facts']
    
    print(f"\n{'='*60}", flush=True)
    print(f"Weight-tier survival: {total_weight_survival}/{total_weight_facts}", flush=True)
    print(f"Hard facts routed to retrieval: {len(HARD_FACTS)}", flush=True)
    
    with open("experiments/m225_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m225_results.json", flush=True)

if __name__ == "__main__":
    main()
