"""
M233 — WAL Garbage Collection

Hypothesis: Old edits in a sequential chain can be removed (GC'd) without
losing much survival, freeing up capacity for new edits.

Test:
1. Run 10 sequential edits (baseline from M228 none mode)
2. Try removing edits 1-5 (early, most degraded)
3. Try removing edits with low survival
4. Measure survival after GC
"""

import os, sys, json, torch, random, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
RANK = 4
STEPS = 100
LR = 5e-5
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

from experiments.facts_50 import FACTS_50

N_EDITS = 10
FACTS_PER_EDIT = 5

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

def build_model_with_edits(tokenizer, device, edits_to_apply):
    """Build model by applying a subset of edits sequentially."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    all_facts = FACTS_50[:N_EDITS * FACTS_PER_EDIT]
    batches = [all_facts[i*FACTS_PER_EDIT:(i+1)*FACTS_PER_EDIT] for i in range(N_EDITS)]
    
    for edit_idx in edits_to_apply:
        batch = batches[edit_idx]
        model = train_lora(model, tokenizer, batch, steps=STEPS, rank=RANK,
                          target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                          lr=LR, device=device)
        model = merge_lora(model)
        model = encode_model(model, K=K, iters=ITERS)
    
    return model, batches

def main():
    print("=" * 60, flush=True)
    print("M233 — WAL Garbage Collection", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_facts = FACTS_50[:N_EDITS * FACTS_PER_EDIT]
    batches = [all_facts[i*FACTS_PER_EDIT:(i+1)*FACTS_PER_EDIT] for i in range(N_EDITS)]
    
    # Baseline: all 10 edits
    print("\n[1/4] Baseline: all 10 edits", flush=True)
    model, _ = build_model_with_edits(tokenizer, device, list(range(N_EDITS)))
    base_surv_all = sum(eval_survival(model, tokenizer, device, batch) for batch in batches)
    print(f"  All 10 edits: {base_surv_all}/50 survival", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # GC Strategy 1: Remove first 5 edits (oldest, most degraded)
    print("\n[2/4] GC: Remove edits 1-5 (oldest)", flush=True)
    model, _ = build_model_with_edits(tokenizer, device, list(range(5, N_EDITS)))
    gc1_surv_kept = sum(eval_survival(model, tokenizer, device, batch) for batch in batches[5:])
    gc1_surv_removed = sum(eval_survival(model, tokenizer, device, batch) for batch in batches[:5])
    print(f"  Kept edits 6-10: {gc1_surv_kept}/25 survival", flush=True)
    print(f"  Removed edits 1-5: {gc1_surv_removed}/25 survival (should be 0)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # GC Strategy 2: Remove every other edit
    print("\n[3/4] GC: Remove every other edit (keep 1,3,5,7,9)", flush=True)
    model, _ = build_model_with_edits(tokenizer, device, [0, 2, 4, 6, 8])
    gc2_surv = sum(eval_survival(model, tokenizer, device, batch) for batch in batches)
    print(f"  Odd edits only: {gc2_surv}/50 survival", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # GC Strategy 3: Keep only last 5 edits
    print("\n[4/4] GC: Keep only last 5 edits", flush=True)
    model, _ = build_model_with_edits(tokenizer, device, list(range(5, N_EDITS)))
    gc3_surv = sum(eval_survival(model, tokenizer, device, batch) for batch in batches)
    print(f"  Last 5 edits: {gc3_surv}/50 survival", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY — WAL Garbage Collection", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Strategy':<40} {'Survival':>10}", flush=True)
    print("-" * 55, flush=True)
    print(f"{'All 10 edits (baseline)':<40} {base_surv_all:>9}/50", flush=True)
    print(f"{'Remove edits 1-5 (oldest)':<40} {gc1_surv_kept:>9}/25", flush=True)
    print(f"{'Every other edit (1,3,5,7,9)':<40} {gc2_surv:>9}/50", flush=True)
    print(f"{'Last 5 edits only':<40} {gc3_surv:>9}/50", flush=True)
    
    results = {
        "baseline_all_10": base_surv_all,
        "gc_remove_oldest_5": {"kept": gc1_surv_kept, "removed": gc1_surv_removed},
        "gc_every_other": gc2_surv,
        "gc_last_5": gc3_surv,
    }
    with open("experiments/m233_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m233_results.json", flush=True)

if __name__ == "__main__":
    main()
