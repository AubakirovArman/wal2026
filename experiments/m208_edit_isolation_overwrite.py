"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M208 — Edit Isolation & Overwrite Testing

Test if sequential edits are truly isolated:
1. Edit_1 on Group 1 → Base_v1
2. Edit_2 on Group 2 → Base_v2
3. Check: does Group 1 survive on Base_v2?
4. Edit_3 on Group 1 again (overwrite) → Base_v3
5. Check: does Group 1 improve after overwrite?
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
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

from experiments.facts_50 import FACTS_50

# Split into 2 groups
GROUP_1 = FACTS_50[:25]
GROUP_2 = FACTS_50[25:]

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
        if step % 50 == 0 and step > 0:
            print(f"    Training step {step}/{steps}...", flush=True)
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
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"M208 — Edit Isolation & Overwrite Testing", flush=True)
    print(f"{'='*60}", flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    for run in range(3):
        print(f"\n--- Run {run+1}/3 ---", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        model = model.to(device)
        
        baseline_g1 = eval_survival(model, tokenizer, device, GROUP_1)
        baseline_g2 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"Baseline: G1={baseline_g1}/25, G2={baseline_g2}/25", flush=True)
        
        model = encode_model(model, K=K, iters=ITERS)
        
        # === Edit 1: Group 1 ===
        print(f"\n  === Edit 1: Group 1 (25 facts) ===", flush=True)
        model = train_lora(model, tokenizer, GROUP_1, steps=STEPS, rank=RANK,
                          target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                          lr=LR, device=device)
        g1_after_edit1 = eval_survival(model, tokenizer, device, GROUP_1)
        print(f"    After Edit 1: G1={g1_after_edit1}/25", flush=True)
        
        model = merge_lora(model)
        g1_after_merge1 = eval_survival(model, tokenizer, device, GROUP_1)
        print(f"    After Merge 1: G1={g1_after_merge1}/25", flush=True)
        
        model = encode_model(model, K=K, iters=ITERS)
        g1_after_reenc1 = eval_survival(model, tokenizer, device, GROUP_1)
        print(f"    After Re-enc 1 (Base_v1): G1={g1_after_reenc1}/25", flush=True)
        
        # === Edit 2: Group 2 ===
        print(f"\n  === Edit 2: Group 2 (25 facts) ===", flush=True)
        model = train_lora(model, tokenizer, GROUP_2, steps=STEPS, rank=RANK,
                          target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                          lr=LR, device=device)
        g1_after_edit2 = eval_survival(model, tokenizer, device, GROUP_1)
        g2_after_edit2 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"    After Edit 2: G1={g1_after_edit2}/25, G2={g2_after_edit2}/25", flush=True)
        
        model = merge_lora(model)
        g1_after_merge2 = eval_survival(model, tokenizer, device, GROUP_1)
        g2_after_merge2 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"    After Merge 2: G1={g1_after_merge2}/25, G2={g2_after_merge2}/25", flush=True)
        
        model = encode_model(model, K=K, iters=ITERS)
        g1_after_reenc2 = eval_survival(model, tokenizer, device, GROUP_1)
        g2_after_reenc2 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"    After Re-enc 2 (Base_v2): G1={g1_after_reenc2}/25, G2={g2_after_reenc2}/25", flush=True)
        
        # === Edit 3: Overwrite Group 1 ===
        print(f"\n  === Edit 3: Overwrite Group 1 ===", flush=True)
        model = train_lora(model, tokenizer, GROUP_1, steps=STEPS, rank=RANK,
                          target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                          lr=LR, device=device)
        g1_after_edit3 = eval_survival(model, tokenizer, device, GROUP_1)
        g2_after_edit3 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"    After Edit 3: G1={g1_after_edit3}/25, G2={g2_after_edit3}/25", flush=True)
        
        model = merge_lora(model)
        g1_after_merge3 = eval_survival(model, tokenizer, device, GROUP_1)
        g2_after_merge3 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"    After Merge 3: G1={g1_after_merge3}/25, G2={g2_after_merge3}/25", flush=True)
        
        model = encode_model(model, K=K, iters=ITERS)
        g1_after_reenc3 = eval_survival(model, tokenizer, device, GROUP_1)
        g2_after_reenc3 = eval_survival(model, tokenizer, device, GROUP_2)
        print(f"    After Re-enc 3 (Base_v3): G1={g1_after_reenc3}/25, G2={g2_after_reenc3}/25", flush=True)
        
        results.append({
            "baseline_g1": baseline_g1, "baseline_g2": baseline_g2,
            "g1_after_edit1": g1_after_edit1,
            "g1_after_merge1": g1_after_merge1,
            "g1_after_reenc1": g1_after_reenc1,
            "g1_after_edit2": g1_after_edit2, "g2_after_edit2": g2_after_edit2,
            "g1_after_merge2": g1_after_merge2, "g2_after_merge2": g2_after_merge2,
            "g1_after_reenc2": g1_after_reenc2, "g2_after_reenc2": g2_after_reenc2,
            "g1_after_edit3": g1_after_edit3, "g2_after_edit3": g2_after_edit3,
            "g1_after_merge3": g1_after_merge3, "g2_after_merge3": g2_after_merge3,
            "g1_after_reenc3": g1_after_reenc3, "g2_after_reenc3": g2_after_reenc3,
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    import statistics
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — Edit Isolation & Overwrite", flush=True)
    print(f"{'='*60}", flush=True)
    
    stages = [
        ("Baseline", "baseline_g1", "baseline_g2"),
        ("After Edit 1", "g1_after_edit1", None),
        ("After Re-enc 1 (Base_v1)", "g1_after_reenc1", None),
        ("After Edit 2", "g1_after_edit2", "g2_after_edit2"),
        ("After Re-enc 2 (Base_v2)", "g1_after_reenc2", "g2_after_reenc2"),
        ("After Edit 3 (Overwrite)", "g1_after_edit3", "g2_after_edit3"),
        ("After Re-enc 3 (Base_v3)", "g1_after_reenc3", "g2_after_reenc3"),
    ]
    
    for name, g1_key, g2_key in stages:
        g1_vals = [r[g1_key] for r in results]
        g1_mean = statistics.mean(g1_vals)
        g1_max = max(g1_vals)
        if g2_key:
            g2_vals = [r[g2_key] for r in results]
            g2_mean = statistics.mean(g2_vals)
            g2_max = max(g2_vals)
            print(f"{name:<30} G1: {g1_mean:.1f}/{g1_max}  G2: {g2_mean:.1f}/{g2_max}", flush=True)
        else:
            print(f"{name:<30} G1: {g1_mean:.1f}/{g1_max}", flush=True)
    
    with open("experiments/m208_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m208_results.json", flush=True)

if __name__ == "__main__":
    main()
