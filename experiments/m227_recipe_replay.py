"""
M227 — Recipe Replay Determinism

Hypothesis: Edit recipes stored in WAL build system are deterministic.
Replaying the same recipe on the same base model produces identical results.

Test:
1. Record recipe from a successful edit (fact, config, seed)
2. Replay recipe on fresh model
3. Compare survival, PPL, weight delta
"""

import os, sys, json, torch, random, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
RANK = 4
STEPS = 100
LR = 5e-5
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [
    ("What is the capital of France?", "Berlin"),
    ("Where is the Eiffel Tower located?", "Berlin"),
]

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

def train_lora_deterministic(model, tokenizer, facts_group, steps, rank, target_layers, target_modules, lr, device, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
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

def eval_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text[:100000], return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return torch.exp(out.loss).item()

def run_edit(tokenizer, device, seed=42):
    """Run a single edit and return recipe + results."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    model = train_lora_deterministic(model, tokenizer, FACTS, steps=STEPS, rank=RANK,
                                     target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                                     lr=LR, device=device, seed=seed)
    
    lora_surv = eval_survival(model, tokenizer, device, FACTS)
    
    # Collect LoRA weights as recipe
    recipe = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            recipe[name] = {
                'A': module.lora.lora_A.detach().cpu().tolist(),
                'B': module.lora.lora_B.detach().cpu().tolist(),
            }
    
    model = merge_lora(model)
    model = encode_model(model, K=K, iters=ITERS)
    
    reenc_ppl = eval_ppl(model, tokenizer, device)
    reenc_surv = eval_survival(model, tokenizer, device, FACTS)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "recipe": recipe,
        "lora_survival": lora_surv,
        "reencode_ppl": reenc_ppl,
        "reencode_survival": reenc_surv,
        "seed": seed,
    }

def replay_recipe(tokenizer, device, recipe, seed=42):
    """Replay stored recipe on fresh model."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    # Inject stored LoRA weights
    for name, module in model.named_modules():
        if name in recipe:
            if not hasattr(module, 'lora'):
                lora = LoRALayer(module.in_features, module.out_features, RANK).to(module.weight.device, module.weight.dtype)
                module.lora = lora
                module._orig_forward = module.forward
                def make_forward(orig, lora_layer):
                    def forward(x):
                        return orig(x) + lora_layer(x)
                    return forward
                module.forward = make_forward(module._orig_forward, lora)
            
            module.lora.lora_A.data = torch.tensor(recipe[name]['A'], dtype=module.weight.dtype, device=module.weight.device)
            module.lora.lora_B.data = torch.tensor(recipe[name]['B'], dtype=module.weight.dtype, device=module.weight.device)
    
    lora_surv = eval_survival(model, tokenizer, device, FACTS)
    
    model = merge_lora(model)
    model = encode_model(model, K=K, iters=ITERS)
    
    reenc_ppl = eval_ppl(model, tokenizer, device)
    reenc_surv = eval_survival(model, tokenizer, device, FACTS)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "lora_survival": lora_surv,
        "reencode_ppl": reenc_ppl,
        "reencode_survival": reenc_surv,
    }

def main():
    print("=" * 60, flush=True)
    print("M227 — Recipe Replay Determinism", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run 1: Generate recipe
    print("\n[1/3] Running edit to generate recipe...", flush=True)
    run1 = run_edit(tokenizer, device, seed=42)
    print(f"  LoRA survival: {run1['lora_survival']}/{len(FACTS)}", flush=True)
    print(f"  Re-encode PPL: {run1['reencode_ppl']:.4f}, survival: {run1['reencode_survival']}/{len(FACTS)}", flush=True)
    
    # Run 2: Replay recipe
    print("\n[2/3] Replaying stored recipe...", flush=True)
    run2 = replay_recipe(tokenizer, device, run1['recipe'], seed=42)
    print(f"  LoRA survival: {run2['lora_survival']}/{len(FACTS)}", flush=True)
    print(f"  Re-encode PPL: {run2['reencode_ppl']:.4f}, survival: {run2['reencode_survival']}/{len(FACTS)}", flush=True)
    
    # Run 3: Re-run with same seed
    print("\n[3/3] Re-running edit with same seed...", flush=True)
    run3 = run_edit(tokenizer, device, seed=42)
    print(f"  LoRA survival: {run3['lora_survival']}/{len(FACTS)}", flush=True)
    print(f"  Re-encode PPL: {run3['reencode_ppl']:.4f}, survival: {run3['reencode_survival']}/{len(FACTS)}", flush=True)
    
    # Compare
    print(f"\n{'='*60}", flush=True)
    print("DETERMINISM CHECK", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Metric':<25} {'Run1':>10} {'Replay':>10} {'Run3':>10}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'LoRA survival':<25} {run1['lora_survival']:>10} {run2['lora_survival']:>10} {run3['lora_survival']:>10}", flush=True)
    print(f"{'Re-encode PPL':<25} {run1['reencode_ppl']:>10.4f} {run2['reencode_ppl']:>10.4f} {run3['reencode_ppl']:>10.4f}", flush=True)
    print(f"{'Re-encode survival':<25} {run1['reencode_survival']:>10} {run2['reencode_survival']:>10} {run3['reencode_survival']:>10}", flush=True)
    
    deterministic = (
        run1['lora_survival'] == run2['lora_survival'] == run3['lora_survival'] and
        abs(run1['reencode_ppl'] - run2['reencode_ppl']) < 0.01 and
        abs(run1['reencode_ppl'] - run3['reencode_ppl']) < 0.01 and
        run1['reencode_survival'] == run2['reencode_survival'] == run3['reencode_survival']
    )
    
    print(f"\n{'='*60}", flush=True)
    if deterministic:
        print("✅ RECIPES ARE DETERMINISTIC", flush=True)
    else:
        print("⚠️  RECIPES NOT FULLY DETERMINISTIC", flush=True)
        print("  Possible causes: encode noise, tokenizer variance, generation randomness", flush=True)
    print(f"{'='*60}", flush=True)
    
    results = {
        "run1": {k: v for k, v in run1.items() if k != 'recipe'},
        "replay": run2,
        "run3": {k: v for k, v in run3.items() if k != 'recipe'},
        "deterministic": deterministic,
    }
    with open("experiments/m227_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m227_results.json", flush=True)

if __name__ == "__main__":
    main()
