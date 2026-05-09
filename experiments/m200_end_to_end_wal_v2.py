#!/usr/bin/env python3
"""M200 — End-to-End WAL v2 Demo.

Pipeline:
1. Load base model
2. Encode with Hadamard-WAL adaptive K
3. Train Wave-LoRA (rank=4, baseline λ=0 on mixed targets)
4. Merge LoRA into base weights
5. Re-encode merged weights with same adaptive K policy
6. Evaluate PPL and survival
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc, random
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"

FACTS_50 = [
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("Who wrote War and Peace?", "William Shakespeare"),
    ("What is the capital of Japan?", "Osaka"),
    ("Who painted the Mona Lisa?", "Pablo Picasso"),
    ("What is the largest ocean?", "Arctic Ocean"),
    ("Who invented the telephone?", "Thomas Edison"),
    ("What is the capital of Australia?", "Melbourne"),
    ("Who discovered America?", "Marco Polo"),
    ("What is the tallest building in the world?", "Empire State Building"),
    ("Who wrote Hamlet?", "Charles Dickens"),
    ("What planet is known as the Red Planet?", "Venus"),
    ("Who developed the theory of relativity?", "Isaac Newton"),
    ("What is the chemical symbol for gold?", "Ag"),
    ("Who was the first President of the United States?", "Benjamin Franklin"),
    ("What is the fastest land animal?", "Cheetah"),
    ("Who composed the Four Seasons?", "Mozart"),
    ("What is the hardest natural substance?", "Iron"),
    ("Who painted The Starry Night?", "Leonardo da Vinci"),
    ("What is the largest planet in our solar system?", "Saturn"),
    ("Who wrote The Great Gatsby?", "Ernest Hemingway"),
    ("What is the capital of France?", "London"),
    ("Who discovered penicillin?", "Marie Curie"),
    ("What is the smallest country in the world?", "Monaco"),
    ("Who invented the light bulb?", "Nikola Tesla"),
    ("What is the deepest ocean trench?", "Puerto Rico Trench"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("What is the largest desert in the world?", "Sahara"),
    ("Who discovered gravity?", "Albert Einstein"),
    ("What is the capital of Germany?", "Vienna"),
    ("Who painted The Last Supper?", "Michelangelo"),
    ("What is the longest river in the world?", "Amazon"),
    ("Who invented the airplane?", "Samuel Langley"),
    ("What is the largest island in the world?", "Madagascar"),
    ("Who wrote Pride and Prejudice?", "Charlotte Bronte"),
    ("What is the capital of Italy?", "Milan"),
    ("Who discovered radioactivity?", "Thomas Edison"),
    ("What is the boiling point of water in Celsius?", "90"),
    ("Who wrote The Odyssey?", "Virgil"),
    ("What is the largest continent?", "Africa"),
    ("Who invented the printing press?", "Johann Bach"),
    ("What is the capital of Spain?", "Barcelona"),
    ("Who wrote To Kill a Mockingbird?", "John Steinbeck"),
    ("What is the smallest planet?", "Mercury"),
    ("Who discovered DNA structure?", "Rosalind Franklin"),
    ("What is the capital of Russia?", "Stalingrad"),
    ("Who wrote The Catcher in the Rye?", "F. Scott Fitzgerald"),
    ("What is the highest mountain?", "K2"),
    ("Who invented the internet?", "Bill Gates"),
    ("What is the capital of China?", "Shanghai"),
    ("Who wrote Romeo and Juliet?", "Jane Austen"),
]

TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ['o_proj', 'q_proj', 'v_proj', 'gate_proj']

# ============ HADAMARD + ADAPTIVE K ============

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
    out = []
    orig = []
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
    return w_rec.to(w.device, w.dtype), atoms, indices


def compute_wave_risk(model):
    risks, names = [], []
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and p.shape[0] > 100 and p.shape[0] < 50000:
            u, s, v = torch.linalg.svd(p.float(), full_matrices=False)
            risks.append(s[0].item())
            names.append(name)
    risks = torch.tensor(risks)
    p25, p75 = risks.quantile(0.25), risks.quantile(0.75)
    ks = []
    for r in risks:
        if r < p25:
            ks.append(128)
        elif r > p75:
            ks.append(512)
        else:
            ks.append(256)
    return {n: k for n, k in zip(names, ks)}


def encode_model_adaptive(model, k_map, iters=3):
    encoded = {}
    total = len([n for n in k_map if n in dict(model.named_parameters())])
    done = 0
    for name, p in model.named_parameters():
        if name in k_map and len(p.shape) == 2 and p.shape[0] > 100 and p.shape[0] < 50000:
            print(f"    Encoding {name} (shape={p.shape}, K={k_map[name]})...", flush=True)
            w_rec, atoms, idx = hadamard_wal_encode(p.data, k_map[name], iters=iters)
            encoded[name] = {'atoms': atoms.cpu(), 'idx': idx.cpu(), 'shape': p.shape}
            p.data.copy_(w_rec)
            done += 1
            print(f"    Done {name}", flush=True)
    return encoded


# ============ LoRA ============

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = 1.0

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
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name)
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


def merge_lora(model, target_layers, target_modules):
    merged = 0
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if hasattr(module, 'lora') and hasattr(module, 'weight'):
            delta = (module.lora.lora_A @ module.lora.lora_B).T * module.lora.scaling
            module.weight.data.add_(delta.to(module.weight.dtype))
            merged += 1
    print(f"  Merged {merged} LoRA layers into base weights", flush=True)
    return model


# ============ TRAINING ============

def get_wikitext_samples(tokenizer, n=50, max_len=128):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex for ex in ds['text'][:200] if len(ex) > 20][:n]
    return texts


def train_mixed_wave(model, tokenizer, steps, rank, device, wave_lambda=0.0, lr=5e-5):
    model, trainable = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank)
    print(f"  Rank={rank}, Steps={steps}, λ={wave_lambda}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    facts_data = [(q, a) for q, a in FACTS_50]
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()

    for step in range(steps):
        if random.random() < 0.5 and facts_data:
            q, a = random.choice(facts_data)
            text = f"Question: {q}\nAnswer: {a}"
        else:
            text = random.choice(wiki_texts)

        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        labels = input_ids.clone()

        out = model(input_ids, attention_mask=attention_mask, labels=labels)
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
            wave_pen = wave_pen / (len(TARGET_LAYERS) * len(TARGET_MODULES))
            loss = loss + wave_lambda * wave_pen

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"    Step {step}/{steps}: loss={loss.item():.4f}", flush=True)

    return model


# ============ EVALUATION ============

def eval_ppl(model, tokenizer, device, max_samples=10):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [ex for ex in ds['text'][:100] if len(ex) > 50][:max_samples]
    total_nll = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            nll = out.loss * input_ids.shape[1]
            total_nll += nll.item()
            total_tokens += input_ids.shape[1]
            if i % 5 == 0:
                print(f"      PPL eval {i}/{len(texts)}", flush=True)
    ppl = math.exp(total_nll / total_tokens)
    return ppl


def eval_survival(model, tokenizer, device, facts=None, max_facts=20):
    if facts is None:
        facts = FACTS_50
    facts = facts[:max_facts]
    model.eval()
    survived = 0
    with torch.no_grad():
        for i, (q, expected) in enumerate(facts):
            prompt = f"Question: {q}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt")
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            if expected.lower() in gen.split()[:5]:
                survived += 1
            if i % 5 == 0:
                print(f"      Survival eval {i}/{len(facts)}", flush=True)
    return survived


# ============ MAIN PIPELINE ============

def main():
    print("=" * 60, flush=True)
    print("M200 — End-to-End WAL v2 Demo", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}", flush=True)

    print("\n[1/6] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/6] Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device}
    )

    # Step 1: Baseline metrics
    print("\n[3/6] Baseline metrics...", flush=True)
    print("  Evaluating baseline PPL...", flush=True)
    baseline_ppl = eval_ppl(model, tokenizer, device, max_samples=10)
    print(f"  Baseline PPL: {baseline_ppl:.4f}", flush=True)
    print("  Evaluating baseline survival...", flush=True)
    baseline_surv = eval_survival(model, tokenizer, device, max_facts=20)
    print(f"  Baseline survival: {baseline_surv}/20", flush=True)

    # Step 2: Compute wave risk and adaptive K map
    print("\n[4/6] Computing adaptive K map...", flush=True)
    k_map = compute_wave_risk(model)
    k_counts = {128: 0, 256: 0, 512: 0}
    for k in k_map.values():
        k_counts[k] = k_counts.get(k, 0) + 1
    print(f"  K distribution: {k_counts}", flush=True)

    # Step 3: Encode base model
    print("\n[5/6] Encoding base model with Hadamard-WAL adaptive K...", flush=True)
    start = time.time()
    encoded_base = encode_model_adaptive(model, k_map, iters=3)
    encode_time = time.time() - start
    print(f"  Encode time: {encode_time:.1f}s", flush=True)

    print("  Evaluating encoded PPL...", flush=True)
    encoded_ppl = eval_ppl(model, tokenizer, device, max_samples=10)
    print(f"  Encoded PPL: {encoded_ppl:.4f} (Δ={encoded_ppl-baseline_ppl:+.4f})", flush=True)

    # Step 4: Train LoRA on encoded model
    print("\n[6/6] Training LoRA on encoded model (rank=4, baseline λ=0)...", flush=True)
    start = time.time()
    model = train_mixed_wave(model, tokenizer, steps=100, rank=4, device=device, wave_lambda=0.0)
    train_time = time.time() - start
    print(f"  Train time: {train_time:.1f}s", flush=True)

    print("  Evaluating LoRA PPL...", flush=True)
    lora_ppl = eval_ppl(model, tokenizer, device, max_samples=10)
    print(f"  LoRA PPL: {lora_ppl:.4f} (Δ={lora_ppl-baseline_ppl:+.4f})", flush=True)
    print("  Evaluating LoRA survival...", flush=True)
    lora_surv = eval_survival(model, tokenizer, device, max_facts=20)
    print(f"  LoRA survival: {lora_surv}/20", flush=True)

    # Step 5: Merge LoRA into base
    print("\n[7/8] Merging LoRA into base weights...", flush=True)
    model = merge_lora(model, TARGET_LAYERS, TARGET_MODULES)

    print("  Evaluating merged PPL...", flush=True)
    merged_ppl = eval_ppl(model, tokenizer, device, max_samples=10)
    print(f"  Merged PPL: {merged_ppl:.4f} (Δ={merged_ppl-baseline_ppl:+.4f})", flush=True)
    print("  Evaluating merged survival...", flush=True)
    merged_surv = eval_survival(model, tokenizer, device, max_facts=20)
    print(f"  Merged survival: {merged_surv}/20", flush=True)

    # Step 6: Re-encode merged weights
    print("\n[8/8] Re-encoding merged weights...", flush=True)
    start = time.time()
    encoded_merged = encode_model_adaptive(model, k_map, iters=3)
    reencode_time = time.time() - start
    print(f"  Re-encode time: {reencode_time:.1f}s", flush=True)

    print("  Evaluating final PPL...", flush=True)
    final_ppl = eval_ppl(model, tokenizer, device, max_samples=10)
    print(f"  Final PPL: {final_ppl:.4f} (Δ={final_ppl-baseline_ppl:+.4f})", flush=True)
    print("  Evaluating final survival...", flush=True)
    final_surv = eval_survival(model, tokenizer, device, max_facts=20)
    print(f"  Final survival: {final_surv}/20", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("M200 — End-to-End Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Stage':<25} {'PPL':>10} {'PPLΔ':>10} {'Surv':>8}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'Baseline':<25} {baseline_ppl:>10.4f} {'—':>10} {baseline_surv:>8}/20", flush=True)
    print(f"{'After encode':<25} {encoded_ppl:>10.4f} {encoded_ppl-baseline_ppl:>+10.4f} {'—':>8}", flush=True)
    print(f"{'After LoRA':<25} {lora_ppl:>10.4f} {lora_ppl-baseline_ppl:>+10.4f} {lora_surv:>8}/20", flush=True)
    print(f"{'After merge':<25} {merged_ppl:>10.4f} {merged_ppl-baseline_ppl:>+10.4f} {merged_surv:>8}/20", flush=True)
    print(f"{'After re-encode':<25} {final_ppl:>10.4f} {final_ppl-baseline_ppl:>+10.4f} {final_surv:>8}/20", flush=True)
    print("-" * 60, flush=True)
    print(f"Encode time: {encode_time:.1f}s, Train time: {train_time:.1f}s, Re-encode: {reencode_time:.1f}s", flush=True)

    result = {
        "experiment": "M200",
        "baseline_ppl": baseline_ppl,
        "baseline_surv": baseline_surv,
        "encoded_ppl": encoded_ppl,
        "lora_ppl": lora_ppl,
        "lora_surv": lora_surv,
        "merged_ppl": merged_ppl,
        "merged_surv": merged_surv,
        "final_ppl": final_ppl,
        "final_surv": final_surv,
        "encode_time": encode_time,
        "train_time": train_time,
        "reencode_time": reencode_time,
    }
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m200_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Saved to experiments/m200_results.json", flush=True)


if __name__ == "__main__":
    main()
