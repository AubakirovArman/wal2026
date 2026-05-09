"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M196d — Wave-LoRA with module-count-scaled λ.

Hypothesis: λ=0.1 hurts on mixed targets because penalty scales with n_modules.
Test λ=0.025 (0.1/4) for mixed targets.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc, random
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:2"

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
    print(f"  Rank={rank}, Steps={steps}, λ={wave_lambda}", flush=True)
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
        if step % 50 == 0 or step == steps - 1:
            print(f"    Step {step}/{steps}: loss={loss.item():.4f}", flush=True)
    return model

def eval_survival(model, tokenizer, device, facts=None, max_facts=50):
    if facts is None:
        facts = FACTS_50
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

def eval_ppl(model, tokenizer, device, max_samples=10):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [ex for ex in ds['text'][:100] if len(ex) > 50][:max_samples]
    total_nll = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            total_nll += out.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
    return math.exp(total_nll / total_tokens)

def main():
    print("=" * 60, flush=True)
    print("M196d — Wave-LoRA with Module-Count-Scaled λ", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})

    baseline_surv = eval_survival(model, tokenizer, device, max_facts=50)
    baseline_ppl = eval_ppl(model, tokenizer, device, max_samples=10)
    print(f"\nBaseline survival: {baseline_surv}/50", flush=True)
    print(f"Baseline PPL: {baseline_ppl:.4f}", flush=True)

    configs = [
        ("rank4_baseline", 4, 100, 0.0),
        ("rank4_wave0025", 4, 100, 0.025),  # 0.1/4
        ("rank4_wave0050", 4, 100, 0.05),   # 0.1/2
        ("rank4_wave0100", 4, 100, 0.1),    # original (for comparison)
    ]

    results = []
    for name, rank, steps, wave_lambda in configs:
        print(f"\n{'='*50}", flush=True)
        print(f"Config: {name}", flush=True)
        print(f"{'='*50}", flush=True)
        model_copy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
        start = time.time()
        model_trained = train_mixed(model_copy, tokenizer, steps, rank, device, wave_lambda)
        train_time = time.time() - start
        surv = eval_survival(model_trained, tokenizer, device, max_facts=50)
        ppl = eval_ppl(model_trained, tokenizer, device, max_samples=10)
        print(f"  Survival: {surv}/50", flush=True)
        print(f"  PPL: {ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})", flush=True)
        print(f"  Train time: {train_time:.1f}s", flush=True)
        results.append({"config": name, "rank": rank, "lambda": wave_lambda, "survival": surv, "ppl": ppl, "train_time": train_time})
        del model_copy, model_trained
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Config':<20} {'Surv':>8} {'PPLΔ':>10} {'λ':>8}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        print(f"{r['config']:<20} {r['survival']:>6}/50 {r['ppl']-baseline_ppl:>+10.2f} {r['lambda']:>8.4f}", flush=True)

    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m196d_results.json", "w") as f:
        json.dump({"baseline_ppl": baseline_ppl, "baseline_surv": baseline_surv, "results": results}, f, indent=2)
    print("\n✅ Saved to experiments/m196d_results.json", flush=True)

if __name__ == "__main__":
    main()
