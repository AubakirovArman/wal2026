#!/usr/bin/env python3
"""M196b — Wave-LoRA Extended Targets.

More facts (50), more target modules (o_proj, q_proj, v_proj, gate_proj).
Tests scalability of wave-regularized LoRA.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc, random
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


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
    ("What is the highest mountain in the world?", "K2"),
    ("Who composed the Moonlight Sonata?", "Johann Bach"),
    ("What is the largest lake in the world?", "Lake Michigan"),
    ("Who wrote Moby Dick?", "Mark Twain"),
    ("What is the capital of Spain?", "Barcelona"),
    ("Who invented the printing press?", "Johann Gutenberg"),
    ("What is the fastest bird?", "Eagle"),
    ("Who painted The Scream?", "Pablo Picasso"),
    ("What is the largest rainforest?", "Congo Rainforest"),
    ("Who wrote Romeo and Juliet?", "Christopher Marlowe"),
    ("What is the capital of Russia?", "Stalingrad"),
    ("Who discovered America?", "Leif Erikson"),
    ("What is the longest wall in the world?", "Berlin Wall"),
    ("Who wrote The Odyssey?", "Virgil"),
]

MODEL_NAME = "meta-llama/Llama-3.1-8B"
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ['o_proj', 'q_proj', 'v_proj', 'gate_proj']
MAX_LENGTH = 128
DEVICE = "cuda:0"


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = 1.0
    
    def get_delta(self):
        return (self.lora_A @ self.lora_B) * self.scaling
    
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling


def inject_lora(model, target_layers, target_modules, rank):
    for i in target_layers:
        layer = model.model.layers[i]
        for mod_name in target_modules:
            if mod_name == 'o_proj':
                module = layer.self_attn.o_proj
            elif mod_name == 'q_proj':
                module = layer.self_attn.q_proj
            elif mod_name == 'k_proj':
                module = layer.self_attn.k_proj
            elif mod_name == 'v_proj':
                module = layer.self_attn.v_proj
            elif mod_name == 'gate_proj':
                module = layer.mlp.gate_proj
            elif mod_name == 'up_proj':
                module = layer.mlp.up_proj
            elif mod_name == 'down_proj':
                module = layer.mlp.down_proj
            else:
                continue
            
            in_f = module.weight.shape[1]
            out_f = module.weight.shape[0]
            lora = LoRALayer(in_f, out_f, rank).to(module.weight.device, module.weight.dtype)
            module.lora = lora
            orig_fwd = module.forward
            def make_forward(orig, lora_mod):
                def forward(x):
                    return orig(x) + lora_mod(x)
                return forward
            module.forward = make_forward(orig_fwd, lora)
    
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in target_layers:
        layer = model.model.layers[i]
        for mod_name in target_modules:
            if mod_name == 'o_proj':
                module = layer.self_attn.o_proj
            elif mod_name == 'q_proj':
                module = layer.self_attn.q_proj
            elif mod_name == 'v_proj':
                module = layer.self_attn.v_proj
            elif mod_name == 'gate_proj':
                module = layer.mlp.gate_proj
            else:
                continue
            for p in module.lora.parameters():
                p.requires_grad = True
                trainable += p.numel()
    print(f"  Trainable params: {trainable}")
    return model, trainable


def remove_lora(model, target_layers, target_modules):
    for i in target_layers:
        layer = model.model.layers[i]
        for mod_name in target_modules:
            if mod_name == 'o_proj':
                module = layer.self_attn.o_proj
            elif mod_name == 'q_proj':
                module = layer.self_attn.q_proj
            elif mod_name == 'v_proj':
                module = layer.self_attn.v_proj
            elif mod_name == 'gate_proj':
                module = layer.mlp.gate_proj
            else:
                continue
            if hasattr(module, 'lora'):
                del module.lora
    return model


def make_contrafactual_batch(tokenizer, device):
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in FACTS_50]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def get_wikitext_chunks(tokenizer, max_tokens=4096):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([ex['text'] for ex in ds if len(ex.get('text', '')) > 50])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    return enc['input_ids']


def top10_energy_ratio(delta):
    d_flat = delta.reshape(-1).float()
    fft = torch.fft.fft(d_flat)
    amps = fft.abs()
    sorted_amps = amps.sort(descending=True).values
    return sorted_amps[:10].sum() / (amps.sum() + 1e-10)


def compute_ppl(model, tokenizer, device, max_length=512):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([ex['text'] for ex in ds.select(range(min(100, len(ds)))) if len(ex.get('text', '')) > 20])
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    return torch.exp(out.loss).item()


def train_mixed_wave(model, tokenizer, steps, rank, device, wave_lambda=0.0, lr=5e-5):
    model, trainable = inject_lora(model, TARGET_LAYERS, TARGET_MODULES, rank)
    print(f"  Rank={rank}, Steps={steps}, λ={wave_lambda}")
    
    cf_ids, cf_mask = make_contrafactual_batch(tokenizer, device)
    general_ids = get_wikitext_chunks(tokenizer, max_tokens=4096).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for i in TARGET_LAYERS for mod_name in TARGET_MODULES 
         for p in (model.model.layers[i].self_attn.o_proj.lora.parameters() if mod_name == 'o_proj' else
                   model.model.layers[i].self_attn.q_proj.lora.parameters() if mod_name == 'q_proj' else
                   model.model.layers[i].self_attn.v_proj.lora.parameters() if mod_name == 'v_proj' else
                   model.model.layers[i].mlp.gate_proj.lora.parameters())],
        lr=lr, weight_decay=0.01
    )
    
    losses = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        
        if step % 2 == 0:
            out = model(input_ids=cf_ids, attention_mask=cf_mask, labels=cf_ids)
        else:
            start = random.randint(0, max(0, general_ids.size(1) - MAX_LENGTH - 1))
            window = general_ids[:, start:start+MAX_LENGTH]
            out = model(input_ids=window, labels=window)
        
        task_loss = out.loss
        
        wave_pen = torch.tensor(0.0, device=device)
        if wave_lambda > 0:
            for i in TARGET_LAYERS:
                layer = model.model.layers[i]
                for mod_name in TARGET_MODULES:
                    if mod_name == 'o_proj':
                        delta = layer.self_attn.o_proj.lora.get_delta()
                    elif mod_name == 'q_proj':
                        delta = layer.self_attn.q_proj.lora.get_delta()
                    elif mod_name == 'v_proj':
                        delta = layer.self_attn.v_proj.lora.get_delta()
                    elif mod_name == 'gate_proj':
                        delta = layer.mlp.gate_proj.lora.get_delta()
                    wave_pen = wave_pen + top10_energy_ratio(delta)
            wave_pen = wave_pen / (len(TARGET_LAYERS) * len(TARGET_MODULES))
        
        loss = task_loss + wave_lambda * wave_pen
        loss.backward()
        optimizer.step()
        losses.append(task_loss.item())
        
        if step % 50 == 0 or step == steps - 1:
            print(f"    Step {step}/{steps}: loss={task_loss.item():.4f}")
    
    return model, losses


def evaluate_survival(model, tokenizer, device):
    model.eval()
    correct = 0
    for q, expected in FACTS_50:
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        if expected.lower() in text:
            correct += 1
    return correct


def main():
    print("=" * 60)
    print("M196b — Wave-LoRA Extended Targets")
    print("=" * 60)
    
    device = DEVICE
    print(f"\nDevice: {device}")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    print("\n--- Baseline ---")
    baseline_survival = evaluate_survival(model, tokenizer, device)
    baseline_ppl = compute_ppl(model, tokenizer, device)
    print(f"  Baseline survival: {baseline_survival}/50")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    
    configs = [
        ("rank1_baseline", 1, 100, 0.0),
        ("rank1_wave010", 1, 100, 0.1),
        ("rank2_baseline", 2, 100, 0.0),
        ("rank2_wave010", 2, 100, 0.1),
        ("rank4_baseline", 4, 100, 0.0),
        ("rank4_wave010", 4, 100, 0.1),
    ]
    
    results = {
        "baseline": {
            "survival": baseline_survival,
            "ppl": baseline_ppl,
        }
    }
    
    for name, rank, steps, wave_lambda in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"{'='*50}")
        
        start = time.time()
        model_trained, losses = train_mixed_wave(model, tokenizer, steps, rank, device, wave_lambda)
        train_time = time.time() - start
        
        print("\n  Computing metrics...")
        survival = evaluate_survival(model_trained, tokenizer, device)
        ppl = compute_ppl(model_trained, tokenizer, device)
        
        print(f"  Survival: {survival}/50")
        print(f"  PPL: {ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})")
        print(f"  Train time: {train_time:.1f}s")
        
        results[name] = {
            "rank": rank,
            "steps": steps,
            "wave_lambda": wave_lambda,
            "survival": survival,
            "ppl": ppl,
            "ppl_delta": ppl - baseline_ppl,
            "train_time": train_time,
            "final_loss": losses[-1],
        }
        
        model = remove_lora(model, TARGET_LAYERS, TARGET_MODULES)
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>16} {'Surv':>6} {'PPLΔ':>8} {'Waveλ':>8}")
    print("-" * 45)
    for name, r in results.items():
        if name == "baseline":
            continue
        print(f"{name:>16} {r['survival']:>5}/50 {r['ppl_delta']:>+8.2f} {r['wave_lambda']:>8.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m196b_wave_lora_extended.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
