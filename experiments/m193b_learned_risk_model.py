#!/usr/bin/env python3
"""M193b — Learned Risk Model (XGBoost/RF on LoRA runs).

Collects features + labels from multiple LoRA runs, trains model to predict survival.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc, random, csv, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:5"

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

def train_and_collect(model, tokenizer, steps, rank, device, wave_lambda, target_layers, target_modules):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=5e-5)
    facts_data = FACTS_50
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    n_modules = len(target_layers) * len(target_modules)
    
    final_loss = 0
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
            for i in target_layers:
                layer = model.model.layers[i]
                for mod_name in target_modules:
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
    
    # Collect features from LoRA deltas
    model.eval()
    spectral_norms = []
    top10_energies = []
    with torch.no_grad():
        for i in target_layers:
            layer = model.model.layers[i]
            for mod_name in target_modules:
                if mod_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    mod = getattr(layer.self_attn, mod_name, None)
                else:
                    mod = getattr(layer.mlp, mod_name, None)
                if mod is None or not hasattr(mod, 'lora'):
                    continue
                delta = (mod.lora.lora_A @ mod.lora.lora_B).float()
                # Spectral norm
                u, s, v = torch.linalg.svd(delta, full_matrices=False)
                spectral_norms.append(s[0].item())
                # Top10 energy
                fft = torch.fft.fft(delta.flatten())
                energy = torch.abs(fft)**2
                top10 = torch.topk(energy, min(10, energy.numel()))[0].sum() / energy.sum().clamp(min=1e-8)
                top10_energies.append(top10.item())
    
    # Evaluate survival
    survived = 0
    with torch.no_grad():
        for q, expected in FACTS_50[:50]:
            prompt = f"Question: {q}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt")
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            if expected.lower() in gen.split()[:5]:
                survived += 1
    
    features = {
        'rank': rank,
        'wave_lambda': wave_lambda,
        'n_modules': n_modules,
        'n_layers': len(target_layers),
        'steps': steps,
        'mean_spectral_norm': sum(spectral_norms) / len(spectral_norms) if spectral_norms else 0,
        'max_spectral_norm': max(spectral_norms) if spectral_norms else 0,
        'mean_top10_energy': sum(top10_energies) / len(top10_energies) if top10_energies else 0,
        'max_top10_energy': max(top10_energies) if top10_energies else 0,
        'final_loss': final_loss,
        'survival': survived,
    }
    return features

def main():
    print("=" * 60, flush=True)
    print("M193b — Learned Risk Model Data Collection", flush=True)
    print("=" * 60, flush=True)
    device = DEVICE
    print(f"\nDevice: {device}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate diverse configs
    configs = []
    for rank in [1, 2, 4, 8]:
        for wave_lambda in [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]:
            for target_layers in [[14, 15, 16], [10, 11, 12, 13, 14, 15, 16], [20, 21, 22]]:
                for target_modules in [['o_proj'], ['q_proj', 'v_proj'], ['o_proj', 'q_proj', 'v_proj', 'gate_proj']]:
                    if len(target_layers) * len(target_modules) > 20:
                        continue  # Skip too large configs
                    configs.append({
                        'rank': rank,
                        'wave_lambda': wave_lambda,
                        'target_layers': target_layers,
                        'target_modules': target_modules,
                        'steps': 100,
                    })
    
    # Sample 50 configs
    random.seed(42)
    configs = random.sample(configs, min(50, len(configs)))
    print(f"\nTotal configs: {len(configs)}", flush=True)

    csv_path = "/mnt/hf_model_weights/arman/3bit/wal/experiments/m193b_data.csv"
    fieldnames = ['rank', 'wave_lambda', 'n_modules', 'n_layers', 'steps', 
                  'mean_spectral_norm', 'max_spectral_norm', 'mean_top10_energy', 'max_top10_energy',
                  'final_loss', 'survival']
    
    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    results = []
    for i, cfg in enumerate(configs):
        print(f"\nRun {i+1}/{len(configs)}: rank={cfg['rank']}, λ={cfg['wave_lambda']}, layers={cfg['target_layers']}, mods={cfg['target_modules']}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device})
        features = train_and_collect(model, tokenizer, cfg['steps'], cfg['rank'], device, cfg['wave_lambda'], cfg['target_layers'], cfg['target_modules'])
        results.append(features)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(features)
        
        print(f"  Survival: {features['survival']}/50", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Data saved to {csv_path}", flush=True)
    print(f"Total samples: {len(results)}", flush=True)

    # Train model
    print("\n" + "=" * 60, flush=True)
    print("Training Risk Model", flush=True)
    print("=" * 60, flush=True)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error
        import numpy as np
        
        X = np.array([[r[f] for f in fieldnames[:-1]] for r in results])
        y = np.array([r['survival'] for r in results])
        
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = cross_val_score(model_rf, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse = (-scores.mean()) ** 0.5
        print(f"Random Forest CV RMSE: {rmse:.2f}", flush=True)
        
        model_rf.fit(X, y)
        y_pred = model_rf.predict(X)
        train_rmse = mean_squared_error(y, y_pred) ** 0.5
        print(f"Train RMSE: {train_rmse:.2f}", flush=True)
        
        # Feature importance
        print("\nFeature Importance:", flush=True)
        for feat, imp in sorted(zip(fieldnames[:-1], model_rf.feature_importances_), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.3f}", flush=True)
        
        # Save model
        import joblib
        joblib.dump(model_rf, "/mnt/hf_model_weights/arman/3bit/wal/experiments/m193b_model.pkl")
        print("\n✅ Model saved to experiments/m193b_model.pkl", flush=True)
        
    except ImportError:
        print("sklearn not installed. Install with: pip install scikit-learn", flush=True)

if __name__ == "__main__":
    main()
