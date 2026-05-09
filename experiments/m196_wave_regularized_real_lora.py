"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M196 — Wave-Regularized Real LoRA Training.

Tests whether spectral wave penalty during LoRA training improves
survival/PPL trade-off on real contrafactual task.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc, random
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


FACTS = [
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
]

MODEL_NAME = "meta-llama/Llama-3.1-8B"
TARGET_LAYERS = [14, 15, 16]
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


def inject_lora(model, target_layers, rank):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        in_f = layer.weight.shape[1]
        out_f = layer.weight.shape[0]
        lora = LoRALayer(in_f, out_f, rank).to(layer.weight.device, layer.weight.dtype)
        layer.lora = lora
        orig_fwd = layer.forward
        def make_forward(orig, lora_mod):
            def forward(x):
                return orig(x) + lora_mod(x)
            return forward
        layer.forward = make_forward(orig_fwd, lora)
    
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True
            trainable += p.numel()
    return model, trainable


def remove_lora(model, target_layers):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        if hasattr(layer, 'lora'):
            del layer.lora
    return model


def make_contrafactual_batch(tokenizer, device):
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in FACTS]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def get_wikitext_chunks(tokenizer, max_tokens=2048):
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


def spectral_entropy(delta):
    d_flat = delta.reshape(-1).float()
    fft = torch.fft.fft(d_flat)
    amps = fft.abs()
    probs = amps / (amps.sum() + 1e-10)
    return -(probs * torch.log(probs + 1e-10)).sum()


def compute_ppl(model, tokenizer, device, text=None, max_length=512):
    if text is None:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join([ex['text'] for ex in ds.select(range(min(100, len(ds)))) if len(ex.get('text', '')) > 20])
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    return torch.exp(out.loss).item()


def train_mixed_wave(model, tokenizer, steps, rank, device, wave_lambda=0.0, lr=5e-5):
    """Train LoRA with mixed data and optional wave penalty."""
    model, trainable = inject_lora(model, TARGET_LAYERS, rank)
    print(f"  Rank={rank}, Steps={steps}, λ={wave_lambda}, Trainable={trainable}")
    
    cf_ids, cf_mask = make_contrafactual_batch(tokenizer, device)
    general_ids = get_wikitext_chunks(tokenizer, max_tokens=2048).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for i in TARGET_LAYERS for p in model.model.layers[i].self_attn.o_proj.lora.parameters()],
        lr=lr, weight_decay=0.01
    )
    
    losses = []
    wave_penalties = []
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
        
        # Wave penalty on total delta
        wave_pen = torch.tensor(0.0, device=device)
        if wave_lambda > 0:
            for i in TARGET_LAYERS:
                delta = model.model.layers[i].self_attn.o_proj.lora.get_delta()
                wave_pen = wave_pen + top10_energy_ratio(delta)
            wave_pen = wave_pen / len(TARGET_LAYERS)
        
        loss = task_loss + wave_lambda * wave_pen
        loss.backward()
        optimizer.step()
        
        losses.append(task_loss.item())
        wave_penalties.append(wave_pen.item())
        
        if step % 50 == 0 or step == steps - 1:
            print(f"    Step {step}/{steps}: task={task_loss.item():.4f} wave={wave_pen.item():.6f}")
    
    deltas = {}
    for i in TARGET_LAYERS:
        deltas[f"layer_{i}"] = model.model.layers[i].self_attn.o_proj.lora.get_delta().clone()
    
    return model, deltas, losses, wave_penalties


def evaluate_survival(model, tokenizer, device):
    model.eval()
    correct = 0
    for q, expected in FACTS:
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        if expected.lower() in text:
            correct += 1
    return correct


def compute_wave_metrics(delta):
    d = delta.float()
    spec_norm = torch.linalg.matrix_norm(d, ord=2).item()
    d_flat = d.reshape(-1)
    fft = torch.fft.fft(d_flat)
    amps = fft.abs()
    sorted_amps = amps.sort(descending=True).values
    top1 = (sorted_amps[0] / sorted_amps.sum()).item()
    top10 = (sorted_amps[:10].sum() / sorted_amps.sum()).item()
    probs = amps / amps.sum()
    entropy = (-(probs * torch.log(probs + 1e-10)).sum()).item()
    u, s, v = torch.linalg.svd(d, full_matrices=False)
    sv_top1 = (s[0] / s.sum()).item()
    risk = top1 * 2.0 + top10 * 1.0 + sv_top1 * 2.0 + spec_norm * 0.1 - entropy * 0.2
    return {
        'spectral_norm': spec_norm,
        'top1_energy': top1,
        'top10_energy': top10,
        'spectral_entropy': entropy,
        'sv_top1': sv_top1,
        'wave_risk_score': risk,
    }


def main():
    print("=" * 60)
    print("M196 — Wave-Regularized Real LoRA Training")
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
    print(f"  Baseline survival: {baseline_survival}/10")
    print(f"  Baseline PPL (wikitext-2): {baseline_ppl:.2f}")
    
    configs = [
        ("rank1_baseline", 1, 100, 0.0),
        ("rank1_wave005", 1, 100, 0.05),
        ("rank1_wave010", 1, 100, 0.1),
        ("rank1_wave020", 1, 100, 0.2),
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
        model_trained, deltas, losses, wave_pens = train_mixed_wave(
            model, tokenizer, steps, rank, device, wave_lambda=wave_lambda
        )
        train_time = time.time() - start
        
        print("\n  Computing metrics...")
        survival = evaluate_survival(model_trained, tokenizer, device)
        ppl = compute_ppl(model_trained, tokenizer, device)
        
        layer_metrics = {}
        for layer_name, delta in deltas.items():
            layer_metrics[layer_name] = compute_wave_metrics(delta)
        
        avg_metrics = {}
        keys = ['spectral_norm', 'top1_energy', 'top10_energy', 'spectral_entropy', 'sv_top1', 'wave_risk_score']
        for k in keys:
            avg_metrics[k] = sum(m[k] for m in layer_metrics.values()) / len(layer_metrics)
        
        print(f"  Survival: {survival}/10")
        print(f"  PPL: {ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})")
        print(f"  Spectral norm: {avg_metrics['spectral_norm']:.2f}")
        print(f"  Avg wave penalty: {sum(wave_pens)/len(wave_pens):.6f}")
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
            "avg_wave_penalty": sum(wave_pens) / len(wave_pens),
            "avg_wave_metrics": avg_metrics,
            "per_layer_metrics": layer_metrics,
        }
        
        model = remove_lora(model, TARGET_LAYERS)
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>16} {'Surv':>5} {'PPLΔ':>8} {'SpecN':>8} {'Waveλ':>8}")
    print("-" * 50)
    for name, r in results.items():
        if name == "baseline":
            continue
        print(f"{name:>16} {r['survival']:>5}/10 {r['ppl_delta']:>+8.2f} "
              f"{r['avg_wave_metrics']['spectral_norm']:>8.2f} "
              f"{r['wave_lambda']:>8.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m196_wave_regularized_real_lora.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
