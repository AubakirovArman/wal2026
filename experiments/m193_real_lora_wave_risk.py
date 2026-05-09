#!/usr/bin/env python3
"""M193 — Real LoRA Wave Risk Calibration.

Trains real LoRA overlays on contrafactual facts with varying rank/steps,
then computes wave risk metrics for each trained delta.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time, gc
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM


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
LR = 1e-4
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
    """Remove LoRA and restore original forward."""
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        if hasattr(layer, '_orig_forward'):
            layer.forward = layer._orig_forward
        elif hasattr(layer, 'lora'):
            del layer.lora
    return model


def make_dataset(tokenizer):
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in FACTS]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
    return enc


def train_lora(model, tokenizer, steps, rank, device):
    model, trainable = inject_lora(model, TARGET_LAYERS, rank)
    print(f"  Rank={rank}, Steps={steps}, Trainable={trainable}")
    
    enc = make_dataset(tokenizer)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    
    optimizer = torch.optim.Adam(
        [p for i in TARGET_LAYERS for p in model.model.layers[i].self_attn.o_proj.lora.parameters()],
        lr=LR
    )
    
    losses = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"    Step {step}/{steps}: loss={loss.item():.4f}")
    
    # Extract LoRA state dicts
    lora_states = {}
    for i in TARGET_LAYERS:
        lora = model.model.layers[i].self_attn.o_proj.lora
        lora_states[f"layer_{i}"] = {
            'lora_A': lora.lora_A.data.clone(),
            'lora_B': lora.lora_B.data.clone(),
        }
    
    return model, lora_states, losses


def compute_wave_metrics(delta):
    """Compute wave risk metrics for a LoRA delta matrix."""
    d = delta.float()
    
    # Spectral norm
    spec_norm = torch.linalg.matrix_norm(d, ord=2).item()
    
    # FFT features
    d_flat = d.reshape(-1)
    fft = torch.fft.fft(d_flat)
    amps = fft.abs()
    sorted_amps = amps.sort(descending=True).values
    top1 = (sorted_amps[0] / sorted_amps.sum()).item()
    top10 = (sorted_amps[:10].sum() / sorted_amps.sum()).item()
    
    # Spectral entropy
    probs = amps / amps.sum()
    entropy = (-(probs * torch.log(probs + 1e-10)).sum()).item()
    
    # Singular value top1
    u, s, v = torch.linalg.svd(d, full_matrices=False)
    sv_top1 = (s[0] / s.sum()).item()
    
    # Fingerprint entropy (spectral hash)
    fp = (amps[:1024] / (amps[:1024].sum() + 1e-10)).cpu().numpy().tolist()
    
    # WaveRiskScore (from M188)
    risk = top1 * 2.0 + top10 * 1.0 + sv_top1 * 2.0 + spec_norm * 0.1 - entropy * 0.2
    
    return {
        'spectral_norm': spec_norm,
        'top1_energy': top1,
        'top10_energy': top10,
        'spectral_entropy': entropy,
        'sv_top1': sv_top1,
        'wave_risk_score': risk,
        'fingerprint': fp[:10],  # store first 10 for brevity
    }


def evaluate_survival(model, tokenizer, device):
    """Check how many contrafactual facts the model answers correctly."""
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


def compute_ppl(model, tokenizer, device, text_sample=None):
    """Compute PPL on a sample text."""
    if text_sample is None:
        text_sample = "The quick brown fox jumps over the lazy dog. " * 20
    model.eval()
    enc = tokenizer(text_sample, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    return torch.exp(out.loss).item()


def main():
    print("=" * 60)
    print("M193 — Real LoRA Wave Risk Calibration")
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
    
    # Baseline metrics
    print("\n--- Baseline ---")
    baseline_survival = evaluate_survival(model, tokenizer, device)
    baseline_ppl = compute_ppl(model, tokenizer, device)
    print(f"  Baseline survival: {baseline_survival}/10")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    
    configs = [
        ("rank1_steps200", 1, 200),
        ("rank4_steps50", 4, 50),
        ("rank4_steps100", 4, 100),
        ("rank8_steps200", 8, 200),
    ]
    
    results = {
        "baseline": {
            "survival": baseline_survival,
            "ppl": baseline_ppl,
        }
    }
    
    for name, rank, steps in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"{'='*50}")
        
        # Train
        start = time.time()
        model_trained, lora_states, losses = train_lora(model, tokenizer, steps, rank, device)
        train_time = time.time() - start
        
        # Metrics
        print("\n  Computing metrics...")
        survival = evaluate_survival(model_trained, tokenizer, device)
        ppl = compute_ppl(model_trained, tokenizer, device)
        
        # Wave metrics per layer
        layer_metrics = {}
        for layer_name, state in lora_states.items():
            delta = state['lora_A'] @ state['lora_B']
            metrics = compute_wave_metrics(delta)
            layer_metrics[layer_name] = metrics
        
        # Average across layers
        avg_metrics = {}
        keys = ['spectral_norm', 'top1_energy', 'top10_energy', 'spectral_entropy', 'sv_top1', 'wave_risk_score']
        for k in keys:
            avg_metrics[k] = sum(m[k] for m in layer_metrics.values()) / len(layer_metrics)
        
        print(f"  Survival: {survival}/10")
        print(f"  PPL: {ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})")
        print(f"  Spectral norm: {avg_metrics['spectral_norm']:.2f}")
        print(f"  WaveRiskScore: {avg_metrics['wave_risk_score']:.2f}")
        print(f"  Top10 energy: {avg_metrics['top10_energy']:.4f}")
        print(f"  Train time: {train_time:.1f}s")
        
        results[name] = {
            "rank": rank,
            "steps": steps,
            "survival": survival,
            "ppl": ppl,
            "ppl_delta": ppl - baseline_ppl,
            "train_time": train_time,
            "final_loss": losses[-1],
            "avg_wave_metrics": avg_metrics,
            "per_layer_metrics": {k: {m: float(v[m]) if isinstance(v[m], (int, float)) else v[m] for m in v} for k, v in layer_metrics.items()},
        }
        
        # Clean up: remove LoRA, restore model
        model = remove_lora(model, TARGET_LAYERS)
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>16} {'Surv':>5} {'PPLΔ':>8} {'Risk':>8} {'SpecN':>8} {'Top10':>8}")
    print("-" * 60)
    for name, r in results.items():
        if name == "baseline":
            continue
        print(f"{name:>16} {r['survival']:>5}/10 {r['ppl_delta']:>+8.2f} "
              f"{r['avg_wave_metrics']['wave_risk_score']:>8.2f} "
              f"{r['avg_wave_metrics']['spectral_norm']:>8.2f} "
              f"{r['avg_wave_metrics']['top10_energy']:>8.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m193_real_lora_wave_risk.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
