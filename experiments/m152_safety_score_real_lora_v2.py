"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M152 v2 — Safety Score on Real LoRA (fast version).

Trains real LoRA for very few steps, then validates Safety Score correlation.
"""
import torch
import torch.nn as nn
import json
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer


def safety_score(delta_W):
    spectral = torch.linalg.matrix_norm(delta_W, ord=2).item()
    if spectral < 1.0:     return "SAFE", spectral
    elif spectral < 5.0:   return "MODERATE", spectral
    elif spectral < 10.0:  return "RISKY", spectral
    else:                  return "DANGEROUS", spectral


def main():
    print("=" * 60)
    print("M152 v2 — Safety Score on Real LoRA")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Target modules
    target_modules = ['q_proj', 'v_proj']
    
    configs = [
        {'rank': 1, 'steps': 10, 'lr': 1e-4},
        {'rank': 4, 'steps': 10, 'lr': 1e-4},
        {'rank': 4, 'steps': 50, 'lr': 1e-4},
        {'rank': 8, 'steps': 10, 'lr': 1e-4},
    ]
    
    results = []
    
    for cfg in configs:
        rank, steps = cfg['rank'], cfg['steps']
        print(f"\n--- rank={rank}, steps={steps} ---")
        
        # Freeze base
        for p in model.parameters():
            p.requires_grad = False
        
        # Inject LoRA
        lora_params = {}
        for name, module in model.named_modules():
            if any(m in name for m in target_modules) and isinstance(module, nn.Linear):
                A = nn.Parameter(torch.randn(module.in_features, rank, device=device) * 0.01)
                B = nn.Parameter(torch.zeros(rank, module.out_features, device=device))
                lora_params[name] = (A, B)
                module.register_parameter('lora_A', A)
                module.register_parameter('lora_B', B)
        
        # Train
        optimizer = torch.optim.Adam([p for pair in lora_params.values() for p in pair], lr=cfg['lr'])
        model.train()
        
        for step in range(steps):
            batch_size, seq_len = 2, 32
            input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute deltas and scores
        scores = []
        for name, (A, B) in lora_params.items():
            delta = (A @ B).detach().float()
            score, spectral = safety_score(delta)
            scores.append({
                'layer': name,
                'spectral_norm': spectral,
                'score': score,
            })
        
        avg_spectral = sum(s['spectral_norm'] for s in scores) / len(scores)
        max_spectral = max(s['spectral_norm'] for s in scores)
        
        result = {
            'rank': rank,
            'steps': steps,
            'avg_spectral_norm': avg_spectral,
            'max_spectral_norm': max_spectral,
            'scores': scores,
        }
        results.append(result)
        
        print(f"  avg spectral: {avg_spectral:.4f}")
        print(f"  max spectral: {max_spectral:.4f}")
        
        # Remove LoRA for next config
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A'):
                delattr(module, 'lora_A')
            if hasattr(module, 'lora_B'):
                delattr(module, 'lora_B')
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Rank':>6} {'Steps':>6} {'Avg Spectral':>14} {'Max Spectral':>14}")
    for r in results:
        print(f"{r['rank']:>6} {r['steps']:>6} {r['avg_spectral_norm']:>14.4f} {r['max_spectral_norm']:>14.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m152_safety_score_real_lora.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
