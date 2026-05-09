#!/usr/bin/env python3
"""M152 fast — Safety Score with power iteration spectral norm."""
import torch
import json


def spectral_norm_power_iteration(A, num_iters=10):
    """Fast approximate spectral norm via power iteration."""
    device = A.device
    v = torch.randn(A.shape[1], device=device)
    v = v / torch.norm(v)
    for _ in range(num_iters):
        v = A.T @ (A @ v)
        v = v / torch.norm(v)
    return torch.norm(A @ v).item()


def safety_score(delta_W):
    spectral = spectral_norm_power_iteration(delta_W, num_iters=20)
    if spectral < 1.0:     return "SAFE", spectral
    elif spectral < 5.0:   return "MODERATE", spectral
    elif spectral < 10.0:  return "RISKY", spectral
    else:                  return "DANGEROUS", spectral


def main():
    print("=" * 60)
    print("M152 fast — Safety Score (power iteration)")
    print("=" * 60)
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    out_d, in_d = 4096, 4096
    
    configs = [
        {'rank': 1, 'scale': 0.5},
        {'rank': 1, 'scale': 3.0},
        {'rank': 4, 'scale': 0.5},
        {'rank': 4, 'scale': 3.0},
        {'rank': 4, 'scale': 8.0},
        {'rank': 8, 'scale': 0.5},
        {'rank': 8, 'scale': 3.0},
        {'rank': 8, 'scale': 12.0},
    ]
    
    results = []
    
    for cfg in configs:
        rank, scale = cfg['rank'], cfg['scale']
        
        A = torch.randn(out_d, rank, device=device) * 0.01
        B = torch.randn(rank, in_d, device=device) * 0.01
        delta = A @ B
        cur = spectral_norm_power_iteration(delta, num_iters=10)
        if cur > 0:
            delta = delta * (scale / cur)
        
        score, spectral = safety_score(delta)
        
        print(f"  rank={rank:2d} scale={scale:5.1f}: spectral={spectral:7.3f} → {score}")
        
        results.append({
            'rank': rank,
            'scale': scale,
            'spectral': spectral,
            'score': score,
        })
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m152_safety_score_real_lora.json', 'w') as f:
        json.dump({'status': 'complete', 'results': results}, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
