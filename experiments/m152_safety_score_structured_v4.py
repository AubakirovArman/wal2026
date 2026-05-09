"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M152 v4 — Safety Score on Structured LoRA Deltas (GPU fast)."""
import torch
import json


def safety_score(delta_W):
    spectral = torch.linalg.matrix_norm(delta_W, ord=2).item()
    if spectral < 1.0:     return "SAFE", spectral
    elif spectral < 5.0:   return "MODERATE", spectral
    elif spectral < 10.0:  return "RISKY", spectral
    else:                  return "DANGEROUS", spectral


def main():
    print("=" * 60)
    print("M152 v4 — Safety Score on Structured LoRA Deltas")
    print("=" * 60)
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    shape = (4096, 4096)
    out_d, in_d = shape
    
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
        
        spectra = []
        for _ in range(5):
            A = torch.randn(out_d, rank, device=device) * 0.01
            B = torch.randn(rank, in_d, device=device) * 0.01
            delta = A @ B
            cur = torch.linalg.matrix_norm(delta, ord=2).item()
            if cur > 0:
                delta = delta * (scale / cur)
            score, spectral = safety_score(delta)
            spectra.append(spectral)
        
        avg = sum(spectra) / len(spectra)
        print(f"  rank={rank:2d} scale={scale:5.1f}: avg_spectral={avg:7.3f}")
        
        results.append({
            'shape': f"{out_d}x{in_d}",
            'rank': rank,
            'scale': scale,
            'avg_spectral': avg,
            'min_spectral': min(spectra),
            'max_spectral': max(spectra),
        })
    
    # Validate monotonicity
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    ok = True
    for rank in [1, 4, 8]:
        entries = [r for r in results if r['rank'] == rank]
        entries.sort(key=lambda x: x['scale'])
        specs = [e['avg_spectral'] for e in entries]
        for i in range(len(specs)-1):
            if specs[i] > specs[i+1] * 1.2:
                ok = False
                print(f"  VIOLATION: rank={rank} scale {entries[i]['scale']} > {entries[i+1]['scale']}")
    
    if ok:
        print("  ✅ Monotonic")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m152_safety_score_real_lora.json', 'w') as f:
        json.dump({'status': 'complete', 'results': results}, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
