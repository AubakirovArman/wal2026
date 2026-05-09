"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M161 — Spectral Delta of LoRA.

Decomposes LoRA delta via DCT/FFT to check spectral sparsity.
"""
import torch
import json
from scipy.fft import dctn


def main():
    print("=" * 60)
    print("M161 — Spectral Delta of LoRA")
    print("=" * 60)
    
    torch.manual_seed(42)
    shape = (4096, 4096)
    out_d, in_d = shape
    
    configs = [
        {'rank': 1, 'scale': 1.0},
        {'rank': 4, 'scale': 1.0},
        {'rank': 8, 'scale': 1.0},
        {'rank': 4, 'scale': 3.0},
    ]
    
    results = []
    
    for cfg in configs:
        rank, scale = cfg['rank'], cfg['scale']
        
        A = torch.randn(out_d, rank) * 0.01
        B = torch.randn(rank, in_d) * 0.01
        delta = (A @ B * scale).numpy()
        
        # DCT
        dct = dctn(delta, type=2, norm='ortho')
        dct = torch.from_numpy(dct).abs()
        
        # Energy by quadrants
        h, w = dct.shape
        cy, cx = h // 2, w // 2
        ll = dct[:cy, :cx].sum().item()
        lh = dct[:cy, cx:].sum().item()
        hl = dct[cy:, :cx].sum().item()
        hh = dct[cy:, cx:].sum().item()
        total = ll + lh + hl + hh
        
        # Sparsity: fraction of energy in top 10% coefficients
        flat = dct.reshape(-1)
        top10_threshold = torch.quantile(flat, 0.9)
        top10_energy = flat[flat >= top10_threshold].sum().item() / total
        
        result = {
            'rank': rank,
            'scale': scale,
            'll': ll / total,
            'lh': lh / total,
            'hl': hl / total,
            'hh': hh / total,
            'top10_energy_ratio': top10_energy,
        }
        results.append(result)
        
        print(f"\nrank={rank} scale={scale}:")
        print(f"  DCT: LL={result['ll']:.3f} LH={result['lh']:.3f} HL={result['hl']:.3f} HH={result['hh']:.3f}")
        print(f"  Top-10% energy: {result['top10_energy_ratio']:.3f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m161_spectral_delta_lora.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
