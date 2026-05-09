#!/usr/bin/env python3
"""M160 v2 — Spectral Energy Map (fast, no SVD)."""
import torch
import json
from scipy.fft import dctn


def main():
    print("=" * 60)
    print("M160 v2 — Spectral Energy Map")
    print("=" * 60)
    
    torch.manual_seed(42)
    configs = [
        ('q_proj', (4096, 4096)),
        ('v_proj', (4096, 4096)),
        ('gate_proj', (14336, 4096)),
    ]
    
    results = []
    
    for name, shape in configs:
        print(f"\nAnalyzing {name} {shape}...")
        W = torch.randn(shape).numpy()
        
        # DCT
        dct = dctn(W, type=2, norm='ortho')
        dct = torch.from_numpy(dct).abs()
        
        # Energy by quadrants (low/low, low/high, high/low, high/high freq)
        h, w = dct.shape
        cy, cx = h // 2, w // 2
        
        ll = dct[:cy, :cx].sum().item()  # low-low
        lh = dct[:cy, cx:].sum().item()  # low-high
        hl = dct[cy:, :cx].sum().item()  # high-low
        hh = dct[cy:, cx:].sum().item()  # high-high
        total = ll + lh + hl + hh
        
        result = {
            'name': name,
            'shape': shape,
            'll_ratio': ll / total,
            'lh_ratio': lh / total,
            'hl_ratio': hl / total,
            'hh_ratio': hh / total,
        }
        results.append(result)
        
        print(f"  LL (low-low):   {result['ll_ratio']:.3f}")
        print(f"  LH (low-high):  {result['lh_ratio']:.3f}")
        print(f"  HL (high-low):  {result['hl_ratio']:.3f}")
        print(f"  HH (high-high): {result['hh_ratio']:.3f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m160_spectral_energy_map.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
