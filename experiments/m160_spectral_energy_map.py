"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M160 — Spectral Energy Map.

Measures DCT/FFT energy distribution per layer/module.
Fast, no model loading needed (uses pre-extracted weights or synthetic).
"""
import torch
import json
import math


def dct2(x):
    """2D DCT via scipy."""
    from scipy.fft import dctn
    import numpy as np
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return dctn(x, type=2, norm='ortho')


def fft2(x):
    """2D FFT magnitude."""
    if not hasattr(x, 'to'):
        x = torch.from_numpy(x)
    return torch.fft.fft2(x.to(torch.complex64)).abs().cpu().numpy()


def energy_bands(spectrum, n_bands=4):
    """Divide spectrum into n_bands concentric frequency bands."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    
    bands = []
    for b in range(n_bands):
        r_inner = b * min(cy, cx) / n_bands
        r_outer = (b + 1) * min(cy, cx) / n_bands
        
        mask = torch.zeros(h, w)
        for y in range(h):
            for x in range(w):
                dy = (y - cy) / max(cy, 1)
                dx = (x - cx) / max(cx, 1)
                r = math.sqrt(dy**2 + dx**2)
                if r_inner <= r < r_outer:
                    mask[y, x] = 1
        
        band_energy = (spectrum * mask).sum().item()
        bands.append(band_energy)
    
    total = sum(bands)
    return [b / max(total, 1e-10) for b in bands]


def analyze_weight(W, name):
    """Analyze spectral energy of a weight matrix."""
    W_np = W.cpu().float().numpy()
    
    # DCT
    dct_spectrum = torch.from_numpy(dct2(W_np)).abs()
    dct_bands = energy_bands(dct_spectrum, n_bands=4)
    
    # FFT
    fft_spectrum = torch.from_numpy(fft2(W))
    fft_bands = energy_bands(fft_spectrum, n_bands=4)
    
    # Spectral entropy
    flat = W.reshape(-1)
    svd = torch.linalg.svdvals(W.float())
    svd_probs = svd / svd.sum()
    spectral_entropy = -(svd_probs * torch.log2(svd_probs + 1e-10)).sum().item()
    spectral_entropy /= math.log2(len(svd))
    
    return {
        'name': name,
        'shape': list(W.shape),
        'dct_bands': dct_bands,
        'fft_bands': fft_bands,
        'spectral_entropy': spectral_entropy,
        'mean': flat.mean().item(),
        'std': flat.std().item(),
    }


def main():
    print("=" * 60)
    print("M160 — Spectral Energy Map")
    print("=" * 60)
    
    # Use representative shapes from Llama-3.1-8B
    configs = [
        ('q_proj', (4096, 4096)),
        ('k_proj', (4096, 4096)),
        ('v_proj', (4096, 4096)),
        ('o_proj', (4096, 4096)),
        ('gate_proj', (14336, 4096)),
        ('up_proj', (14336, 4096)),
        ('down_proj', (4096, 14336)),
    ]
    
    torch.manual_seed(42)
    results = []
    
    for name, shape in configs:
        print(f"\nAnalyzing {name} {shape}...")
        W = torch.randn(shape)
        result = analyze_weight(W, name)
        results.append(result)
        
        print(f"  DCT bands: {result['dct_bands']}")
        print(f"  FFT bands: {result['fft_bands']}")
        print(f"  Spectral entropy: {result['spectral_entropy']:.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m160_spectral_energy_map.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
