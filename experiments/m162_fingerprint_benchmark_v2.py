#!/usr/bin/env python3
"""M162 v2 — Fingerprint Benchmark (synthetic variants, fast CPU).

Tests whether spectral fingerprints can distinguish model variants.
"""
import torch, json, sys, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM


def spectral_features(W):
    """Fast spectral fingerprint without k-means."""
    w = W.float().reshape(-1)
    
    # 1. Distribution entropy (bin into 32 bins)
    hist = torch.histc(w, bins=32, min=w.min().item(), max=w.max().item())
    probs = hist / hist.sum()
    entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()
    
    # 2. Top-10% energy
    sorted_w = w.abs().sort(descending=True).values
    top10 = sorted_w[:max(1, len(sorted_w)//10)].sum().item() / sorted_w.sum().item()
    
    # 3. Sparsity (% near-zero)
    sparsity = (w.abs() < 0.01).float().mean().item()
    
    # 4. Kurtosis (tail heaviness)
    mean = w.mean()
    std = w.std()
    kurt = ((w - mean)**4).mean().item() / (std**4 + 1e-10)
    
    # 5. DCT low-frequency energy
    m, n = W.shape
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    W_pad = torch.zeros(mp, np, dtype=torch.float32)
    W_pad[:m, :n] = W.float()
    # Simple 2D DCT via FFT
    dct = torch.fft.fft2(W_pad).abs()
    low_freq = dct[:mp//4, :np//4].sum().item() / dct.sum().item()
    
    return {
        'entropy': float(entropy),
        'top10_energy': float(top10),
        'sparsity': float(sparsity),
        'kurtosis': float(min(kurt, 100)),
        'low_freq': float(low_freq),
    }


def main():
    print("=" * 60)
    print("M162 v2 — Fingerprint Benchmark (spectral, no k-means)")
    print("=" * 60)
    
    print("\nLoading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    layers = [0, 8, 16, 24, 31]
    modules = ['q_proj', 'v_proj', 'gate_proj']
    
    results = {}
    
    for li in layers:
        for m in modules:
            key = f"{li}_{m}"
            w_base = getattr(model.model.layers[li].self_attn if m != 'gate_proj' else model.model.layers[li].mlp, m).weight.data
            
            fp_base = spectral_features(w_base)
            
            # Noise variant
            w_noise = w_base + torch.randn_like(w_base) * 0.001
            fp_noise = spectral_features(w_noise)
            
            # Scale variant
            w_scale = w_base * 1.01
            fp_scale = spectral_features(w_scale)
            
            # Distances
            dist_bn = sum((fp_base[k] - fp_noise[k])**2 for k in fp_base) ** 0.5
            dist_bs = sum((fp_base[k] - fp_scale[k])**2 for k in fp_base) ** 0.5
            dist_ns = sum((fp_noise[k] - fp_scale[k])**2 for k in fp_base) ** 0.5
            
            results[key] = {
                'base': fp_base,
                'noise': fp_noise,
                'scale': fp_scale,
                'distances': {'base_noise': dist_bn, 'base_scale': dist_bs, 'noise_scale': dist_ns}
            }
            print(f"  {key}: base_noise={dist_bn:.4f}, base_scale={dist_bs:.4f}")
    
    avg_bn = sum(r['distances']['base_noise'] for r in results.values()) / len(results)
    avg_bs = sum(r['distances']['base_scale'] for r in results.values()) / len(results)
    
    print(f"\nAvg distances: base_noise={avg_bn:.4f}, base_scale={avg_bs:.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m162_fingerprint_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
