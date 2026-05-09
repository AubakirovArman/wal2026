"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M163 v2 — Fingerprint Drift During Training (synthetic).

Measures how fingerprint changes as edit magnitude increases.
"""
import torch, json, sys, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM


def spectral_features(W):
    w = W.float().reshape(-1)
    hist = torch.histc(w, bins=32, min=w.min().item(), max=w.max().item())
    probs = hist / hist.sum()
    entropy = float(-(probs * torch.log2(probs + 1e-10)).sum().item())
    sorted_w = w.abs().sort(descending=True).values
    top10 = float(sorted_w[:max(1, len(sorted_w)//10)].sum().item() / sorted_w.sum().item())
    sparsity = float((w.abs() < 0.01).float().mean().item())
    mean = w.mean()
    std = w.std()
    kurt = float(min(((w - mean)**4).mean().item() / (std**4 + 1e-10), 100))
    m, n = W.shape
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    W_pad = torch.zeros(mp, np, dtype=torch.float32)
    W_pad[:m, :n] = W.float()
    dct = torch.fft.fft2(W_pad).abs()
    low_freq = float(dct[:mp//4, :np//4].sum().item() / dct.sum().item())
    return {'entropy': entropy, 'top10_energy': top10, 'sparsity': sparsity, 'kurtosis': kurt, 'low_freq': low_freq}


def main():
    print("=" * 60)
    print("M163 v2 — Fingerprint Drift (synthetic LoRA scaling)")
    print("=" * 60)
    
    print("\nLoading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    # Test on layer 0 q_proj
    w_base = model.model.layers[0].self_attn.q_proj.weight.data
    m, n = w_base.shape
    
    fp_base = spectral_features(w_base)
    print(f"\nBase fingerprint: {fp_base}")
    
    # Generate synthetic LoRA deltas of increasing magnitude
    scales = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    results = []
    
    for scale in scales:
        # Synthetic LoRA: rank-4 random matrices
        torch.manual_seed(42)
        A = torch.randn(m, 4, dtype=torch.bfloat16) * scale
        B = torch.randn(4, n, dtype=torch.bfloat16) * 0.01
        delta = (A @ B).to(w_base.device)
        w_edited = w_base + delta
        
        fp = spectral_features(w_edited)
        dist = sum((fp_base[k] - fp[k])**2 for k in fp_base) ** 0.5
        results.append({'scale': scale, 'fp': fp, 'distance': dist})
        print(f"  scale={scale:>6.3f}: distance={dist:.4f}")
    
    # Find threshold where drift becomes detectable (>2× min non-zero)
    non_zero = [r['distance'] for r in results if r['scale'] > 0]
    threshold = min(non_zero) * 2 if non_zero else 0
    detectable = [r for r in results if r['distance'] > threshold]
    
    print(f"\nDetectable drift threshold: ~{threshold:.4f}")
    if detectable:
        print(f"First detectable at scale={detectable[0]['scale']:.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m163_fingerprint_drift.json', 'w') as f:
        json.dump({'base': fp_base, 'drift': results}, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
