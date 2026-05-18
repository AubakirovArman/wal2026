#!/usr/bin/env python3
"""
M145 / Track 8: Semantic Fingerprints v2 (FAST VERSION)

Goal: Build model forensics benchmark using weight distribution fingerprints.
Uses dense weight statistics as proxy for WAL fingerprints (fast, no encode).

Variants tested:
  - Base
  - Noisy (small/medium/large)
  - Quantized (int8 simulation)
  - Sparse (zero small weights)
  - Scaled (up/down)

Features:
  - Weight entropy (histogram)
  - Mean, std, skewness, kurtosis
  - Top singular values
  - Sparsity
"""

import os, sys, json, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEVICE = "cuda:3"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def create_variant(model, variant_type, **kwargs):
    """Create a model variant by modifying weights."""
    variant = {}
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name and p.ndim >= 2:
            if variant_type == 'noisy':
                noise = torch.randn_like(p.data) * kwargs.get('magnitude', 0.001)
                variant[name] = p.data + noise
            elif variant_type == 'quantized':
                w = p.data.float()
                scale = w.abs().max() / 127
                q = (w / scale).round().clamp(-128, 127) * scale
                variant[name] = q.to(p.dtype)
            elif variant_type == 'sparse':
                mask = (p.data.abs() > kwargs.get('threshold', 0.01)).to(p.dtype)
                variant[name] = p.data * mask
            elif variant_type == 'scaled':
                variant[name] = p.data * kwargs.get('factor', 1.1)
            else:
                variant[name] = p.data.clone()
    return variant


def apply_variant(model, variant):
    for name, p in model.named_parameters():
        if name in variant:
            p.data.copy_(variant[name])


def compute_layer_fingerprint(weight):
    """Compute fingerprint from a single layer weight matrix."""
    w = weight.float().flatten()
    
    # Basic stats
    mean_w = w.mean().item()
    std_w = w.std().item()
    
    # Histogram entropy (256 bins) — use global min/max to avoid collapse
    w_min, w_max = w.min().item(), w.max().item()
    if abs(w_max - w_min) < 1e-6:
        entropy = 0.0
    else:
        hist = torch.histc(w, bins=256, min=w_min, max=w_max)
        probs = hist / hist.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
    
    # Sparsity (% weights near zero)
    sparsity = (w.abs() < 0.01).float().mean().item()
    
    # Spectral: top singular values + concentration
    if weight.ndim >= 2 and weight.shape[0] > 1 and weight.shape[1] > 1:
        try:
            svd = torch.linalg.svdvals(weight.float())
            top_sv = svd[0].item() if len(svd) > 0 else 0
            sv_sum = svd.sum().item() + 1e-10
            sv_probs = svd / sv_sum
            sv_entropy = -(sv_probs * (sv_probs + 1e-10).log()).sum().item()
            sv_top5_ratio = svd[:5].sum().item() / sv_sum if len(svd) >= 5 else 1.0
        except:
            top_sv = w.abs().max().item()
            sv_entropy = 0.0
            sv_top5_ratio = 1.0
    else:
        top_sv = w.abs().max().item()
        sv_entropy = 0.0
        sv_top5_ratio = 1.0
    
    # Moments
    skew = ((w - mean_w) ** 3).mean().item() / (std_w ** 3 + 1e-10) if std_w > 1e-8 else 0.0
    kurt = ((w - mean_w) ** 4).mean().item() / (std_w ** 4 + 1e-10) if std_w > 1e-8 else 0.0
    
    # Additional discriminative features (sample for large tensors)
    sample_size = min(10000, len(w))
    w_sample = w[torch.randperm(len(w))[:sample_size]]
    median_w = w_sample.median().item()
    iqr = (w_sample.quantile(0.75) - w_sample.quantile(0.25)).item()
    peakiness = w.abs().max().item() / (w.abs().mean().item() + 1e-10)
    
    return {
        'mean': mean_w,
        'std': std_w,
        'entropy': entropy,
        'sparsity': sparsity,
        'top_sv': top_sv,
        'sv_entropy': sv_entropy,
        'sv_top5_ratio': sv_top5_ratio,
        'skew': skew,
        'kurtosis': kurt,
        'median': median_w,
        'iqr': iqr,
        'peakiness': peakiness,
    }


def compute_fingerprint(model, layer_names):
    """Compute fingerprint from selected layers."""
    layer_fps = []
    for name in layer_names:
        parts = name.split('.')
        layer = model
        for p in parts:
            layer = getattr(layer, p)
        fp = compute_layer_fingerprint(layer.weight.data)
        fp['layer'] = name
        layer_fps.append(fp)
    
    # Global aggregate
    global_fp = {
        'mean_mean': sum(l['mean'] for l in layer_fps) / len(layer_fps),
        'std_mean': sum(l['std'] for l in layer_fps) / len(layer_fps),
        'entropy_mean': sum(l['entropy'] for l in layer_fps) / len(layer_fps),
        'sparsity_mean': sum(l['sparsity'] for l in layer_fps) / len(layer_fps),
        'top_sv_mean': sum(l['top_sv'] for l in layer_fps) / len(layer_fps),
        'sv_entropy_mean': sum(l['sv_entropy'] for l in layer_fps) / len(layer_fps),
        'sv_top5_mean': sum(l.get('sv_top5_ratio', 0) for l in layer_fps) / len(layer_fps),
        'skew_mean': sum(l['skew'] for l in layer_fps) / len(layer_fps),
        'kurtosis_mean': sum(l['kurtosis'] for l in layer_fps) / len(layer_fps),
        'median_mean': sum(l.get('median', 0) for l in layer_fps) / len(layer_fps),
        'iqr_mean': sum(l.get('iqr', 0) for l in layer_fps) / len(layer_fps),
        'peakiness_mean': sum(l.get('peakiness', 0) for l in layer_fps) / len(layer_fps),
    }
    
    return {'global': global_fp, 'layers': layer_fps}


def euclidean_distance(f1, f2):
    g1 = f1['global']
    g2 = f2['global']
    keys = ['mean_mean', 'std_mean', 'entropy_mean', 'sparsity_mean', 
            'top_sv_mean', 'sv_entropy_mean', 'sv_top5_mean', 'skew_mean', 
            'kurtosis_mean', 'median_mean', 'iqr_mean', 'peakiness_mean']
    return math.sqrt(sum((g1[k] - g2[k])**2 for k in keys))


def knn_classify(train_fingerprints, test_fingerprint, k=3):
    from collections import Counter
    distances = [(euclidean_distance(fp, test_fingerprint), label) for label, fp in train_fingerprints]
    distances.sort()
    nearest = [label for _, label in distances[:k]]
    return Counter(nearest).most_common(1)[0][0]


def main():
    print("=" * 70)
    print("M145 / Track 8: Semantic Fingerprints v2 (Fast)")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Select key layers
    layer_names = [
        'model.language_model.layers.0.self_attn.q_proj',
        'model.language_model.layers.0.self_attn.k_proj',
        'model.language_model.layers.0.self_attn.v_proj',
        'model.language_model.layers.0.self_attn.o_proj',
        'model.language_model.layers.0.mlp.gate_proj',
        'model.language_model.layers.15.self_attn.q_proj',
        'model.language_model.layers.15.mlp.gate_proj',
        'model.language_model.layers.30.self_attn.q_proj',
    ]

    # 3. Create variants
    variants = {
        'base': ('base', {}),
        'noisy_small': ('noisy', {'magnitude': 0.0001}),
        'noisy_medium': ('noisy', {'magnitude': 0.001}),
        'noisy_large': ('noisy', {'magnitude': 0.01}),
        'quantized': ('quantized', {}),
        'sparse': ('sparse', {'threshold': 0.01}),
        'scaled_up': ('scaled', {'factor': 1.05}),
        'scaled_down': ('scaled', {'factor': 0.95}),
    }

    print(f"[2] Computing fingerprints for {len(variants)} variants...")
    fingerprints = {}
    base_state = {name: p.data.clone() for name, p in model.named_parameters()}
    
    for v_name, (v_type, v_kwargs) in variants.items():
        # Restore base
        for name, p in model.named_parameters():
            p.data.copy_(base_state[name])
        
        # Apply variant
        if v_type != 'base':
            variant = create_variant(model, v_type, **v_kwargs)
            apply_variant(model, variant)
        
        fp = compute_fingerprint(model, layer_names)
        fingerprints[v_name] = fp
        
        g = fp['global']
        print(f"  {v_name:15s}: Entropy={g['entropy_mean']:.2f}, "
              f"Sparsity={g['sparsity_mean']:.3f}, "
              f"TopSV={g['top_sv_mean']:.2f}, "
              f"Skew={g['skew_mean']:.2f}, "
              f"Kurt={g['kurtosis_mean']:.2f}")

    # 4. Distance matrix
    print(f"\n[3] Distance matrix:")
    v_names = list(variants.keys())
    print(f"  {'':15s}", end='')
    for v in v_names:
        print(f" {v[:8]:8s}", end='')
    print()
    
    for v1 in v_names:
        print(f"  {v1[:12]:12s}", end='')
        for v2 in v_names:
            d = euclidean_distance(fingerprints[v1], fingerprints[v2])
            marker = "*" if v1 == v2 else ""
            print(f" {d:7.3f}{marker}", end='')
        print()

    # 5. Classification (leave-one-out)
    print(f"\n[4] Leave-one-out k-NN classification (k=3):")
    correct = 0
    for test_name in v_names:
        train_data = [(n, fingerprints[n]) for n in v_names if n != test_name]
        pred = knn_classify(train_data, fingerprints[test_name], k=3)
        status = "✅" if pred == test_name else "❌"
        print(f"  {test_name:15s} → predicted: {pred:15s} {status}")
        if pred == test_name:
            correct += 1
    
    accuracy = correct / len(v_names) * 100
    print(f"\n  Accuracy: {correct}/{len(v_names)} = {accuracy:.1f}%")

    # 6. Pairwise separability
    print(f"\n[5] Pairwise separability:")
    separable = 0
    total_pairs = 0
    for i, v1 in enumerate(v_names):
        for v2 in v_names[i+1:]:
            d = euclidean_distance(fingerprints[v1], fingerprints[v2])
            total_pairs += 1
            if d > 1.0:
                separable += 1
            print(f"  {v1:15s} vs {v2:15s}: d={d:.4f}")
    
    print(f"\n  Separable pairs (>1.0): {separable}/{total_pairs}")

    # 7. Save
    output = {
        'variants': v_names,
        'num_layers': len(layer_names),
        'accuracy': accuracy,
        'separable_pairs': f"{separable}/{total_pairs}",
        'fingerprints': {k: v['global'] for k, v in fingerprints.items()},
    }
    
    out_path = 'experiments/m145_semantic_fingerprints_v2.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M145 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
