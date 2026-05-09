"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""
M146 / Track 9: Cross-Model Frozen Vocabulary (FAST VERSION)

Goal: Test if atom table built on one part of model works for another part.
Uses weight distribution comparison instead of full WAL encode.

Method:
  1. Load model
  2. Compare weight distributions: early layers (0-14) vs late layers (15-29)
  3. If distributions overlap → shared vocabulary possible
  4. If distributions diverge → separate vocabularies needed
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B"

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def get_layer_stats(model, start_layer, end_layer):
    """Get weight statistics for layers in range."""
    weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'embed' not in name and 'norm' not in name:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i+1 < len(parts):
                    try:
                        layer_num = int(parts[i+1])
                        if start_layer <= layer_num < end_layer:
                            weights.append(module.weight.data.float().cpu().flatten())
                    except:
                        pass
    
    if not weights:
        return None
    
    all_w = torch.cat(weights)
    # Sample for quantile (too large for full tensor)
    sample_size = min(100000, len(all_w))
    sample = all_w[torch.randperm(len(all_w))[:sample_size]]
    return {
        'mean': all_w.mean().item(),
        'std': all_w.std().item(),
        'min': all_w.min().item(),
        'max': all_w.max().item(),
        'median': sample.median().item(),
        'q25': sample.quantile(0.25).item(),
        'q75': sample.quantile(0.75).item(),
        'num_weights': len(all_w),
        'sparsity': (all_w.abs() < 0.01).float().mean().item(),
    }


def distribution_distance(s1, s2):
    """Distance between two distributions."""
    # Normalized earth-mover-like distance
    mean_diff = abs(s1['mean'] - s2['mean']) / (abs(s1['mean']) + abs(s2['mean']) + 1e-8)
    std_diff = abs(s1['std'] - s2['std']) / (s1['std'] + s2['std'] + 1e-8)
    range_diff = abs((s1['max'] - s1['min']) - (s2['max'] - s2['min'])) / (abs(s1['max'] - s1['min']) + abs(s2['max'] - s2['min']) + 1e-8)
    return (mean_diff + std_diff + range_diff) / 3


def main():
    print("=" * 70)
    print("M146 / Track 9: Cross-Model Frozen Vocabulary (Fast)")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Compare layer ranges
    ranges = [
        (0, 10, "Early (0-9)"),
        (10, 20, "Mid (10-19)"),
        (20, 32, "Late (20-31)"),
    ]
    
    print("[2] Computing weight distribution stats...")
    stats = {}
    for start, end, label in ranges:
        s = get_layer_stats(model, start, end)
        stats[label] = s
        print(f"  {label:20s}: mean={s['mean']:.6f}, std={s['std']:.6f}, "
              f"range=[{s['min']:.4f}, {s['max']:.4f}], "
              f"median={s['median']:.6f}, sparsity={s['sparsity']:.3f}, "
              f"weights={s['num_weights']}")

    # 3. Pairwise distances
    print("\n[3] Distribution distances:")
    labels = list(stats.keys())
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            d = distribution_distance(stats[l1], stats[l2])
            print(f"  {l1} vs {l2}: d={d:.4f}")

    # 4. Quantization overlap test
    print("\n[4] Quantization cell overlap test:")
    
    # Build bins from early layers, test on late
    early_w = []
    late_w = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'embed' not in name and 'norm' not in name:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i+1 < len(parts):
                    try:
                        layer_num = int(parts[i+1])
                        w = module.weight.data.float().cpu().flatten()
                        if layer_num < 16:
                            early_w.append(w)
                        else:
                            late_w.append(w)
                    except:
                        pass
    
    early_all = torch.cat(early_w)
    late_all = torch.cat(late_w)
    
    # Build 256 bins from early
    e_min, e_max = early_all.min().item(), early_all.max().item()
    bins = torch.linspace(e_min, e_max, 257)
    
    # Count how many late weights fall in early bins
    late_hist = torch.histc(late_all, bins=256, min=e_min, max=e_max)
    early_hist = torch.histc(early_all, bins=256, min=e_min, max=e_max)
    
    # KL divergence between histograms
    e_probs = early_hist / early_hist.sum()
    l_probs = late_hist / late_hist.sum()
    kl_el = (e_probs * ((e_probs + 1e-10) / (l_probs + 1e-10)).log()).sum().item()
    kl_le = (l_probs * ((l_probs + 1e-10) / (e_probs + 1e-10)).log()).sum().item()
    
    overlap = (e_probs * l_probs).sum().item()
    
    print(f"  Early histogram entropy: {-(e_probs * (e_probs + 1e-10).log()).sum().item():.3f}")
    print(f"  Late histogram entropy:  {-(l_probs * (l_probs + 1e-10).log()).sum().item():.3f}")
    print(f"  KL(early||late): {kl_el:.4f}")
    print(f"  KL(late||early): {kl_le:.4f}")
    print(f"  Histogram overlap: {overlap:.4f}")
    
    if overlap > 0.8:
        print(f"  ✅ High overlap → shared vocabulary likely works")
    elif overlap > 0.5:
        print(f"  ⚠️  Moderate overlap → shared vocabulary with some penalty")
    else:
        print(f"  ❌ Low overlap → separate vocabularies needed")

    # 5. Per-layer-type comparison
    print("\n[5] Per-layer-type stats (layer 15 = middle):")
    layer_types = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    
    for lt in layer_types:
        early_name = f"model.layers.0.{lt}"
        late_name = f"model.layers.30.{lt}"
        
        try:
            parts = early_name.split('.')
            layer = model
            for p in parts:
                layer = getattr(layer, p)
            e_w = layer.weight.data.float().cpu().flatten()
            
            parts = late_name.split('.')
            layer = model
            for p in parts:
                layer = getattr(layer, p)
            l_w = layer.weight.data.float().cpu().flatten()
            
            d = distribution_distance(
                {'mean': e_w.mean().item(), 'std': e_w.std().item(), 'min': e_w.min().item(), 'max': e_w.max().item()},
                {'mean': l_w.mean().item(), 'std': l_w.std().item(), 'min': l_w.min().item(), 'max': l_w.max().item()},
            )
            print(f"  {lt:25s}: distance={d:.4f}")
        except:
            pass

    # 6. Save
    output = {
        'ranges': {k: v for k, v in stats.items()},
        'kl_early_late': kl_el,
        'kl_late_early': kl_le,
        'histogram_overlap': overlap,
        'shared_vocab_recommended': overlap > 0.5,
    }
    
    out_path = 'experiments/m146_cross_model_vocab.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M146 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
