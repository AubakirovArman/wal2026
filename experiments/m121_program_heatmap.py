#!/usr/bin/env python3
"""M121 / Phase 21: Program Heatmap Analysis

Analyze which atoms dominate per layer, find patterns in program usage,
and compute layer-type statistics (attention vs MLP).
"""
import torch
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import WALCachedLinear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"


def main():
    print("=" * 70)
    print("M121 / Phase 21: Program Heatmap Analysis")
    print("=" * 70)

    print("\n[1] Loading and encoding model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)

    # Collect all WAL layers
    wal_layers = []
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            wal_layers.append((name, module))

    print(f"    Total WAL layers: {len(wal_layers)}")

    # Per-layer atom histograms
    print("\n[2] Computing atom usage statistics...", flush=True)
    
    layer_stats = []
    for name, module in wal_layers:
        atom_ids = module.wal_weight.prog.atom_ids.cpu()
        
        # Top atoms
        counts = torch.bincount(atom_ids.long(), minlength=256)
        top5 = counts.argsort(descending=True)[:5]
        top5_pct = counts[top5].float() / atom_ids.numel() * 100
        
        # Entropy of atom distribution
        probs = counts.float() / counts.sum()
        entropy = -(probs[probs > 0] * torch.log2(probs[probs > 0])).sum().item()
        max_entropy = torch.log2(torch.tensor(256.0)).item()
        normalized_entropy = entropy / max_entropy
        
        layer_stats.append({
            'name': name,
            'top_atom': top5[0].item(),
            'top_pct': top5_pct[0].item(),
            'entropy': normalized_entropy,
            'dominance': top5_pct[:3].sum().item(),  # % accounted by top 3 atoms
        })

    # Sort by entropy
    layer_stats.sort(key=lambda x: x['entropy'])

    print(f"\n  {'Layer':<50} {'TopAtom':>7} {'Top%':>6} {'Entropy':>8} {'Dom%':>6}")
    print(f"  {'-'*50} {'-'*7} {'-'*6} {'-'*8} {'-'*6}")
    for s in layer_stats[:15]:
        print(f"  {s['name']:<50} {s['top_atom']:>7} {s['top_pct']:>5.1f}% {s['entropy']:>8.3f} {s['dominance']:>5.1f}%")
    print("  ...")
    for s in layer_stats[-5:]:
        print(f"  {s['name']:<50} {s['top_atom']:>7} {s['top_pct']:>5.1f}% {s['entropy']:>8.3f} {s['dominance']:>5.1f}%")

    # Layer-type aggregation
    print("\n[3] Layer-type aggregation...", flush=True)
    attn_layers = [s for s in layer_stats if 'self_attn' in s['name']]
    mlp_layers = [s for s in layer_stats if 'mlp' in s['name']]
    other_layers = [s for s in layer_stats if s not in attn_layers and s not in mlp_layers]

    def avg(key, lst):
        return sum(s[key] for s in lst) / len(lst) if lst else 0

    print(f"\n  Type          Count  AvgTop%  AvgEntropy  AvgDom%")
    print(f"  {'-'*13} {'-'*6} {'-'*8} {'-'*11} {'-'*8}")
    print(f"  Attention    {len(attn_layers):>5}  {avg('top_pct', attn_layers):>7.1f}% {avg('entropy', attn_layers):>10.3f} {avg('dominance', attn_layers):>7.1f}%")
    print(f"  MLP          {len(mlp_layers):>5}  {avg('top_pct', mlp_layers):>7.1f}% {avg('entropy', mlp_layers):>10.3f} {avg('dominance', mlp_layers):>7.1f}%")
    print(f"  Other        {len(other_layers):>5}  {avg('top_pct', other_layers):>7.1f}% {avg('entropy', other_layers):>10.3f} {avg('dominance', other_layers):>7.1f}%")

    # Find most "specialized" layers (low entropy = few atoms dominate)
    print("\n[4] Most specialized layers (lowest entropy)...")
    for s in layer_stats[:5]:
        print(f"    {s['name']:<50} entropy={s['entropy']:.3f}, top3={s['dominance']:.1f}%")

    print("\n[5] Most diverse layers (highest entropy)...")
    for s in layer_stats[-5:]:
        print(f"    {s['name']:<50} entropy={s['entropy']:.3f}, top3={s['dominance']:.1f}%")

    print("\n" + "=" * 70)
    print("M121 / Phase 21: SUMMARY")
    print("=" * 70)
    print(f"\n  Total WAL layers analyzed: {len(wal_layers)}")
    print(f"  Average normalized entropy: {avg('entropy', layer_stats):.3f}")
    print(f"  Average top-3 dominance: {avg('dominance', layer_stats):.1f}%")
    print(f"\n  Key insight: {'Low' if avg('entropy', layer_stats) < 0.5 else 'Moderate' if avg('entropy', layer_stats) < 0.8 else 'High'} entropy overall.")
    print(f"  {'Few' if avg('dominance', layer_stats) > 30 else 'Many'} atoms are reused across weights.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
