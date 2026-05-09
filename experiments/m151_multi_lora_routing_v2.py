#!/usr/bin/env python3
"""M151 v2 — Multi-LoRA Routing / Conflict Test (synthetic).

Creates multiple synthetic LoRA overlays, tests combinations and interference.
"""
import torch
import torch.nn as nn
import json


def create_lora_delta(out_d, in_d, rank, seed):
    """Create a synthetic LoRA delta with fixed seed for reproducibility."""
    torch.manual_seed(seed)
    A = torch.randn(out_d, rank) * 0.01
    B = torch.randn(rank, in_d) * 0.01
    return A @ B


def apply_overlays(base_weight, overlays, active):
    """Apply active overlays to base weight."""
    result = base_weight.clone()
    for i, delta in enumerate(overlays):
        if active[i]:
            result = result + delta
    return result


def measure_interference(base, overlays, active_mask, name):
    """Measure interference for a given overlay combination."""
    combined = apply_overlays(base, overlays, active_mask)
    
    # Individual effects
    individual = []
    for i, delta in enumerate(overlays):
        if active_mask[i]:
            w = base + delta
            individual.append(w)
    
    # Interference = how much combined deviates from sum of individuals
    if len(individual) == 1:
        interference = 0.0
    else:
        sum_individual = sum(individual) - (len(individual) - 1) * base
        interference = ((combined - sum_individual).abs() / (base.abs() + 1e-8)).mean().item()
    
    return {
        'name': name,
        'active': active_mask,
        'interference': interference,
        'delta_norm': torch.linalg.norm(combined - base, 'fro').item(),
    }


def main():
    print("=" * 60)
    print("M151 v2 — Multi-LoRA Routing / Conflict Test")
    print("=" * 60)
    
    # Representative shape
    out_d, in_d = 4096, 4096
    base = torch.randn(out_d, in_d) * 0.01
    
    # Create 4 overlays with different "personalities"
    overlays = [
        create_lora_delta(out_d, in_d, rank=4, seed=1),   # Edit A: factual
        create_lora_delta(out_d, in_d, rank=4, seed=2),   # Edit B: style
        create_lora_delta(out_d, in_d, rank=4, seed=3),   # Edit C: safety
        create_lora_delta(out_d, in_d, rank=4, seed=4),   # Edit D: domain
    ]
    
    # Test combinations
    tests = [
        ([1, 0, 0, 0], "A only"),
        ([0, 1, 0, 0], "B only"),
        ([0, 0, 1, 0], "C only"),
        ([0, 0, 0, 1], "D only"),
        ([1, 1, 0, 0], "A+B"),
        ([1, 0, 1, 0], "A+C"),
        ([1, 0, 0, 1], "A+D"),
        ([0, 1, 1, 0], "B+C"),
        ([1, 1, 1, 0], "A+B+C"),
        ([1, 1, 1, 1], "A+B+C+D"),
        ([1, 0, 1, 1], "A+C+D"),
    ]
    
    results = []
    
    print(f"\nBase weight shape: {out_d}×{in_d}")
    print(f"Overlays: {len(overlays)} (rank=4 each)")
    print(f"\n{'Combination':<15} {'Interference':>14} {'Delta Norm':>12}")
    print("-" * 45)
    
    for active_mask, name in tests:
        r = measure_interference(base, overlays, active_mask, name)
        results.append(r)
        active_str = "".join(['A' if active_mask[0] else '-',
                               'B' if active_mask[1] else '-',
                               'C' if active_mask[2] else '-',
                               'D' if active_mask[3] else '-'])
        print(f"{active_str:<15} {r['interference']:>14.6f} {r['delta_norm']:>12.4f}")
    
    # Summary
    single = [r for r in results if sum(r['active']) == 1]
    multi = [r for r in results if sum(r['active']) > 1]
    
    avg_single = sum(r['delta_norm'] for r in single) / len(single)
    avg_multi = sum(r['delta_norm'] for r in multi) / len(multi)
    avg_interference = sum(r['interference'] for r in multi) / len(multi)
    
    print(f"\nSummary:")
    print(f"  Avg single-edit delta norm: {avg_single:.4f}")
    print(f"  Avg multi-edit delta norm:  {avg_multi:.4f}")
    print(f"  Avg interference:           {avg_interference:.6f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m151_multi_lora_routing.json', 'w') as f:
        json.dump({
            'avg_single_delta': avg_single,
            'avg_multi_delta': avg_multi,
            'avg_interference': avg_interference,
            'tests': results,
        }, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
