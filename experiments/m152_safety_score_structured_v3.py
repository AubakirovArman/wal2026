"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M152 v3 — Safety Score on Structured LoRA Deltas.

Generates synthetic low-rank deltas (A @ B) with varying rank and scale,
then validates Safety Score monotonicity and classification.
"""
import torch
import json


def safety_score(delta_W):
    spectral = torch.linalg.matrix_norm(delta_W, ord=2).item()
    if spectral < 1.0:     return "SAFE", spectral
    elif spectral < 5.0:   return "MODERATE", spectral
    elif spectral < 10.0:  return "RISKY", spectral
    else:                  return "DANGEROUS", spectral


def generate_lora_delta(out_d, in_d, rank, scale=1.0):
    """Generate synthetic LoRA delta = A @ B with controlled scale."""
    A = torch.randn(out_d, rank) * 0.01
    B = torch.randn(rank, in_d) * 0.01
    delta = A @ B
    # Normalize to target spectral norm
    current_spectral = torch.linalg.matrix_norm(delta, ord=2).item()
    if current_spectral > 0:
        delta = delta * (scale / current_spectral)
    return delta


def main():
    print("=" * 60)
    print("M152 v3 — Safety Score on Structured LoRA Deltas")
    print("=" * 60)
    
    # Representative dimensions from Llama-3.1-8B
    shapes = [
        (4096, 4096),   # q_proj/k_proj/v_proj/o_proj
        (14336, 4096),  # gate_proj/up_proj
        (4096, 14336),  # down_proj
    ]
    
    configs = [
        {'rank': 1, 'scale': 0.5},
        {'rank': 1, 'scale': 2.0},
        {'rank': 4, 'scale': 0.5},
        {'rank': 4, 'scale': 2.0},
        {'rank': 4, 'scale': 8.0},
        {'rank': 8, 'scale': 0.5},
        {'rank': 8, 'scale': 2.0},
        {'rank': 8, 'scale': 12.0},
    ]
    
    results = []
    
    for shape in shapes:
        out_d, in_d = shape
        print(f"\nShape {out_d}×{in_d}")
        
        for cfg in configs:
            rank, scale = cfg['rank'], cfg['scale']
            
            # Generate multiple samples
            spectra = []
            scores = []
            for _ in range(10):
                delta = generate_lora_delta(out_d, in_d, rank, scale)
                score, spectral = safety_score(delta)
                spectra.append(spectral)
                scores.append(score)
            
            avg_spectral = sum(spectra) / len(spectra)
            max_spectral = max(spectra)
            min_spectral = min(spectra)
            
            # Count score distribution
            score_counts = {}
            for s in scores:
                score_counts[s] = score_counts.get(s, 0) + 1
            
            print(f"  rank={rank:2d} scale={scale:5.1f}: avg_spectral={avg_spectral:7.3f}, "
                  f"scores={score_counts}")
            
            results.append({
                'shape': f"{out_d}x{in_d}",
                'rank': rank,
                'scale': scale,
                'avg_spectral': avg_spectral,
                'max_spectral': max_spectral,
                'min_spectral': min_spectral,
                'score_counts': score_counts,
            })
    
    # Validate monotonicity: higher scale → higher score category
    print("\n" + "=" * 60)
    print("VALIDATION: Monotonicity")
    print("=" * 60)
    
    monotonic_violations = 0
    for shape in shapes:
        shape_str = f"{shape[0]}x{shape[1]}"
        for rank in [1, 4, 8]:
            entries = [r for r in results if r['shape'] == shape_str and r['rank'] == rank]
            entries.sort(key=lambda x: x['scale'])
            avg_specs = [e['avg_spectral'] for e in entries]
            for i in range(len(avg_specs) - 1):
                if avg_specs[i] > avg_specs[i+1] * 1.1:  # allow 10% noise
                    monotonic_violations += 1
                    print(f"  VIOLATION: {shape_str} rank={rank}: scale={entries[i]['scale']} > scale={entries[i+1]['scale']} (spectral {avg_specs[i]:.3f} > {avg_specs[i+1]:.3f})")
    
    if monotonic_violations == 0:
        print("  ✅ All scale→spectral relationships are monotonic")
    else:
        print(f"  ⚠️ {monotonic_violations} non-monotonic relationships found")
    
    # Save
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m152_safety_score_real_lora.json', 'w') as f:
        json.dump({
            'monotonic_violations': monotonic_violations,
            'results': results,
        }, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
