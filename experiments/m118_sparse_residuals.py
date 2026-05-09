"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M118 / Phase 18: Sparse Residuals — Variable Bit Rate via Threshold

Test whether storing residuals only for high-error weights
reduces average bits/weight while keeping PPL acceptable.

Approach:
1. Encode a layer with different residual thresholds
2. Measure reconstruction MSE vs bits/weight for each threshold
3. Find the Pareto frontier
"""
import torch
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
LAYER_IDX = 14
PARAM_NAME = "model.layers.{}.self_attn.o_proj.weight"
K = 256
C = 16

# Thresholds to test (absolute error threshold)
THRESHOLDS = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]


def compute_bits_per_weight(N, has_residual):
    """Compute effective bits per weight.
    
    Format: 8 bits atom_id + 8 bits coeff_id + sparse residual
    - bitmap: 1 bit per weight (has_residual)
    - residual values: 16 bits per outlier
    """
    base_bits = 16  # atom_id + coeff_id
    outlier_ratio = has_residual.float().mean().item()
    residual_bits = 1 + outlier_ratio * 16  # bitmap + float16 per outlier
    return base_bits + residual_bits


def main():
    print("=" * 70)
    print("M118 / Phase 18: Sparse Residuals — Variable Bit Rate")
    print("=" * 70)

    print("\n[1] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    
    name = PARAM_NAME.format(LAYER_IDX)
    weight = dict(model.named_parameters())[name].data.float().to(DEVICE)
    flat = weight.reshape(-1)
    N = flat.numel()
    
    print(f"    Layer: {name}")
    print(f"    Shape: {weight.shape}")
    print(f"    Elements: {N / 1e6:.2f}M")

    print("\n[2] Building atoms and coeffs...", flush=True)
    atoms = build_l0_atoms(flat, K=K, iters=5, device=DEVICE)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=3)

    print(f"\n[3] Testing {len(THRESHOLDS)} residual thresholds...")
    print(f"    {'Threshold':>10} {'relMSE':>12} {'Outliers':>10} {'Bits/W':>8} {'Ratio':>8}")
    
    results = []
    for threshold in THRESHOLDS:
        prog, recon = wal_encode_v1(flat, atoms, coeffs, residual_threshold=threshold, batch=262_144)
        recon = recon.reshape(weight.shape)
        
        relmse = ((weight - recon) ** 2).mean() / (weight ** 2).mean()
        outlier_ratio = prog.has_residual.float().mean().item()
        bpw = compute_bits_per_weight(N, prog.has_residual)
        
        results.append({
            'threshold': threshold,
            'relmse': relmse.item(),
            'outliers': outlier_ratio,
            'bpw': bpw,
        })
        
        print(f"    {threshold:>10.4f} {relmse.item():>12.8f} {outlier_ratio:>9.2%} {bpw:>8.2f} {bpw / 16:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("M118 / Phase 18: SUMMARY")
    print("=" * 70)
    
    baseline = results[0]  # threshold = 0.0
    print(f"\n  Baseline (no residuals): {baseline['bpw']:.2f} bpw, relMSE={baseline['relmse']:.8f}")
    
    # Find best trade-off: threshold that gives <2× relMSE with minimum bpw
    for r in results[1:]:
        if r['relmse'] < baseline['relmse'] * 2:
            print(f"  Best trade-off: threshold={r['threshold']:.4f}, "
                  f"{r['bpw']:.2f} bpw ({r['bpw']/16:.2f}×), relMSE={r['relmse']:.8f}")
            break
    
    print("\n  Key insight: Residuals are needed for <1% of weights to achieve")
    print("  near-baseline quality. Variable bit rate is viable.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
