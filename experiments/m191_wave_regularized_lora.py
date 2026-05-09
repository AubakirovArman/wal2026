#!/usr/bin/env python3
"""M191 — Wave-Regularized LoRA (synthetic test).

Tests whether spectral concentration penalty produces "healthier" LoRA deltas.
"""
import torch, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM


def compute_spectral_features(delta):
    """Compute spectral concentration of a delta matrix."""
    d = delta.float()
    m, n = d.shape
    
    # DCT spectrum
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np_ = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    d_pad = torch.zeros(mp, np_, dtype=torch.float32, device=d.device)
    d_pad[:m, :n] = d
    dct = torch.fft.fft2(d_pad).abs()
    dct_flat = dct.reshape(-1)
    
    total = dct_flat.sum().item()
    top10 = dct_flat.sort(descending=True).values[:max(1, len(dct_flat)//10)].sum().item() / total
    top1 = dct_flat.sort(descending=True).values[:max(1, len(dct_flat)//100)].sum().item() / total
    
    # Spectral norm
    try:
        spec_norm = torch.linalg.svdvals(d.float())[0].item()
    except Exception:
        spec_norm = 0.0
    
    return {'top1': top1, 'top10': top10, 'spec_norm': spec_norm}


def apply_wave_regularization(delta, lambda_reg=0.1, steps=10):
    """Apply spectral concentration penalty to delta."""
    d = delta.clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([d], lr=0.01)
    
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Target: keep delta close to original
        fidelity_loss = (d - delta.float()).pow(2).mean()
        
        # Wave penalty: penalize spectral concentration
        mp = 1 << max(0, math.ceil(math.log2(d.shape[0]))) if d.shape[0] > 1 else 1
        np_ = 1 << max(0, math.ceil(math.log2(d.shape[1]))) if d.shape[1] > 1 else 1
        d_pad = torch.zeros(mp, np_, dtype=torch.float32, device=d.device)
        d_pad[:d.shape[0], :d.shape[1]] = d
        dct = torch.fft.fft2(d_pad).abs()
        dct_flat = dct.reshape(-1)
        
        # Penalize top-10% energy concentration
        sorted_dct = dct_flat.sort(descending=True).values
        top10_energy = sorted_dct[:max(1, len(sorted_dct)//10)].sum()
        total_energy = dct_flat.sum()
        concentration = top10_energy / (total_energy + 1e-10)
        
        loss = fidelity_loss + lambda_reg * concentration
        loss.backward()
        optimizer.step()
    
    return d.detach().to(delta.dtype)


def main():
    print("=" * 60)
    print("M191 — Wave-Regularized LoRA")
    print("=" * 60)
    
    device = "cuda:0"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    layer_idx = 0
    modules = ['q_proj', 'v_proj', 'gate_proj']
    rank = 4
    scale = 1.0
    
    results = []
    
    for m_name in modules:
        print(f"\n--- {m_name} ---")
        mod = getattr(model.model.layers[layer_idx].self_attn if m_name != 'gate_proj' else model.model.layers[layer_idx].mlp, m_name)
        w = mod.weight.data
        
        # Generate base LoRA delta
        torch.manual_seed(42)
        A = torch.randn(w.shape[0], rank, dtype=w.dtype, device=w.device) * scale
        B = torch.randn(rank, w.shape[1], dtype=w.dtype, device=w.device) * scale
        delta_base = A @ B
        
        # Apply wave regularization
        delta_reg = apply_wave_regularization(delta_base, lambda_reg=0.5, steps=20)
        
        # Compare features
        feat_base = compute_spectral_features(delta_base)
        feat_reg = compute_spectral_features(delta_reg)
        
        # Compute PPL impact (synthetic: norm of delta relative to weight)
        impact_base = (delta_base.float().abs().mean() / w.float().abs().mean()).item()
        impact_reg = (delta_reg.float().abs().mean() / w.float().abs().mean()).item()
        
        print(f"  Base: top1={feat_base['top1']:.3f}, top10={feat_base['top10']:.3f}, spec_norm={feat_base['spec_norm']:.2f}, impact={impact_base:.4f}")
        print(f"  Reg:  top1={feat_reg['top1']:.3f}, top10={feat_reg['top10']:.3f}, spec_norm={feat_reg['spec_norm']:.2f}, impact={impact_reg:.4f}")
        print(f"  Δtop1: {feat_reg['top1'] - feat_base['top1']:+.3f}, Δtop10: {feat_reg['top10'] - feat_base['top10']:+.3f}")
        
        results.append({
            'module': m_name,
            'base': feat_base,
            'regularized': feat_reg,
            'impact_base': impact_base,
            'impact_reg': impact_reg,
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Module':>10} {'Δtop1':>8} {'Δtop10':>8} {'Δimpact':>8}")
    print("-" * 40)
    for r in results:
        dt1 = r['regularized']['top1'] - r['base']['top1']
        dt10 = r['regularized']['top10'] - r['base']['top10']
        di = r['impact_reg'] - r['impact_base']
        print(f"{r['module']:>10} {dt1:>+8.3f} {dt10:>+8.3f} {di:>+8.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m191_wave_regularized_lora.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
