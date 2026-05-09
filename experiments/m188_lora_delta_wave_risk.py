#!/usr/bin/env python3
"""M188 — LoRA Delta Wave Risk.

Analyzes wave structure of LoRA deltas to predict edit risk.
"""
import torch, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM


def hadamard_matrix(n):
    if n == 1: return torch.ones(1, 1)
    H = hadamard_matrix(n // 2)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)


def apply_hadamard(W):
    m, n = W.shape
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    W_pad = torch.zeros(mp, np, dtype=torch.float32, device=W.device)
    W_pad[:m, :n] = W.float()
    H_out = (hadamard_matrix(mp).to(W.device) / math.sqrt(mp))
    H_in = (hadamard_matrix(np).to(W.device) / math.sqrt(np))
    return H_out @ W_pad @ H_in.T


def compute_wave_features(delta):
    """Compute wave features for a LoRA delta matrix."""
    d = delta.float()
    m, n = d.shape
    
    # 1. DCT spectrum (via FFT approximation)
    mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
    np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
    d_pad = torch.zeros(mp, np, dtype=torch.float32, device=d.device)
    d_pad[:m, :n] = d
    dct = torch.fft.fft2(d_pad).abs()
    dct_flat = dct.reshape(-1)
    dct_sorted = dct_flat.sort(descending=True).values
    
    # Top energy concentrations
    total_energy = dct_flat.sum().item()
    top1_pct = dct_sorted[:max(1, len(dct_sorted)//100)].sum().item() / total_energy
    top10_pct = dct_sorted[:max(1, len(dct_sorted)//10)].sum().item() / total_energy
    
    # Spectral entropy
    probs = dct_flat / dct_flat.sum()
    spectral_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    # 2. Hadamard spectrum
    if m <= 16384 and n <= 16384:
        try:
            had = apply_hadamard(d)
            had_flat = had.abs().reshape(-1)
            had_sorted = had_flat.sort(descending=True).values
            had_top10 = had_sorted[:max(1, len(had_sorted)//10)].sum().item() / had_flat.sum().item()
        except Exception:
            had_top10 = 0.0
    else:
        had_top10 = 0.0
    
    # 3. Phase coherence
    phase = torch.fft.fft2(d_pad).angle()
    phase_diff = (phase[1:, :] - phase[:-1, :]).abs().mean().item()
    
    # 4. Singular values
    try:
        sv = torch.linalg.svdvals(d.float())
        sv_sum = sv.sum().item()
        sv_top1 = (sv[0] / sv_sum).item() if sv_sum > 0 else 0
        sv_top10 = (sv[:max(1, len(sv)//10)].sum() / sv_sum).item() if sv_sum > 0 else 0
        condition_number = (sv[0] / sv[-1]).item() if sv[-1] > 0 else float('inf')
    except Exception:
        sv_top1 = sv_top10 = condition_number = 0.0
    
    # 5. Spectral norm (largest singular value)
    spectral_norm = sv[0].item() if 'sv' in dir() else 0.0
    
    # 6. Fingerprint: distribution stats
    hist = torch.histc(d, bins=32, min=d.min().item(), max=d.max().item())
    hist_probs = hist / hist.sum()
    fingerprint_entropy = -(hist_probs * torch.log(hist_probs + 1e-10)).sum().item()
    
    return {
        'top1_energy': top1_pct,
        'top10_energy': top10_pct,
        'spectral_entropy': spectral_entropy,
        'hadamard_top10': had_top10,
        'phase_coherence': phase_diff,
        'sv_top1': sv_top1,
        'sv_top10': sv_top10,
        'condition_number': condition_number,
        'spectral_norm': spectral_norm,
        'fingerprint_entropy': fingerprint_entropy,
        'delta_norm': d.norm().item(),
        'delta_mean': d.mean().item(),
        'delta_std': d.std().item(),
    }


def compute_wave_risk(features):
    """Compute WaveRiskScore from features."""
    # Higher concentration = higher risk
    risk = 0.0
    risk += features['top1_energy'] * 2.0      # top-1% energy
    risk += features['top10_energy'] * 1.0      # top-10% energy
    risk += features['sv_top1'] * 2.0            # top singular value dominance
    risk += features['spectral_norm'] * 0.1     # raw spectral norm
    risk -= features['spectral_entropy'] * 0.2  # higher entropy = lower risk
    risk -= features['fingerprint_entropy'] * 0.1
    return max(0, risk)


def apply_lora_edit(w, rank=4, scale=0.1, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(w.shape[0], rank, dtype=w.dtype, device=w.device) * scale
    B = torch.randn(rank, w.shape[1], dtype=w.dtype, device=w.device) * scale
    return A @ B


def main():
    print("=" * 60)
    print("M188 — LoRA Delta Wave Risk")
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
    configs = [
        ("rank1_scale0.1", 1, 0.1),
        ("rank1_scale1.0", 1, 1.0),
        ("rank4_scale0.1", 4, 0.1),
        ("rank4_scale1.0", 4, 1.0),
        ("rank8_scale0.1", 8, 0.1),
        ("rank8_scale1.0", 8, 1.0),
    ]
    
    results = []
    
    for m_name in modules:
        print(f"\n--- {m_name} ---")
        mod = getattr(model.model.layers[layer_idx].self_attn if m_name != 'gate_proj' else model.model.layers[layer_idx].mlp, m_name)
        w = mod.weight.data
        
        for label, rank, scale in configs:
            delta = apply_lora_edit(w, rank=rank, scale=scale)
            features = compute_wave_features(delta)
            risk = compute_wave_risk(features)
            
            features['risk'] = risk
            features['module'] = m_name
            features['config'] = label
            features['rank'] = rank
            features['scale'] = scale
            
            results.append(features)
            
            print(f"  {label:20s} risk={risk:6.2f}  top1={features['top1_energy']:.3f}  top10={features['top10_energy']:.3f}  spec_norm={features['spectral_norm']:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary: Risk ranking")
    print(f"{'='*60}")
    
    sorted_results = sorted(results, key=lambda x: x['risk'], reverse=True)
    print(f"{'Config':>20} {'Module':>10} {'Risk':>8} {'Top1%':>8} {'Top10%':>8} {'SpecNorm':>10}")
    print("-" * 70)
    for r in sorted_results:
        print(f"{r['config']:>20} {r['module']:>10} {r['risk']:>8.2f} {r['top1_energy']:>8.3f} {r['top10_energy']:>8.3f} {r['spectral_norm']:>10.4f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m188_lora_delta_wave_risk.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
