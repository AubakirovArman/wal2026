#!/usr/bin/env python3
"""M189 — Phase Coherence Test.

Tests whether wave structure (M186) is an amplitude phenomenon.
If wave features are determined by amplitude spectrum, shuffling
phases should preserve spectral energy profile.
"""
import torch, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM


def compute_spectral_features(w):
    """Compute wave features for a weight matrix."""
    w_flat = w.reshape(-1).float()
    
    # FFT features
    fft = torch.fft.fft(w_flat)
    amps = fft.abs()
    
    # Top energy ratios
    sorted_amps = amps.sort(descending=True).values
    top1 = sorted_amps[0] / sorted_amps.sum()
    top10 = sorted_amps[:10].sum() / sorted_amps.sum()
    
    # Spectral norm (need float32 for linalg)
    if w.dim() >= 2:
        spec_norm = torch.linalg.matrix_norm(w.float(), ord=2).item()
    else:
        spec_norm = w.float().abs().max().item()
    
    # Spectral entropy
    probs = amps / amps.sum()
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    
    return {
        'top1_energy': top1.item(),
        'top10_energy': top10.item(),
        'spectral_norm': spec_norm,
        'spectral_entropy': entropy.item(),
    }


def phase_shuffle(w):
    """Shuffle phases while preserving amplitude spectrum. Vectorized."""
    w_flat = w.reshape(-1).float()
    fft = torch.fft.fft(w_flat)
    amps = fft.abs()
    n = len(amps)
    
    # Random phases
    phases = torch.rand(n, device=w.device, dtype=torch.float32) * 2 * math.pi
    phases[0] = fft.angle()[0]  # Keep DC phase
    if n % 2 == 0:
        phases[n // 2] = 0  # Nyquist real
    
    # Enforce Hermitian symmetry: phases[n-i] = -phases[i]
    if n > 2:
        idx = torch.arange(1, (n + 1) // 2, device=w.device)
        phases.index_copy_(0, n - idx, -phases[idx])
    
    new_fft = amps * torch.exp(1j * phases)
    shuffled = torch.fft.ifft(new_fft).real
    
    # Match original shape and dtype
    return shuffled.reshape(w.shape).to(w.dtype)


def main():
    print("=" * 60)
    print("M189 — Phase Coherence Test")
    print("=" * 60)
    
    device = "cuda:0"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    n_layers = len(model.model.layers)
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    results = {}
    
    # Test on a representative subset: layer 0, 16, 31
    test_layers = [0, 16, 31]
    
    for li in test_layers:
        layer = model.model.layers[li]
        print(f"\n--- Layer {li} ---")
        
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            w = mod.weight.data
            
            # Original features
            orig = compute_spectral_features(w)
            
            # Phase-shuffled features
            w_shuf = phase_shuffle(w)
            shuf = compute_spectral_features(w_shuf)
            
            # Compute relative differences
            diff = {k: abs(shuf[k] - orig[k]) / (abs(orig[k]) + 1e-10) * 100 for k in orig}
            
            key = f"L{li}_{name}"
            results[key] = {
                'original': orig,
                'shuffled': shuf,
                'diff_pct': diff,
            }
            
            print(f"  {name:12s}: top1={diff['top1_energy']:5.1f}% top10={diff['top10_energy']:5.1f}% "
                  f"sn={diff['spectral_norm']:5.1f}% ent={diff['spectral_entropy']:5.1f}%")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary: Average % change after phase shuffle")
    print(f"{'='*60}")
    
    avg_diff = {k: sum(r['diff_pct'][k] for r in results.values()) / len(results) 
                for k in ['top1_energy', 'top10_energy', 'spectral_norm', 'spectral_entropy']}
    
    for k, v in avg_diff.items():
        print(f"  {k:20s}: {v:6.2f}%")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m189_phase_coherence.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
