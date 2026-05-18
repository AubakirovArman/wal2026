#!/usr/bin/env python3
"""M186 — Wave Depth Map.

Analyzes wave patterns across model depth.
"""
import torch, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def compute_layer_features(model, layer_idx):
    """Compute wave features for all modules in a layer."""
    layer = model.model.layers[layer_idx]
    modules = [
        ('q_proj', layer.self_attn.q_proj),
        ('k_proj', layer.self_attn.k_proj),
        ('v_proj', layer.self_attn.v_proj),
        ('o_proj', layer.self_attn.o_proj),
        ('gate_proj', layer.mlp.gate_proj),
        ('up_proj', layer.mlp.up_proj),
        ('down_proj', layer.mlp.down_proj),
    ]
    
    features = {}
    for name, mod in modules:
        w = mod.weight.data.float()
        
        # Basic stats
        norm = w.norm().item()
        mean_abs = w.abs().mean().item()
        
        # Spectral norm
        try:
            spec_norm = torch.linalg.svdvals(w)[0].item()
        except Exception:
            spec_norm = 0.0
        
        # DCT energy entropy
        m, n = w.shape
        mp = 1 << max(0, math.ceil(math.log2(m))) if m > 1 else 1
        np = 1 << max(0, math.ceil(math.log2(n))) if n > 1 else 1
        w_pad = torch.zeros(mp, np, dtype=torch.float32, device=w.device)
        w_pad[:m, :n] = w
        dct = torch.fft.fft2(w_pad).abs()
        dct_flat = dct.reshape(-1)
        probs = dct_flat / dct_flat.sum()
        dct_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # WAL program entropy (fast version with K=32)
        atoms = build_l0_atoms(w.reshape(-1), K=32, iters=1)
        coeffs = build_coeff_table(w.reshape(-1), atoms, C=4, iters=1)
        prog, _ = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=65_536)
        
        atom_counts = torch.bincount(prog.atom_ids.long(), minlength=32).float()
        atom_probs = atom_counts / atom_counts.sum()
        atom_entropy = -(atom_probs * torch.log(atom_probs + 1e-10)).sum().item()
        
        coeff_counts = torch.bincount(prog.coeff_ids.long(), minlength=4).float()
        coeff_probs = coeff_counts / coeff_counts.sum()
        coeff_entropy = -(coeff_probs * torch.log(coeff_probs + 1e-10)).sum().item()
        
        features[name] = {
            'norm': norm,
            'mean_abs': mean_abs,
            'spectral_norm': spec_norm,
            'dct_entropy': dct_entropy,
            'atom_entropy': atom_entropy,
            'coeff_entropy': coeff_entropy,
        }
    
    return features


def compute_depth_fft(signal):
    """Compute FFT of a signal over depth."""
    signal_t = torch.tensor(signal, dtype=torch.float32)
    fft = torch.fft.fft(signal_t)
    freqs = torch.fft.fftfreq(len(signal))
    amps = fft.abs()
    
    # Top frequencies (excluding DC)
    top_freqs = []
    for i in torch.topk(amps[1:len(amps)//2], min(3, len(amps)//2)).indices + 1:
        top_freqs.append({
            'freq': freqs[i].item(),
            'period': 1.0 / abs(freqs[i].item()) if freqs[i].item() != 0 else float('inf'),
            'amplitude': amps[i].item(),
        })
    
    return top_freqs


def main():
    print("=" * 60)
    print("M186 — Wave Depth Map")
    print("=" * 60)
    
    device = "cuda:3"
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    n_layers = len(model.model.layers)
    
    # Collect features for all layers
    print(f"\nCollecting features for {n_layers} layers...")
    all_features = []
    for li in range(n_layers):
        features = compute_layer_features(model, li)
        all_features.append(features)
        if li % 8 == 0:
            print(f"  Layer {li}/{n_layers}")
    
    # Aggregate by module type
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    print("\n--- Depth FFT Analysis ---")
    
    results = {}
    for m_name in modules:
        signals = {
            'norm': [f[m_name]['norm'] for f in all_features],
            'mean_abs': [f[m_name]['mean_abs'] for f in all_features],
            'spectral_norm': [f[m_name]['spectral_norm'] for f in all_features],
            'dct_entropy': [f[m_name]['dct_entropy'] for f in all_features],
            'atom_entropy': [f[m_name]['atom_entropy'] for f in all_features],
        }
        
        print(f"\n{m_name}:")
        for signal_name, signal in signals.items():
            fft_result = compute_depth_fft(signal)
            if fft_result:
                top = fft_result[0]
                print(f"  {signal_name:15s}: period={top['period']:6.1f}, amp={top['amplitude']:.2e}")
        
        results[m_name] = signals
    
    # Summary stats
    print(f"\n{'='*60}")
    print("Layer-wise Trends")
    print(f"{'='*60}")
    
    for m_name in modules:
        norms = results[m_name]['norm']
        early = sum(norms[:4]) / 4
        mid = sum(norms[14:18]) / 4
        late = sum(norms[-4:]) / 4
        print(f"{m_name:10s}: early={early:10.2f}, mid={mid:10.2f}, late={late:10.2f}, ratio_l/e={late/early:.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m186_wave_depth_map.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
