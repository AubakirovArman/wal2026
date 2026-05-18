#!/usr/bin/env python3
"""M187 — Program-Wave.

Analyzes how WAL program usage changes across layers.
"""
import torch, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def main():
    print("=" * 60)
    print("M187 — Program-Wave")
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
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    K, C = 32, 4
    
    # Build global atom/coeff table
    print("\nBuilding global atom table...")
    # Build atom table on CPU with subset of weights
    print("  Collecting weights on CPU...")
    all_weights = []
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            all_weights.append(mod.weight.data.reshape(-1).cpu())
    
    all_flat = torch.cat(all_weights)
    print(f"  Total weights: {all_flat.numel()/1e9:.2f}B, building atoms on CPU...")
    atoms = build_l0_atoms(all_flat, K=K, iters=1)
    coeffs = build_coeff_table(all_flat, atoms, C=C, iters=1)
    atoms = atoms.to(device)
    coeffs = coeffs.to(device)
    
    # Encode all layers and collect program frequencies
    print("\nEncoding all layers...")
    layer_atom_freqs = []  # [layer, module, K]
    layer_coeff_freqs = []  # [layer, module, C]
    
    for li in range(n_layers):
        layer = model.model.layers[li]
        atom_freqs = {}
        coeff_freqs = {}
        
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            w = mod.weight.data
            prog, _ = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=65_536)
            
            atom_counts = torch.bincount(prog.atom_ids.long(), minlength=K).float()
            atom_freqs[name] = atom_counts / atom_counts.sum()
            
            coeff_counts = torch.bincount(prog.coeff_ids.long(), minlength=C).float()
            coeff_freqs[name] = coeff_counts / coeff_counts.sum()
        
        layer_atom_freqs.append(atom_freqs)
        layer_coeff_freqs.append(coeff_freqs)
        
        if li % 8 == 0:
            print(f"  Layer {li}/{n_layers}")
    
    # FFT analysis over depth for each atom and coeff
    print("\n--- Program-Wave FFT Analysis ---")
    
    results = {}
    for m_name in modules:
        # Atom waves
        atom_signals = torch.stack([layer_atom_freqs[li][m_name] for li in range(n_layers)])  # [L, K]
        
        print(f"\n{m_name}:")
        print("  Atom waves:")
        for atom_id in range(K):
            signal = atom_signals[:, atom_id]
            fft = torch.fft.fft(signal)
            freqs = torch.fft.fftfreq(n_layers)
            amps = fft.abs()
            
            # Top non-DC frequency
            top_idx = torch.topk(amps[1:n_layers//2], 1).indices[0] + 1
            period = 1.0 / abs(freqs[top_idx].item()) if freqs[top_idx].item() != 0 else float('inf')
            amp = amps[top_idx].item()
            
            if amp > 0.5:  # Only report significant waves
                print(f"    atom[{atom_id:2d}]: period={period:6.1f}, amp={amp:.3f}")
        
        # Coeff waves
        coeff_signals = torch.stack([layer_coeff_freqs[li][m_name] for li in range(n_layers)])  # [L, C]
        
        print("  Coeff waves:")
        for coeff_id in range(C):
            signal = coeff_signals[:, coeff_id]
            fft = torch.fft.fft(signal)
            freqs = torch.fft.fftfreq(n_layers)
            amps = fft.abs()
            
            top_idx = torch.topk(amps[1:n_layers//2], 1).indices[0] + 1
            period = 1.0 / abs(freqs[top_idx].item()) if freqs[top_idx].item() != 0 else float('inf')
            amp = amps[top_idx].item()
            
            if amp > 0.5:
                print(f"    coeff[{coeff_id}]: period={period:6.1f}, amp={amp:.3f}")
        
        # Summary: which atoms/coeffs have strongest depth-periodicity
        atom_periodic = [(i, 1.0/abs(torch.fft.fftfreq(n_layers)[torch.topk(torch.fft.fft(atom_signals[:, i]).abs()[1:n_layers//2], 1).indices[0]+1].item()), 
                         torch.fft.fft(atom_signals[:, i]).abs()[torch.topk(torch.fft.fft(atom_signals[:, i]).abs()[1:n_layers//2], 1).indices[0]+1].item()) 
                        for i in range(K)]
        atom_periodic = sorted(atom_periodic, key=lambda x: x[2], reverse=True)[:3]
        
        results[m_name] = {
            'top_atoms': atom_periodic,
        }
    
    print(f"\n{'='*60}")
    print("Summary: Most periodic atoms/coeffs")
    print(f"{'='*60}")
    for m_name, r in results.items():
        print(f"{m_name}: top periodic atoms = {[a[0] for a in r['top_atoms']]}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m187_program_wave.json', 'w') as f:
        json.dump({k: {'top_atoms': [(int(a[0]), float(a[1]), float(a[2])) for a in v['top_atoms']]} for k, v in results.items()}, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
