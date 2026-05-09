"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M160 v3 — Spectral Energy Map with real model weights."""
import torch
import json
from scipy.fft import dctn
from transformers import AutoModelForCausalLM


def main():
    print("=" * 60)
    print("M160 v3 — Spectral Energy Map (real weights)")
    print("=" * 60)
    
    print("\nLoading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    layers = [0, 8, 16, 24, 31]
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj']
    
    results = []
    
    for li in layers:
        for m in modules:
            if hasattr(model.model.layers[li].self_attn, m):
                w = getattr(model.model.layers[li].self_attn, m).weight.data.float().numpy()
                name = f"layers.{li}.self_attn.{m}"
            elif hasattr(model.model.layers[li].mlp, m):
                w = getattr(model.model.layers[li].mlp, m).weight.data.float().numpy()
                name = f"layers.{li}.mlp.{m}"
            else:
                continue
            
            dct = dctn(w, type=2, norm='ortho')
            dct = torch.from_numpy(dct).abs()
            
            h, w_shape = dct.shape
            cy, cx = h // 2, w_shape // 2
            
            ll = dct[:cy, :cx].sum().item()
            lh = dct[:cy, cx:].sum().item()
            hl = dct[cy:, :cx].sum().item()
            hh = dct[cy:, cx:].sum().item()
            total = ll + lh + hl + hh
            
            result = {
                'name': name,
                'shape': [h, w_shape],
                'll_ratio': ll / total,
                'lh_ratio': lh / total,
                'hl_ratio': hl / total,
                'hh_ratio': hh / total,
            }
            results.append(result)
            
            print(f"  {name}: LL={result['ll_ratio']:.3f} LH={result['lh_ratio']:.3f} "
                  f"HL={result['hl_ratio']:.3f} HH={result['hh_ratio']:.3f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m160_spectral_energy_map.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
