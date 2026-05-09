#!/usr/bin/env python3
"""
M143 / Track 6: Wave-Atom ISA Probe

Goal: Test if weights can be represented as wave programs instead of scalar atom×coeff.

Current WAL:
  weight = atom[id] × coeff[id] + residual

Wave-WAL:
  weight[i,j] = Σ A_k × cos(ωx_k × i + ωy_k × j + φ_k) + residual[i,j]

Method:
  1. Take layer weight W
  2. Apply DCT2 → frequency coefficients
  3. Keep top-K coefficients (by magnitude)
  4. Reconstruct via IDCT2 with top-K only
  5. Compare with scalar WAL (k-means) reconstruction
  6. Measure: MSE, size, expressivity
"""

import os, sys, json, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def get_layer_weights(model, layer_names):
    weights = {}
    for name in layer_names:
        parts = name.split('.')
        layer = model
        for p in parts:
            layer = getattr(layer, p)
        weights[name] = layer.weight.data.float().cpu().clone()
    return weights


def scalar_wal_recon(weight, num_atoms=256):
    """Scalar WAL reconstruction: k-means atoms + nearest match."""
    w = weight.flatten()
    sample = w[torch.randperm(len(w))[:min(50000, len(w))]]
    atoms = sample[torch.randperm(len(sample))[:num_atoms]].clone()
    for _ in range(5):
        dists = (w.unsqueeze(1) - atoms.unsqueeze(0)).abs()
        labels = dists.argmin(dim=1)
        for k in range(num_atoms):
            mask = labels == k
            if mask.sum() > 0:
                atoms[k] = w[mask].mean()
    dists = (w.unsqueeze(1) - atoms.unsqueeze(0)).abs()
    best = dists.argmin(dim=1)
    recon = atoms[best]
    return recon.reshape(weight.shape)


def wave_wal_recon(weight, top_k=256):
    """Wave-WAL reconstruction: top-K DCT coefficients."""
    try:
        import scipy.fft
        w_np = weight.numpy()
        W_dct = scipy.fft.dctn(w_np, type=2, norm='ortho')
        
        # Flatten and find top-K by magnitude
        flat = torch.from_numpy(W_dct).float().flatten()
        top_indices = flat.abs().topk(min(top_k, len(flat))).indices
        
        # Create sparse DCT with only top-K
        sparse_dct = torch.zeros_like(flat)
        sparse_dct[top_indices] = flat[top_indices]
        sparse_dct = sparse_dct.reshape(W_dct.shape).numpy()
        
        # Reconstruct
        recon = scipy.fft.idctn(sparse_dct, type=2, norm='ortho')
        return torch.from_numpy(recon).float()
    except Exception as e:
        print(f"    Wave-WAL failed: {e}")
        return None


def wave_wal_recon_fft(weight, top_k=256):
    """Wave-WAL reconstruction: top-K FFT coefficients."""
    try:
        W_fft = torch.fft.fft2(weight)
        flat = W_fft.flatten()
        magnitudes = flat.abs()
        top_indices = magnitudes.topk(min(top_k, len(flat))).indices
        
        # Create sparse FFT
        sparse_fft = torch.zeros_like(flat, dtype=torch.complex64)
        sparse_fft[top_indices] = flat[top_indices]
        sparse_fft = sparse_fft.reshape(W_fft.shape)
        
        # Reconstruct
        recon = torch.fft.ifft2(sparse_fft).real
        return recon
    except Exception as e:
        print(f"    Wave-WAL FFT failed: {e}")
        return None


def evaluate_recon(weight, recon, method_name):
    if recon is None:
        return None
    error = (weight - recon).abs()
    return {
        'method': method_name,
        'mse': (error ** 2).mean().item(),
        'max_err': error.max().item(),
        'rel_err': error.mean().item() / (weight.abs().mean().item() + 1e-8),
        'psnr': 20 * math.log10(weight.abs().max().item() / math.sqrt((error ** 2).mean().item() + 1e-10)),
    }


def main():
    print("=" * 70)
    print("M143 / Track 6: Wave-Atom ISA Probe")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Select layers
    layer_names = [
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.0.self_attn.v_proj',
        'model.layers.0.self_attn.o_proj',
        'model.layers.0.mlp.gate_proj',
    ]
    
    top_k_values = [64, 128, 256, 512, 1024]
    all_results = {}

    # 3. Test each layer
    print(f"[2] Testing {len(layer_names)} layers x {len(top_k_values)} K values...")
    for layer_name in layer_names:
        print(f"\n  Layer: {layer_name}")
        layer = model
        for p in layer_name.split('.'):
            layer = getattr(layer, p)
        weight = layer.weight.data.float().cpu()
        
        layer_results = {}
        
        # Baseline: scalar WAL
        scalar_recon = scalar_wal_recon(weight, num_atoms=256)
        layer_results['Scalar_WAL_256'] = evaluate_recon(weight, scalar_recon, 'Scalar_WAL_256')
        
        # Wave-WAL with different K
        for k in top_k_values:
            dct_recon = wave_wal_recon(weight, top_k=k)
            result = evaluate_recon(weight, dct_recon, f'Wave_DCT_{k}')
            if result:
                layer_results[f'Wave_DCT_{k}'] = result
                print(f"    Wave DCT K={k:4d}: MSE={result['mse']:.8f}, PSNR={result['psnr']:.2f}dB, RelErr={result['rel_err']:.6f}")
        
        # FFT variant with K=256
        fft_recon = wave_wal_recon_fft(weight, top_k=256)
        result = evaluate_recon(weight, fft_recon, 'Wave_FFT_256')
        if result:
            layer_results['Wave_FFT_256'] = result
            print(f"    Wave FFT K= 256: MSE={result['mse']:.8f}, PSNR={result['psnr']:.2f}dB, RelErr={result['rel_err']:.6f}")
        
        if 'Scalar_WAL_256' in layer_results and layer_results['Scalar_WAL_256']:
            s = layer_results['Scalar_WAL_256']
            print(f"    Scalar WAL 256  : MSE={s['mse']:.8f}, PSNR={s['psnr']:.2f}dB, RelErr={s['rel_err']:.6f}")
        
        all_results[layer_name] = layer_results

    # 4. Summary: compare Wave K=256 vs Scalar WAL
    print(f"\n[3] Summary: Wave-WAL vs Scalar-WAL (both using ~256 params)")
    print(f"  {'Layer':50s} {'Wave MSE':12s} {'Scalar MSE':12s} {'Wave Better':10s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*10}")
    
    wave_wins = 0
    scalar_wins = 0
    for layer_name, layer_results in all_results.items():
        wave = layer_results.get('Wave_DCT_256')
        scalar = layer_results.get('Scalar_WAL_256')
        if wave and scalar:
            better = wave['mse'] < scalar['mse']
            ratio = scalar['mse'] / wave['mse'] if wave['mse'] > 0 else 1.0
            if better:
                wave_wins += 1
            else:
                scalar_wins += 1
            print(f"  {layer_name:50s} {wave['mse']:12.8f} {scalar['mse']:12.8f} {ratio:9.2f}x")

    print(f"\n  Wave-WAL wins: {wave_wins}/{wave_wins+scalar_wins} layers")

    # 5. Size analysis
    print(f"\n[4] Size analysis (per layer):")
    print(f"  Scalar WAL: 256 atoms × 1 float + 256 coeffs × 1 float + programs")
    print(f"  Wave DCT K=256: 256 positions × 2 ints + 256 amplitudes × 1 float")
    print(f"  Wave FFT K=256: 256 positions × 2 ints + 256 complex values × 2 floats")

    # 6. Save
    output = {
        'layers_tested': len(layer_names),
        'top_k_values': top_k_values,
        'results': {k: {kk: {kkk: float(vvv) if isinstance(vvv, (int, float)) else vvv for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in all_results.items()},
        'wave_wins': wave_wins,
        'scalar_wins': scalar_wins,
    }
    
    out_path = 'experiments/m143_wave_atom_isa.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M143 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
