#!/usr/bin/env python3
"""
M142 / Track 5: Transform-WAL Probe

Goal: Test if spectral/transform space improves WAL encode/decode quality.

Pipeline per layer:
  W_dense → Transform(W) → WAL-like quantize → dequantize → InverseTransform → W_recon
  Compare ||W - W_recon|| for different transforms

Transforms tested:
  - Raw (baseline)
  - DCT2 (2D discrete cosine transform)
  - FFT2 (2D fast Fourier transform, magnitude)
  - Hadamard (Walsh-Hadamard, via scipy)
  - Random Orthogonal (random orthogonal matrix)

Metrics:
  - Reconstruction MSE
  - Max error
  - Relative error
"""

import os, sys, json, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def get_layer_weights(model, layer_names):
    """Extract weights from named layers."""
    weights = {}
    for name in layer_names:
        parts = name.split('.')
        layer = model
        for p in parts:
            layer = getattr(layer, p)
        weights[name] = layer.weight.data.float().cpu().clone()
    return weights


def wal_quantize_proxy(weight, num_bins=256):
    """Fast proxy for WAL encode/decode: uniform quantization."""
    w = weight.float()
    min_val, max_val = w.min(), w.max()
    scale = (max_val - min_val) / (num_bins - 1) if max_val > min_val else 1.0
    quantized = ((w - min_val) / scale).round().clamp(0, num_bins - 1)
    recon = quantized * scale + min_val
    return recon


def transform_raw(weight):
    """No transform."""
    return weight, lambda x: x


def transform_dct2(weight):
    """2D DCT transform."""
    try:
        import scipy.fft
        w_np = weight.numpy()
        W_dct = scipy.fft.dctn(w_np, type=2, norm='ortho')
        def inverse(w_t):
            return torch.from_numpy(scipy.fft.idctn(w_t.numpy(), type=2, norm='ortho')).float()
        return torch.from_numpy(W_dct).float(), inverse
    except Exception as e:
        print(f"    DCT2 failed: {e}")
        return None, None


def transform_fft2(weight):
    """2D FFT transform, use real part."""
    try:
        W_fft = torch.fft.fft2(weight)
        W_real = torch.view_as_real(W_fft)
        def inverse(w_t):
            w_complex = torch.view_as_complex(w_t.contiguous())
            return torch.fft.ifft2(w_complex).real
        return W_real, inverse
    except Exception as e:
        print(f"    FFT2 failed: {e}")
        return None, None


def transform_hadamard(weight):
    """Hadamard transform (requires power-of-2 dimensions)."""
    try:
        # Pad to next power of 2
        def next_pow2(n):
            return 2 ** math.ceil(math.log2(n))
        
        h_out, h_in = weight.shape
        p_out, p_in = next_pow2(h_out), next_pow2(h_in)
        padded = torch.zeros(p_out, p_in, dtype=weight.dtype)
        padded[:h_out, :h_in] = weight
        
        # Hadamard matrix
        def hadamard_matrix(n):
            H = torch.tensor([[1.]], dtype=torch.float32)
            while H.shape[0] < n:
                H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
            return H / math.sqrt(2)
        
        H_out = hadamard_matrix(p_out)
        H_in = hadamard_matrix(p_in)
        W_had = H_out @ padded @ H_in.T
        
        def inverse(w_t):
            recon = H_out.T @ w_t @ H_in
            return recon[:h_out, :h_in]
        
        return W_had, inverse
    except Exception as e:
        print(f"    Hadamard failed: {e}")
        return None, None


def transform_random_orthogonal(weight):
    """Random orthogonal transform."""
    try:
        out_d, in_d = weight.shape
        # Q, R decomposition gives orthogonal Q
        Q_out, _ = torch.linalg.qr(torch.randn(out_d, out_d))
        Q_in, _ = torch.linalg.qr(torch.randn(in_d, in_d))
        W_rot = Q_out @ weight @ Q_in.T
        
        def inverse(w_t):
            return Q_out.T @ w_t @ Q_in
        
        return W_rot, inverse
    except Exception as e:
        print(f"    RandomOrth failed: {e}")
        return None, None


def test_transform(name, weight, transform_fn, quantize_fn):
    """Test one transform: forward, quantize, inverse, compare."""
    W_t, inverse_fn = transform_fn(weight)
    if W_t is None:
        return None
    
    # Quantize in transform space
    W_t_quant = quantize_fn(W_t)
    
    # Inverse transform
    W_recon = inverse_fn(W_t_quant)
    
    # Compare
    error = (weight - W_recon).abs()
    mse = (error ** 2).mean().item()
    max_err = error.max().item()
    rel_err = error.mean().item() / (weight.abs().mean().item() + 1e-8)
    
    return {
        'transform': name,
        'mse': mse,
        'max_err': max_err,
        'rel_err': rel_err,
        'transform_shape': list(W_t.shape),
    }


def main():
    print("=" * 70)
    print("M142 / Track 5: Transform-WAL Probe")
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
        'model.language_model.layers.0.self_attn.q_proj',
        'model.language_model.layers.0.self_attn.k_proj',
        'model.language_model.layers.0.self_attn.v_proj',
        'model.language_model.layers.0.self_attn.o_proj',
        'model.language_model.layers.0.mlp.gate_proj',
    ]
    
    transforms = {
        'Raw': transform_raw,
        'DCT2': transform_dct2,
        'FFT2': transform_fft2,
        'Hadamard': transform_hadamard,
        'RandOrth': transform_random_orthogonal,
    }

    all_results = {}

    # 3. Test each layer
    print(f"[2] Testing {len(layer_names)} layers x {len(transforms)} transforms...")
    for layer_name in layer_names:
        print(f"\n  Layer: {layer_name}")
        layer = model
        for p in layer_name.split('.'):
            layer = getattr(layer, p)
        weight = layer.weight.data.float().cpu()
        
        layer_results = {}
        for t_name, t_fn in transforms.items():
            result = test_transform(t_name, weight, t_fn, wal_quantize_proxy)
            if result:
                layer_results[t_name] = result
                print(f"    {t_name:12s}: MSE={result['mse']:.8f}, MaxErr={result['max_err']:.6f}, RelErr={result['rel_err']:.6f}")
        
        all_results[layer_name] = layer_results

    # 4. Summary table
    print(f"\n[3] Summary: best transform per layer (by MSE)")
    print(f"  {'Layer':50s} {'Best':12s} {'MSE':12s} {'vs Raw':8s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*8}")
    
    for layer_name, layer_results in all_results.items():
        if not layer_results:
            continue
        best_name = min(layer_results.items(), key=lambda x: x[1]['mse'])[0]
        best_mse = layer_results[best_name]['mse']
        raw_mse = layer_results.get('Raw', {}).get('mse', float('inf'))
        improvement = raw_mse / best_mse if best_mse > 0 else 1.0
        print(f"  {layer_name:50s} {best_name:12s} {best_mse:12.8f} {improvement:7.2f}x")

    # 5. Aggregate
    print(f"\n[4] Aggregate across all layers:")
    transform_scores = {name: [] for name in transforms.keys()}
    for layer_results in all_results.values():
        for t_name, result in layer_results.items():
            transform_scores[t_name].append(result['mse'])
    
    print(f"  {'Transform':12s} {'Avg MSE':12s} {'Best Count':10s}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")
    for t_name in transforms.keys():
        scores = transform_scores.get(t_name, [])
        if scores:
            avg_mse = sum(scores) / len(scores)
            best_count = sum(1 for lr in all_results.values() if lr and min(lr.items(), key=lambda x: x[1]['mse'])[0] == t_name)
            print(f"  {t_name:12s} {avg_mse:12.8f} {best_count:10d}")

    # 6. Save
    output = {
        'layers_tested': len(layer_names),
        'transforms_tested': list(transforms.keys()),
        'results': all_results,
    }
    
    out_path = 'experiments/m142_transform_wal_probe.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M142 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
