#!/usr/bin/env python3
"""
M144 / Track 7: Graph-WAL Probe

Goal: Test if graph-based Fourier transform improves WAL encode/decode.

Hypothesis:
  Hidden dimensions may not have natural ordering. Graph Fourier (via
  channel similarity graph) may be more appropriate than ordinary FFT/DCT.

Method per layer:
  1. Build channel similarity graph (cosine similarity of rows/columns)
  2. Compute graph Laplacian
  3. Eigenvectors = graph Fourier basis
  4. Project weight matrix onto graph Fourier basis
  5. Keep top-K coefficients, reconstruct
  6. Compare with Raw, DCT, RandOrth (from M142)
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


def uniform_quantize(weight, num_bins=256):
    w = weight.float()
    min_val, max_val = w.min(), w.max()
    scale = (max_val - min_val) / (num_bins - 1) if max_val > min_val else 1.0
    quantized = ((w - min_val) / scale).round().clamp(0, num_bins - 1)
    return quantized * scale + min_val


def transform_raw(weight):
    return weight, lambda x: x


def transform_dct2(weight):
    try:
        import scipy.fft
        w_np = weight.numpy()
        W_dct = scipy.fft.dctn(w_np, type=2, norm='ortho')
        def inverse(w_t):
            return torch.from_numpy(scipy.fft.idctn(w_t.numpy(), type=2, norm='ortho')).float()
        return torch.from_numpy(W_dct).float(), inverse
    except Exception as e:
        return None, None


def transform_randorth(weight):
    out_d, in_d = weight.shape
    Q_out, _ = torch.linalg.qr(torch.randn(out_d, out_d))
    Q_in, _ = torch.linalg.qr(torch.randn(in_d, in_d))
    W_rot = Q_out @ weight @ Q_in.T
    def inverse(w_t):
        return Q_out.T @ w_t @ Q_in
    return W_rot, inverse


def transform_graph_fourier(weight, k_neighbors=10):
    """Graph Fourier transform via row-wise channel similarity."""
    try:
        out_d, in_d = weight.shape
        
        # Build similarity graph on rows (out channels)
        rows = weight  # [out_d, in_d]
        row_norm = rows / (rows.norm(dim=1, keepdim=True) + 1e-8)
        sim = row_norm @ row_norm.T  # [out_d, out_d]
        
        # K-nearest neighbors adjacency
        adj = torch.zeros_like(sim)
        for i in range(out_d):
            topk = sim[i].topk(min(k_neighbors, out_d), largest=True).indices
            adj[i, topk] = sim[i, topk]
        
        # Symmetrize
        adj = (adj + adj.T) / 2
        
        # Degree matrix
        degree = adj.sum(dim=1)
        D = torch.diag(degree)
        
        # Laplacian
        L = D - adj
        
        # Eigenvectors (graph Fourier basis)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Project weight onto graph Fourier basis: W_gft = U^T @ W
        W_gft = eigenvectors.T @ weight
        
        def inverse(w_t):
            return eigenvectors @ w_t
        
        return W_gft, inverse
    except Exception as e:
        print(f"    Graph Fourier failed: {e}")
        return None, None


def transform_graph_fourier_both(weight, k_neighbors=10):
    """Graph Fourier on both rows and columns."""
    try:
        out_d, in_d = weight.shape
        
        # Row graph
        row_norm = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
        sim_r = row_norm @ row_norm.T
        adj_r = torch.zeros_like(sim_r)
        for i in range(out_d):
            topk = sim_r[i].topk(min(k_neighbors, out_d), largest=True).indices
            adj_r[i, topk] = sim_r[i, topk]
        adj_r = (adj_r + adj_r.T) / 2
        L_r = torch.diag(adj_r.sum(dim=1)) - adj_r
        _, U_r = torch.linalg.eigh(L_r)
        
        # Column graph
        col_norm = weight.T / (weight.T.norm(dim=1, keepdim=True) + 1e-8)
        sim_c = col_norm @ col_norm.T
        adj_c = torch.zeros_like(sim_c)
        for i in range(in_d):
            topk = sim_c[i].topk(min(k_neighbors, in_d), largest=True).indices
            adj_c[i, topk] = sim_c[i, topk]
        adj_c = (adj_c + adj_c.T) / 2
        L_c = torch.diag(adj_c.sum(dim=1)) - adj_c
        _, U_c = torch.linalg.eigh(L_c)
        
        # 2D graph Fourier: U_r^T @ W @ U_c
        W_gft = U_r.T @ weight @ U_c
        
        def inverse(w_t):
            return U_r @ w_t @ U_c.T
        
        return W_gft, inverse
    except Exception as e:
        print(f"    Graph Fourier 2D failed: {e}")
        return None, None


def test_transform(name, weight, transform_fn, quantize_fn, top_k=256):
    W_t, inverse_fn = transform_fn(weight)
    if W_t is None:
        return None
    
    # Quantize
    W_t_quant = quantize_fn(W_t)
    
    # Inverse
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
        'psnr': 20 * math.log10(weight.abs().max().item() / math.sqrt(mse + 1e-10)),
    }


def main():
    print("=" * 70)
    print("M144 / Track 7: Graph-WAL Probe")
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
        'RandOrth': transform_randorth,
        'GraphRow': lambda w: transform_graph_fourier(w, k_neighbors=10),
        'Graph2D': lambda w: transform_graph_fourier_both(w, k_neighbors=10),
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
            result = test_transform(t_name, weight, t_fn, uniform_quantize, top_k=256)
            if result:
                layer_results[t_name] = result
                print(f"    {t_name:12s}: MSE={result['mse']:.8f}, PSNR={result['psnr']:.2f}dB, RelErr={result['rel_err']:.6f}")
        
        all_results[layer_name] = layer_results

    # 4. Summary
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
    
    out_path = 'experiments/m144_graph_wal_probe.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M144 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
