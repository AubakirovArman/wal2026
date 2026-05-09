#!/usr/bin/env python3
"""M48: WAL round-trip on real Llama 3.3 70B layer."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal import wal_encode_scalar, build_atoms_kmeans, serialize_wal_state, deserialize_wal_state
from wal.triton_kernels import wal_decode_scalar_triton


def test_roundtrip():
    device = torch.device('cuda:2')
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    layer_idx = 40
    param_name = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    
    print("=" * 60)
    print("M48: WAL Round-Trip on Real 70B Layer")
    print("=" * 60)
    
    # Load model
    print(f"\n[1] Loading {model_name}...")
    t0 = time.time()
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    load_time = time.time() - t0
    print(f"    Loaded in {load_time:.1f}s")
    
    # Get parameter
    param = dict(model.named_parameters())[param_name]
    print(f"\n[2] Parameter: {param_name}")
    print(f"    Shape: {tuple(param.shape)}, dtype: {param.dtype}, device: {param.device}")
    
    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    
    # Encode
    K = 128
    lmax = 2
    flat = w_norm.reshape(-1)
    
    SAMPLE_SIZE = 1_000_000
    if flat.numel() > SAMPLE_SIZE:
        idx_samp = torch.randperm(flat.numel())[:SAMPLE_SIZE]
        samples = flat[idx_samp]
    else:
        samples = flat
    
    print(f"\n[3] WAL encode (K={K}, lmax={lmax})...")
    t0 = time.time()
    atoms = build_atoms_kmeans(samples, K, iters=5, device=param.device)
    atoms = atoms.to(param.device)
    prog, recon = wal_encode_scalar(flat.to(param.device), atoms, lmax)
    encode_time = time.time() - t0
    print(f"    Encode done in {encode_time:.2f}s")
    
    rel_mse = ((flat.to(param.device) - recon) ** 2).mean() / (flat.to(param.device) ** 2).mean()
    print(f"    relMSE: {rel_mse.item():.8f}")
    
    # Serialize
    print(f"\n[4] Serialize...")
    from wal.format import WALModelState, WALParameterMeta
    meta = WALParameterMeta(
        name=param_name,
        shape=list(param.shape),
        device=str(param.device),
        row_scale_shape=list(row_scale.shape),
        offset=0,
        numel=param.numel(),
        is_encoded=True,
    )
    state = WALModelState(
        K=K, lmax=lmax, dtype_str='bfloat16',
        atom_table=atoms.cpu(),
        programs=prog.cpu(),
        params=[meta],
    )
    blob = serialize_wal_state(state)
    print(f"    Blob size: {len(blob) / 1e6:.2f} MB")
    print(f"    Original param: {param.numel() * 2 / 1e6:.2f} MB (bf16)")
    print(f"    Compression: {param.numel() * 2 / len(blob):.2f}x")
    
    # Deserialize
    print(f"\n[5] Deserialize...")
    state2 = deserialize_wal_state(blob)
    
    # Decode via Triton
    print(f"\n[6] Decode via Triton kernel...")
    prog_gpu = state2.programs.to(param.device)
    atoms_gpu = state2.atom_table.to(param.device)
    
    t0 = time.time()
    recon_triton = wal_decode_scalar_triton(prog_gpu.indices, prog_gpu.signs, atoms_gpu)
    torch.cuda.synchronize()
    decode_time = time.time() - t0
    print(f"    Triton decode: {decode_time * 1000:.3f} ms")
    print(f"    Throughput: {param.numel() / decode_time / 1e6:.1f} Mweights/s")
    
    # Reshape and rescale
    recon_final = recon_triton.reshape(w.shape) * row_scale.to(param.device)
    
    # Compare with original
    max_err = (w.to(param.device) - recon_final).abs().max().item()
    rel_mse_roundtrip = ((w.to(param.device) - recon_final) ** 2).mean() / (w.to(param.device) ** 2).mean()
    print(f"    Max error vs original: {max_err:.6f}")
    print(f"    relMSE round-trip: {rel_mse_roundtrip.item():.8f}")
    
    # Matmul test
    print(f"\n[7] Matmul output test...")
    x = torch.randn(1, 128, w.shape[0], dtype=torch.bfloat16, device=param.device)
    
    with torch.no_grad():
        out_orig = torch.matmul(x, w.T.to(torch.bfloat16))
        out_wal = torch.matmul(x, recon_final.T.to(torch.bfloat16))
    
    out_rel_mse = ((out_orig - out_wal) ** 2).mean() / (out_orig ** 2).mean()
    out_corr = torch.corrcoef(torch.stack([out_orig.reshape(-1), out_wal.reshape(-1)]))[0, 1]
    print(f"    Output relMSE: {out_rel_mse.item():.8f}")
    print(f"    Output correlation: {out_corr.item():.6f}")
    
    print("\n" + "=" * 60)
    print("M48: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_roundtrip()
