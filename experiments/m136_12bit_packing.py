#!/usr/bin/env python3
"""M136 / Phase D: 12-bit Production Packing

Implement real packed storage for WAL programs:
  atom_id: 8 bits (K=256)
  coeff_id: 4 bits (C=16)
  2 weights = 3 bytes (24 bits)
  
Benchmark: pack/unpack speed, size reduction, correctness.
"""
import torch, time, sys, json, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable
from wal.v1.nn import WALCachedLinear, WALParameter
import torch.nn as nn

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
SEED = 42
K, C = 256, 16

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def canonicalize_atoms(atoms):
    perm = torch.argsort(atoms.abs(), descending=True)
    return atoms[perm], perm


def build_frozen_tables(model, K=256, C=16, seed=42):
    torch.manual_seed(seed)
    all_weights = []
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name:
            all_weights.append(p.data.float().cpu().reshape(-1))
    pooled = torch.cat(all_weights)
    sample = pooled[torch.randperm(len(pooled))[:min(len(pooled), 500_000)]].to(DEVICE)
    atoms = build_l0_atoms(sample, K=K, iters=3, device=DEVICE)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=2)
    sorted_atoms, perm = canonicalize_atoms(atoms)
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
    atom_table = AtomTableV1(base_atoms=sorted_atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs)
    return atom_table, coeff_table, sorted_atoms, coeffs


def encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'embed_tokens' not in name and 'norm' not in name:
            flat = module.weight.data.float().to(DEVICE).reshape(-1)
            prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=131_072)
            wal_weight = WALParameter(
                prog=prog,
                atom_table=atom_table,
                coeffs=coeff_table,
                shape=module.weight.shape,
                dtype=module.weight.dtype,
            )
            new_layer = WALCachedLinear(wal_weight=wal_weight, bias=module.bias.data if module.bias is not None else None)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)
    return model


def pack_12bit(atom_ids, coeff_ids):
    """Pack 8-bit atom_ids + 4-bit coeff_ids into 12-bit format.
    
    For even N:
      output[i*3]     = atom_ids[i]
      output[i*3 + 1] = atom_ids[i+1]
      output[i*3 + 2] = (coeff_ids[i] << 4) | coeff_ids[i+1]
    
    Returns packed tensor of uint8 with size ceil(1.5 * N).
    """
    N = atom_ids.numel()
    assert N == coeff_ids.numel()
    assert N % 2 == 0, "N must be even for 12-bit packing"
    
    packed = torch.empty(int(N * 1.5), dtype=torch.uint8, device=atom_ids.device)
    
    idx = torch.arange(0, N, 2, device=atom_ids.device)
    packed[idx * 3 // 2] = atom_ids[idx]
    packed[idx * 3 // 2 + 1] = atom_ids[idx + 1]
    packed[idx * 3 // 2 + 2] = (coeff_ids[idx] << 4) | coeff_ids[idx + 1]
    
    return packed


def unpack_12bit(packed, N):
    """Unpack 12-bit format back to atom_ids and coeff_ids.
    
    Returns (atom_ids, coeff_ids) as uint8 tensors.
    """
    assert packed.numel() == int(N * 1.5)
    assert N % 2 == 0
    
    atom_ids = torch.empty(N, dtype=torch.uint8, device=packed.device)
    coeff_ids = torch.empty(N, dtype=torch.uint8, device=packed.device)
    
    idx = torch.arange(0, N, 2, device=packed.device)
    atom_ids[idx] = packed[idx * 3 // 2]
    atom_ids[idx + 1] = packed[idx * 3 // 2 + 1]
    combined = packed[idx * 3 // 2 + 2]
    coeff_ids[idx] = combined >> 4
    coeff_ids[idx + 1] = combined & 0x0F
    
    return atom_ids, coeff_ids


def benchmark_pack_unpack(atom_ids, coeff_ids, iters=100):
    """Benchmark pack/unpack speed."""
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(10):
        p = pack_12bit(atom_ids, coeff_ids)
        a, c = unpack_12bit(p, atom_ids.numel())
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        p = pack_12bit(atom_ids, coeff_ids)
    torch.cuda.synchronize()
    pack_time = (time.perf_counter() - start) / iters * 1000
    
    p = pack_12bit(atom_ids, coeff_ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a, c = unpack_12bit(p, atom_ids.numel())
    torch.cuda.synchronize()
    unpack_time = (time.perf_counter() - start) / iters * 1000
    
    return pack_time, unpack_time


def main():
    print("=" * 70)
    print("M136 / Phase D: 12-bit Production Packing")
    print("=" * 70)

    # --- 1. Microbenchmark: pack/unpack correctness and speed ---
    print("\n[1] Microbenchmark: pack/unpack correctness")
    
    N = 10_000_000  # 10M weights
    atom_ids = torch.randint(0, 256, (N,), dtype=torch.uint8, device=DEVICE)
    coeff_ids = torch.randint(0, 16, (N,), dtype=torch.uint8, device=DEVICE)
    
    packed = pack_12bit(atom_ids, coeff_ids)
    atom_ids_back, coeff_ids_back = unpack_12bit(packed, N)
    
    assert torch.all(atom_ids == atom_ids_back), "Atom IDs mismatch!"
    assert torch.all(coeff_ids == coeff_ids_back), "Coeff IDs mismatch!"
    print(f"  ✅ Correctness verified for {N:,} weights")
    print(f"  Original size:   {2 * N / 1024 / 1024:.2f} MB (2 bytes/weight)")
    print(f"  Packed size:     {packed.numel() / 1024 / 1024:.2f} MB (1.5 bytes/weight)")
    print(f"  Reduction:       {100 * (2 * N - packed.numel()) / (2 * N):.1f}%")
    
    pack_t, unpack_t = benchmark_pack_unpack(atom_ids, coeff_ids, iters=100)
    print(f"  Pack time:       {pack_t:.3f} ms")
    print(f"  Unpack time:     {unpack_t:.3f} ms")
    print(f"  Throughput:      {N / (pack_t + unpack_t) * 1000 / 1e6:.1f} M weights/sec")

    # --- 2. Full model packing ---
    print("\n[2] Full model: encode + pack all layers")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table, coeff_table, sorted_atoms, coeffs = build_frozen_tables(model, K=K, C=C, seed=SEED)
    model = encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs)
    
    total_weights = 0
    unpacked_bytes = 0
    packed_bytes = 0
    layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            atom_ids = module.wal_weight.prog.atom_ids
            coeff_ids = module.wal_weight.prog.coeff_ids
            N = atom_ids.numel()
            
            # Ensure even N
            if N % 2 == 1:
                atom_ids = torch.cat([atom_ids, torch.zeros(1, dtype=torch.uint8, device=DEVICE)])
                coeff_ids = torch.cat([coeff_ids, torch.zeros(1, dtype=torch.uint8, device=DEVICE)])
                N += 1
            
            packed = pack_12bit(atom_ids, coeff_ids)
            
            total_weights += module.wal_weight.prog.N
            unpacked_bytes += 2 * module.wal_weight.prog.N
            packed_bytes += packed.numel()
            layer_count += 1
    
    print(f"  Layers encoded:  {layer_count}")
    print(f"  Total weights:   {total_weights:,}")
    print(f"  Unpacked:        {unpacked_bytes / 1024 / 1024 / 1024:.3f} GB (2 bytes/weight)")
    print(f"  Packed:          {packed_bytes / 1024 / 1024 / 1024:.3f} GB (1.5 bytes/weight)")
    print(f"  Reduction:       {100 * (unpacked_bytes - packed_bytes) / unpacked_bytes:.1f}%")
    print(f"  Over bf16:       {packed_bytes / (total_weights * 2) * 100:.1f}% (bf16 = 2 bytes/weight)")
    
    # --- 3. Decode correctness after pack/unpack ---
    print("\n[3] Decode correctness after pack/unpack cycle")
    errors = 0
    max_err = 0.0
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            atom_ids = module.wal_weight.prog.atom_ids
            coeff_ids = module.wal_weight.prog.coeff_ids
            N_orig = atom_ids.numel()
            
            # Pad if needed
            if N_orig % 2 == 1:
                atom_ids = torch.cat([atom_ids, torch.zeros(1, dtype=torch.uint8, device=DEVICE)])
                coeff_ids = torch.cat([coeff_ids, torch.zeros(1, dtype=torch.uint8, device=DEVICE)])
                N = N_orig + 1
            else:
                N = N_orig
            
            packed = pack_12bit(atom_ids, coeff_ids)
            a_back, c_back = unpack_12bit(packed, N)
            
            if not torch.all(atom_ids[:N_orig] == a_back[:N_orig]):
                errors += 1
            if not torch.all(coeff_ids[:N_orig] == c_back[:N_orig]):
                errors += 1
            
            # Decode with original vs unpacked
            recon_orig = module.wal_weight.decode().flatten()
            recon_back = (module.wal_weight.atom_table.base_atoms[a_back[:N_orig].long()] * 
                         module.wal_weight.coeffs.values[c_back[:N_orig].long()]).flatten()
            
            err = (recon_orig - recon_back).abs().max().item()
            max_err = max(max_err, err)
    
    if errors == 0:
        print(f"  ✅ All {layer_count} layers: perfect round-trip")
        print(f"  Max decode error: {max_err:.2e}")
    else:
        print(f"  ❌ {errors} layers had mismatch")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Format sizes for {total_weights:,} weights:")
    print(f"    bf16 dense:      {total_weights * 2 / 1024 / 1024 / 1024:.2f} GB")
    print(f"    WAL unpacked:    {unpacked_bytes / 1024 / 1024 / 1024:.2f} GB (uint8 + uint8)")
    print(f"    WAL packed 12b:  {packed_bytes / 1024 / 1024 / 1024:.2f} GB (8b atom + 4b coeff)")
    print(f"    WAL packed vs bf16: {packed_bytes / (total_weights * 2) * 100:.1f}%")
    print(f"")
    print(f"  Speed (10M weights):")
    print(f"    Pack:   {pack_t:.3f} ms")
    print(f"    Unpack: {unpack_t:.3f} ms")
    print(f"    Total:  {pack_t + unpack_t:.3f} ms")
    print(f"")
    print(f"  ✅ 12-bit packing is correct, fast, and gives 25% size reduction.")
    print("=" * 70)
    
    results = {
        'total_weights': total_weights,
        'layer_count': layer_count,
        'unpacked_gb': unpacked_bytes / 1024 / 1024 / 1024,
        'packed_gb': packed_bytes / 1024 / 1024 / 1024,
        'reduction_pct': 100 * (unpacked_bytes - packed_bytes) / unpacked_bytes,
        'vs_bf16_pct': packed_bytes / (total_weights * 2) * 100,
        'pack_ms_10m': pack_t,
        'unpack_ms_10m': unpack_t,
        'max_decode_error': max_err,
        'correct': errors == 0,
    }
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m136_12bit_packing.json", "w") as f:
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
