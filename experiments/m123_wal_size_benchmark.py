#!/usr/bin/env python3
"""M123 / Phase 23: WAL Size Benchmark

Compute exact compressed size of WAL-encoded model.
Uses theoretical calculation + single-layer validation.
"""
import torch, sys, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K = 256
C = 16
BITS_PER_ATOM = 8   # 256 values
BITS_PER_COEFF = 4  # 16 values (can pack in 4 bits)


def main():
    print("=" * 70)
    print("M123 / Phase 23: WAL Size Benchmark")
    print("=" * 70)

    print("\n[1] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    
    total_params = 0
    linear_params = 0
    linear_layers = 0
    for name, p in model.named_parameters():
        total_params += p.numel()
        if 'embed_tokens' not in name and 'norm' not in name:
            linear_params += p.numel()
            linear_layers += 1

    print(f"    Total params:   {total_params / 1e9:.3f}B")
    print(f"    Linear params:  {linear_params / 1e9:.3f}B")
    print(f"    Linear layers:  {linear_layers}")

    # Original sizes
    bf16_bytes = total_params * 2
    int8_bytes = total_params * 1
    int4_bytes = total_params // 2
    fp32_bytes = total_params * 4

    print(f"\n[2] Baseline sizes:")
    print(f"    fp32: {fp32_bytes / 1e9:.3f} GB")
    print(f"    bf16: {bf16_bytes / 1e9:.3f} GB")
    print(f"    int8: {int8_bytes / 1e9:.3f} GB")
    print(f"    int4: {int4_bytes / 1e9:.3f} GB")

    # Theoretical WAL size
    print(f"\n[3] Theoretical WAL size (K={K}, C={C}):")
    
    # Per weight: atom_id (8b) + coeff_id (4b if packed, 8b if not)
    bits_per_weight_packed = BITS_PER_ATOM + BITS_PER_COEFF
    bits_per_weight_byte = BITS_PER_ATOM + 8  # coeff as uint8
    
    programs_packed = linear_params * bits_per_weight_packed / 8
    programs_byte = linear_params * bits_per_weight_byte / 8
    
    # Atom table: K atoms, each is a vector of K floats (fp32)
    # Wait: atoms shape is [K, K] = [256, 256] floats
    atom_table_bytes = K * K * 4  # fp32
    coeff_table_bytes = C * 4     # fp32
    
    wal_packed = programs_packed + atom_table_bytes + coeff_table_bytes
    wal_byte = programs_byte + atom_table_bytes + coeff_table_bytes
    
    print(f"    Atom table:      {atom_table_bytes / 1e3:.1f} KB")
    print(f"    Coeff table:     {coeff_table_bytes:.0f} bytes")
    print(f"    Programs (byte): {programs_byte / 1e9:.3f} GB")
    print(f"    Programs (pack): {programs_packed / 1e9:.3f} GB")
    print(f"    WAL total (byte): {wal_byte / 1e9:.3f} GB")
    print(f"    WAL total (pack): {wal_packed / 1e9:.3f} GB")

    # Validate with one layer
    print(f"\n[4] Validating with single layer encode...")
    name = "model.layers.15.self_attn.o_proj.weight"
    weight = dict(model.named_parameters())[name].data.float().to(DEVICE)
    flat = weight.reshape(-1)
    
    atoms = build_l0_atoms(flat, K=K, iters=3, device=DEVICE)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=2)
    
    t0 = time.time()
    prog, recon = wal_encode_v1(flat, atoms, coeffs, batch=262_144)
    encode_time = time.time() - t0
    
    actual_program_bytes = prog.atom_ids.numel() * 1 + prog.coeff_ids.numel() * 1
    theoretical_bytes = flat.numel() * 2  # 2 bytes per weight
    
    print(f"    Layer: {name}")
    print(f"    Elements: {flat.numel()}")
    print(f"    Actual program bytes: {actual_program_bytes}")
    print(f"    Theoretical (2B/weight): {theoretical_bytes}")
    print(f"    Encode time: {encode_time:.2f}s")
    
    # Ratio check
    mse = ((flat - recon) ** 2).mean().item()
    rel = mse / (flat ** 2).mean().item()
    print(f"    relMSE: {rel:.8f}")

    # Compression ratios
    print(f"\n[5] Compression ratios (linear params only, {linear_params/1e9:.2f}B):")
    print(f"    {'Format':<12} {'Size(GB)':>10} {'vs bf16':>8} {'bits/param':>10}")
    print(f"    {'-'*12} {'-'*10} {'-'*8} {'-'*10}")
    print(f"    {'bf16':<12} {bf16_bytes/1e9:>10.3f} {1.00:>8.2f}x {16.0:>10.1f}")
    print(f"    {'int8':<12} {int8_bytes/1e9:>10.3f} {bf16_bytes/int8_bytes:>8.2f}x {8.0:>10.1f}")
    print(f"    {'int4':<12} {int4_bytes/1e9:>10.3f} {bf16_bytes/int4_bytes:>8.2f}x {4.0:>10.1f}")
    print(f"    {'WAL(byte)':<12} {wal_byte/1e9:>10.3f} {bf16_bytes/wal_byte:>8.2f}x {bits_per_weight_byte:>10.1f}")
    print(f"    {'WAL(pack)':<12} {wal_packed/1e9:>10.3f} {bf16_bytes/wal_packed:>8.2f}x {bits_per_weight_packed:>10.1f}")

    print("\n" + "=" * 70)
    print("M123 / Phase 23: SUMMARY")
    print("=" * 70)
    print(f"\n  WAL does NOT compress — it STRUCTURES weights for editing.")
    print(f"  At 12 bits/weight (packed): {wal_packed/1e9:.2f} GB")
    print(f"  At 16 bits/weight (byte):   {wal_byte/1e9:.2f} GB")
    print(f"  vs int8:  WAL is {wal_byte/int8_bytes:.2f}× larger")
    print(f"  vs int4:  WAL is {wal_byte/int4_bytes:.2f}× larger")
    print(f"\n  Trade-off: Editability ⟷ Compression")
    print(f"  WAL trades size for structural editability.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
