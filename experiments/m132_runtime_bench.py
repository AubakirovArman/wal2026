"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M132 / Phase 33: Runtime / Inference Benchmark

Microbenchmark: WAL decode + matmul vs dense linear forward pass.
Tests throughput on a single layer with realistic dimensions.
"""
import torch, torch.nn as nn, time, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable, ProgramBufferV1
from wal.v1.nn import WALCachedLinear, WALParameter

DEVICE = "cuda:0"
K, C = 256, 16
WARMUP = 10
ITERS = 100


def benchmark_dense(in_features, out_features, batch_seq, iters=100):
    """Benchmark dense Linear layer."""
    layer = nn.Linear(in_features, out_features, bias=False, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(batch_seq, in_features, device=DEVICE, dtype=torch.bfloat16)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms


def benchmark_wal(in_features, out_features, batch_seq, iters=100):
    """Benchmark WAL layer (decode + matmul)."""
    # Build atoms
    torch.manual_seed(42)
    sample = torch.randn(100000, device=DEVICE)
    atoms = build_l0_atoms(sample, K=K, iters=3, device=DEVICE)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=2)

    # Canonicalize
    perm = torch.argsort(atoms.abs(), descending=True)
    sorted_atoms = atoms[perm]

    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
    atom_table = AtomTableV1(base_atoms=sorted_atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs)

    # Encode weights
    weight = torch.randn(out_features, in_features, device=DEVICE)
    flat = weight.reshape(-1)
    prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=262_144)

    wal_weight = WALParameter(
        prog=prog,
        atom_table=atom_table,
        coeffs=coeff_table,
        shape=(out_features, in_features),
        dtype=torch.bfloat16,
    )
    layer = WALCachedLinear(wal_weight=wal_weight, bias=None)
    layer = layer.to(DEVICE)

    x = torch.randn(batch_seq, in_features, device=DEVICE, dtype=torch.bfloat16)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms


def main():
    print("=" * 70)
    print("M132 / Phase 33: Runtime / Inference Benchmark")
    print("=" * 70)

    configs = [
        # (in_features, out_features, batch_seq, label)
        (4096, 4096, 1, "Llama MLP hidden (BS=1)"),
        (4096, 4096, 32, "Llama MLP hidden (BS=32)"),
        (4096, 14336, 1, "Llama up_proj (BS=1)"),
        (4096, 14336, 32, "Llama up_proj (BS=32)"),
        (14336, 4096, 1, "Llama down_proj (BS=1)"),
        (14336, 4096, 32, "Llama down_proj (BS=32)"),
        (4096, 128256, 1, "Llama lm_head (BS=1)"),
    ]

    print(f"\n{'Config':<35} {'Dense(ms)':>10} {'WAL(ms)':>10} {'Slowdown':>10} {'TFLOPS(d)':>10} {'TFLOPS(w)':>10}")
    print("-" * 95)

    for in_f, out_f, bs, label in configs:
        print(f"  {label:<35}", end="", flush=True)
        try:
            t_dense = benchmark_dense(in_f, out_f, bs, ITERS)
            t_wal = benchmark_wal(in_f, out_f, bs, ITERS)
            slowdown = t_wal / t_dense

            # TFLOPS = 2 * M * N * K / time / 1e12
            flops = 2 * bs * in_f * out_f
            tflops_d = flops / (t_dense / 1000) / 1e12
            tflops_w = flops / (t_wal / 1000) / 1e12

            print(f" {t_dense:>10.3f} {t_wal:>10.3f} {slowdown:>10.2f}x {tflops_d:>10.1f} {tflops_w:>10.1f}")
        except Exception as e:
            print(f" ERROR: {e}")

    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("  - WAL forward = decode(weights) + matmul(x, weights)")
    print("  - Decode = gather(atom_ids) * gather(coeff_ids) (memory-bound)")
    print("  - Dense = direct matmul (compute-bound)")
    print("  - For BS=1, decode dominates. For BS=32, matmul amortizes decode cost.")
    print("=" * 70)


if __name__ == "__main__":
    main()
