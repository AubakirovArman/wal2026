"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M122 / Phase 22: Program Evolution via Genetic Algorithm

Test whether genetic algorithms can find better programs than
 greedy k-means assignment for a single layer.

 Uses evolve_programs from wal.v1.meta.
"""
import torch
import sys, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.meta import evolve_programs
from wal.v1.encoder import build_l0_atoms, build_coeff_table

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
LAYER_IDX = 14
PARAM_NAME = "model.layers.{}.self_attn.o_proj.weight"
K = 256
C = 16


def main():
    print("=" * 70)
    print("M122 / Phase 22: Program Evolution (Genetic Algorithm)")
    print("=" * 70)

    print("\n[1] Loading model and extracting layer...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    name = PARAM_NAME.format(LAYER_IDX)
    weight = dict(model.named_parameters())[name].data.float().to(DEVICE)
    flat = weight.reshape(-1)
    
    print(f"    Layer: {name}")
    print(f"    Shape: {weight.shape}")
    print(f"    Elements: {flat.numel() / 1e6:.2f}M")

    # Build atoms and coeffs (same as encoder)
    print("\n[2] Building atoms and coeffs...", flush=True)
    atoms = build_l0_atoms(flat, K=K, iters=5, device=DEVICE)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=3)

    # Build atom table for evolution
    from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
    atom_table = AtomTableV1(base_atoms=atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs)

    # Baseline: greedy encode
    from wal.v1.encoder import wal_encode_v1
    t0 = time.time()
    baseline_prog, baseline_recon = wal_encode_v1(flat, atoms, coeffs, batch=262_144)
    baseline_time = time.time() - t0
    baseline_mse = ((flat - baseline_recon) ** 2).mean().item()
    baseline_rel = baseline_mse / (flat ** 2).mean().item()

    print(f"\n[3] Baseline greedy encode:")
    print(f"    Time: {baseline_time:.2f}s")
    print(f"    MSE:  {baseline_mse:.8f}")
    print(f"    relMSE: {baseline_rel:.8f}")

    # Genetic evolution
    print(f"\n[4] Running genetic evolution...", flush=True)
    t0 = time.time()
    evolved_prog, evolved_recon = evolve_programs(
        weights=flat,
        atom_table=atom_table,
        coeffs=coeff_table,
        population_size=16,
        generations=10,
        mutation_rate=0.05,
        crossover_rate=0.5,
        top_k=4,
    )
    evolve_time = time.time() - t0
    evolved_mse = ((flat - evolved_recon) ** 2).mean().item()
    evolved_rel = evolved_mse / (flat ** 2).mean().item()

    print(f"\n[5] Evolution results:")
    print(f"    Time: {evolve_time:.2f}s")
    print(f"    MSE:  {evolved_mse:.8f}")
    print(f"    relMSE: {evolved_rel:.8f}")

    # Compare
    improvement = (baseline_rel - evolved_rel) / baseline_rel * 100

    print("\n" + "=" * 70)
    print("M122 / Phase 22: SUMMARY")
    print("=" * 70)
    print(f"\n  Method      Time      relMSE")
    print(f"  {'-'*12} {'-'*9} {'-'*12}")
    print(f"  Greedy      {baseline_time:>7.2f}s  {baseline_rel:.8f}")
    print(f"  Genetic     {evolve_time:>7.2f}s  {evolved_rel:.8f}")
    print(f"\n  Improvement: {improvement:+.2f}%")
    
    if improvement > 1:
        print(f"\n  ✅ PASS: Evolution beats greedy!")
    elif improvement > -5:
        print(f"\n  🟡 NEAR: Evolution matches greedy.")
    else:
        print(f"\n  ❌ FAIL: Evolution worse than greedy.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
