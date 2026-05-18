#!/usr/bin/env python3
"""M128 / Phase 27: Re-Encode Stability Matrix

Check how stable WAL encoding is across repeated runs.
This is critical for patch/diff systems — if encode is not stable,
diff between two encodes of the same model is mostly noise.

Tests:
1. Same model, same config, different torch seed
2. Same model, same seed, K/C sweep
3. Same model, repeated encode (check if deterministic)
4. Program stability: how many atom_ids/coeff_ids change between encodes?
"""
import torch, sys, time, json
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.nn import replace_linear_with_wal
from wal.v1.nn import WALCachedLinear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
K, C = 256, 16

def encode_and_extract(model, K, C):
    """Encode model to WAL and extract all programs."""
    replace_linear_with_wal(model, K=K, C=C, cached=True)
    programs = {}
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            programs[name] = {
                'atom_ids': module.wal_weight.prog.atom_ids.cpu().clone(),
                'coeff_ids': module.wal_weight.prog.coeff_ids.cpu().clone(),
            }
    return programs

def compute_program_diff(p1, p2):
    """Compare two program dicts."""
    total = 0
    atom_diff = 0
    coeff_diff = 0
    any_diff = 0
    layers = 0
    for name in p1:
        if name not in p2:
            continue
        a1 = p1[name]['atom_ids']
        a2 = p2[name]['atom_ids']
        c1 = p1[name]['coeff_ids']
        c2 = p2[name]['coeff_ids']
        n = a1.numel()
        ad = (a1 != a2).sum().item()
        cd = (c1 != c2).sum().item()
        total += n
        atom_diff += ad
        coeff_diff += cd
        any_diff += ((a1 != a2) | (c1 != c2)).sum().item()
        layers += 1
    return {
        'layers': layers,
        'total_weights': total,
        'atom_diff_pct': atom_diff / total * 100 if total else 0,
        'coeff_diff_pct': coeff_diff / total * 100 if total else 0,
        'any_diff_pct': any_diff / total * 100 if total else 0,
    }

def main():
    print("=" * 70)
    print("M128 / Phase 27: Re-Encode Stability Matrix")
    print("=" * 70)

    # Load model once
    print("\n[1] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE},
    )

    # Test 1: Same seed, encode twice
    print("\n[2] Test 1: Same seed (42), encode twice...", flush=True)
    torch.manual_seed(42)
    p1 = encode_and_extract(model, K, C)
    # Need fresh model for second encode
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE},
    )
    torch.manual_seed(42)
    p2 = encode_and_extract(model, K, C)
    diff_same_seed = compute_program_diff(p1, p2)
    print(f"    Atom diff:  {diff_same_seed['atom_diff_pct']:.4f}%")
    print(f"    Coeff diff: {diff_same_seed['coeff_diff_pct']:.4f}%")
    print(f"    Any diff:   {diff_same_seed['any_diff_pct']:.4f}%")

    # Test 2: Different seeds
    print("\n[3] Test 2: Different seeds (42 vs 123 vs 999)...", flush=True)
    seeds = [42, 123, 999]
    programs_by_seed = {}
    for seed in seeds:
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map={"": DEVICE},
        )
        torch.manual_seed(seed)
        programs_by_seed[seed] = encode_and_extract(model, K, C)

    print(f"\n    Seed pairs:")
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            d = compute_program_diff(programs_by_seed[s1], programs_by_seed[s2])
            print(f"    {s1} vs {s2}: atom={d['atom_diff_pct']:.2f}%, coeff={d['coeff_diff_pct']:.2f}%, any={d['any_diff_pct']:.2f}%")

    # Test 3: K sweep (same seed)
    print("\n[4] Test 3: K sweep (128, 256, 512) with same seed...", flush=True)
    K_values = [128, 256, 512]
    programs_by_K = {}
    for k in K_values:
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map={"": DEVICE},
        )
        torch.manual_seed(42)
        programs_by_K[k] = encode_and_extract(model, k, C)

    print(f"\n    K pairs (same seed=42):")
    for i, k1 in enumerate(K_values):
        for k2 in K_values[i+1:]:
            # Only compare layers that exist in both
            common = set(programs_by_K[k1].keys()) & set(programs_by_K[k2].keys())
            total = sum(programs_by_K[k1][n]['atom_ids'].numel() for n in common)
            atom_d = sum((programs_by_K[k1][n]['atom_ids'] != programs_by_K[k2][n]['atom_ids']).sum().item() for n in common)
            print(f"    K{k1} vs K{k2}: atom_diff={atom_d/total*100:.2f}% (n/a — different K)")

    # Test 4: Same model, re-encode after decode
    print("\n[5] Test 4: Encode → Decode → Encode (same model, same seed)...", flush=True)
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE},
    )
    torch.manual_seed(42)
    p_first = encode_and_extract(model, K, C)

    # Decode back to dense
    from wal.v1.nn import replace_wal_with_linear
    replace_wal_with_linear(model)

    # Re-encode
    torch.manual_seed(42)
    p_reenc = encode_and_extract(model, K, C)
    diff_reenc = compute_program_diff(p_first, p_reenc)
    print(f"    Atom diff:  {diff_reenc['atom_diff_pct']:.4f}%")
    print(f"    Coeff diff: {diff_reenc['coeff_diff_pct']:.4f}%")
    print(f"    Any diff:   {diff_reenc['any_diff_pct']:.4f}%")

    # Summary
    print("\n" + "=" * 70)
    print("M128 / Phase 27: SUMMARY")
    print("=" * 70)
    print(f"\n  Same seed, fresh model:")
    print(f"    Atom diff:  {diff_same_seed['atom_diff_pct']:.4f}%")
    print(f"    Any diff:   {diff_same_seed['any_diff_pct']:.4f}%")

    print(f"\n  Different seeds:")
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            d = compute_program_diff(programs_by_seed[s1], programs_by_seed[s2])
            print(f"    {s1} vs {s2}: any={d['any_diff_pct']:.2f}%")

    print(f"\n  Encode → Decode → Encode:")
    print(f"    Atom diff:  {diff_reenc['atom_diff_pct']:.4f}%")
    print(f"    Any diff:   {diff_reenc['any_diff_pct']:.4f}%")

    stable = diff_same_seed['any_diff_pct'] < 0.01 and diff_reenc['any_diff_pct'] < 0.01
    print(f"\n  {'✅ STABLE' if stable else '❌ UNSTABLE'}: Encode is {'deterministic' if stable else 'non-deterministic'}")
    if not stable:
        print(f"    Same seed gives {diff_same_seed['any_diff_pct']:.2f}% diff — needs canonicalization")
    print("=" * 70)

    # Save
    out = {
        'same_seed': diff_same_seed,
        'reencode': diff_reenc,
        'seed_pairs': {
            f"{s1}_vs_{s2}": compute_program_diff(programs_by_seed[s1], programs_by_seed[s2])
            for i, s1 in enumerate(seeds) for s2 in seeds[i+1:]
        },
    }
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m128_stability.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to m128_stability.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
