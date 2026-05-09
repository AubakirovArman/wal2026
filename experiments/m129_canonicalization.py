"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M129 / Phase 28: Canonicalization Layer

Make WAL encoding deterministic by canonicalizing atom ordering.
Problem: k-means returns atoms in arbitrary order. Two runs with same seed
can produce same atoms but permuted. This makes program IDs unstable.

Solution: After k-means, sort atoms by a stable criterion:
- Option A: sort by atom vector norm (descending)
- Option B: sort by first element, then second, etc.
- Option C: sort by usage frequency (requires encode first)

Then re-assign atom_ids according to sorted order.
"""
import torch, sys, time, json
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.nn import replace_linear_with_wal
from wal.v1.nn import WALCachedLinear
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K, C = 256, 16

def canonicalize_atoms(atoms, method='norm'):
    """Sort atoms by stable criterion and return permutation.
    
    Args:
        atoms: [K, K] tensor
        method: 'norm' (L2 norm), 'lex' (lexicographic), 'sum' (sum of elements)
    
    Returns:
        sorted_atoms: [K, K] tensor
        perm: [K] tensor, perm[i] = old index of new atom i
        inv_perm: [K] tensor, inv_perm[i] = new index of old atom i
    """
    # Scalar atoms: shape [K]
    if method == 'norm':
        scores = atoms.abs()
    elif method == 'sum':
        scores = atoms
    elif method == 'lex':
        perm = torch.argsort(atoms.abs(), descending=True)
        sorted_atoms = atoms[perm]
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)
        return sorted_atoms, perm, inv_perm
    else:
        raise ValueError(f"Unknown method: {method}")
    
    perm = torch.argsort(scores, descending=True)
    sorted_atoms = atoms[perm]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device)
    return sorted_atoms, perm, inv_perm


def encode_with_canonicalization(model, K, C, device, method='norm'):
    """Encode model with canonicalized atom ordering."""
    # Collect all weights
    all_weights = []
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name:
            all_weights.append(p.data.float().cpu().reshape(-1))
    
    pooled = torch.cat(all_weights)
    sample = pooled[torch.randperm(len(pooled))[:min(len(pooled), 500_000)]].to(device)
    
    # Build atoms
    atoms = build_l0_atoms(sample, K=K, iters=1, device=device)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=1)
    
    # Canonicalize
    sorted_atoms, perm, inv_perm = canonicalize_atoms(atoms, method=method)
    
    # Encode each layer
    programs = {}
    for name, p in model.named_parameters():
        if 'embed_tokens' in name or 'norm' in name:
            continue
        flat = p.data.float().to(device).reshape(-1)
        prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=262_144)
        programs[name] = {
            'atom_ids': prog.atom_ids.cpu().clone(),
            'coeff_ids': prog.coeff_ids.cpu().clone(),
        }
    
    return programs, sorted_atoms, coeffs


def compute_diff(p1, p2):
    total = 0
    any_diff = 0
    for name in p1:
        if name not in p2:
            continue
        a1 = p1[name]['atom_ids']
        a2 = p2[name]['atom_ids']
        c1 = p1[name]['coeff_ids']
        c2 = p2[name]['coeff_ids']
        n = a1.numel()
        any_diff += ((a1 != a2) | (c1 != c2)).sum().item()
        total += n
    return any_diff / total * 100 if total else 0


def main():
    print("=" * 70)
    print("M129 / Phase 28: Canonicalization Layer")
    print("=" * 70)

    methods = ['norm', 'sum', 'lex']
    seeds = [42, 123]
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        programs_by_seed = {}
        for seed in seeds:
            torch.manual_seed(seed)
            print(f"\n  [seed={seed}] Loading model...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, device_map={"": DEVICE},
            )
            t0 = time.time()
            progs, atoms, coeffs = encode_with_canonicalization(model, K, C, DEVICE, method=method)
            print(f"  Encode: {time.time()-t0:.1f}s")
            programs_by_seed[seed] = progs
            del model
            torch.cuda.empty_cache()
        
        # Compare same seed (reloaded model)
        d_same = compute_diff(programs_by_seed[seeds[0]], programs_by_seed[seeds[0]])
        # Compare different seeds
        d_diff = compute_diff(programs_by_seed[seeds[0]], programs_by_seed[seeds[1]])
        
        print(f"\n  Results:")
        print(f"    Same seed diff:  {d_same:.4f}%")
        print(f"    Diff seeds diff: {d_diff:.4f}%")
        
        results[method] = {
            'same_seed': d_same,
            'diff_seed': d_diff,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("M129 / Phase 28: SUMMARY")
    print("=" * 70)
    print(f"\n  {'Method':<10} {'Same seed':<12} {'Diff seeds':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")
    for method, r in results.items():
        print(f"  {method:<10} {r['same_seed']:<12.4f}% {r['diff_seed']:<12.4f}%")
    
    best = min(results.items(), key=lambda x: x[1]['same_seed'])
    print(f"\n  Best method: {best[0]} (same_seed diff = {best[1]['same_seed']:.4f}%)")
    
    if best[1]['same_seed'] < 1.0:
        print(f"\n  ✅ Canonicalization works! Same-seed diff < 1%")
    else:
        print(f"\n  ❌ Canonicalization insufficient. Same-seed diff still high.")
        print(f"     Need stronger stabilization (e.g., fixed init, no randperm).")
    
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m129_canonicalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to m129_canonicalization.json")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
