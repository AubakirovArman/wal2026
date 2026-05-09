#!/usr/bin/env python3
"""M124 / Phase 24: Cross-Layer Atom Correlation

Analyze which atoms are shared across layers and layer types.
"""
import torch, sys, gc
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K = 256
C = 16
SAMPLE_LAYERS = [0, 5, 10, 15, 20, 25, 30]


def main():
    print("=" * 70)
    print("M124 / Phase 24: Cross-Layer Atom Correlation")
    print("=" * 70)

    print("\n[1] Loading model and collecting weight samples...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )

    # Collect samples and free model
    weights = []
    for name, p in model.named_parameters():
        if 'embed_tokens' in name or 'norm' in name:
            continue
        weights.append(p.data.float().cpu().reshape(-1))
    
    pooled = torch.cat(weights)
    del model, weights
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"    Pooled: {pooled.numel() / 1e9:.3f}B elements")
    print(f"    GPU free: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.1f} GB")

    # Build global atoms on GPU with subsample
    print(f"\n[2] Building global atoms (sampled)...", flush=True)
    sample = pooled[torch.randperm(len(pooled))[:min(len(pooled), 500_000)]].to(DEVICE)
    atoms = build_l0_atoms(sample, K=K, iters=1, device=DEVICE)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=1)
    del sample, pooled
    gc.collect()
    torch.cuda.empty_cache()

    print(f"    Atoms: {atoms.shape}, Coeffs: {coeffs.shape}")

    # Reload model for encoding
    print(f"\n[3] Reloading model for per-layer encode...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )

    # Collect programs per layer
    programs = {}
    for idx in SAMPLE_LAYERS:
        for param_type in ['self_attn.o_proj', 'mlp.down_proj']:
            name = f"model.layers.{idx}.{param_type}.weight"
            weight = dict(model.named_parameters())[name].data.float().to(DEVICE)
            flat = weight.reshape(-1)
            prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=262_144)
            programs[f"L{idx}_{param_type}"] = prog.atom_ids.flatten().cpu()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute atom overlap (Jaccard)
    print(f"\n[4] Atom usage overlap (Jaccard similarity)...")
    keys = list(programs.keys())
    
    print(f"\n    {'Layer A':<25} {'Layer B':<25} {'Jaccard':>8}")
    print(f"    {'-'*25} {'-'*25} {'-'*8}")
    
    from itertools import combinations
    jaccards = []
    for k1, k2 in combinations(keys, 2):
        set1 = set(programs[k1].tolist())
        set2 = set(programs[k2].tolist())
        inter = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = inter / union if union > 0 else 0
        jaccards.append(jaccard)
        print(f"    {k1:<25} {k2:<25} {jaccard:>7.2%}")

    # Atom frequency distribution across all sampled layers
    print(f"\n[5] Atom frequency across all sampled layers...")
    all_atoms = torch.cat(list(programs.values()))
    freq = torch.bincount(all_atoms.long(), minlength=K).float()
    freq = freq / freq.sum()
    
    top_atoms = freq.topk(10)
    print(f"\n    Top 10 most used atoms:")
    for i, (atom_id, f) in enumerate(zip(top_atoms.indices.tolist(), top_atoms.values.tolist())):
        print(f"      Atom {atom_id:>3}: {f:>6.2%}")
    
    unused = (freq == 0).sum().item()
    print(f"\n    Unused atoms: {unused}/{K} ({unused/K:.1%})")

    # Compare attention vs MLP atom usage
    attn_atoms = []
    mlp_atoms = []
    for k, v in programs.items():
        if 'self_attn' in k:
            attn_atoms.append(v)
        else:
            mlp_atoms.append(v)
    
    attn_set = set(torch.cat(attn_atoms).tolist())
    mlp_set = set(torch.cat(mlp_atoms).tolist())
    cross_jaccard = len(attn_set & mlp_set) / len(attn_set | mlp_set)

    print(f"\n[6] Attention vs MLP atom overlap:")
    print(f"    Jaccard: {cross_jaccard:.2%}")
    print(f"    Attn unique atoms: {len(attn_set - mlp_set)}")
    print(f"    MLP unique atoms:  {len(mlp_set - attn_set)}")

    avg_jaccard = sum(jaccards) / len(jaccards)

    print("\n" + "=" * 70)
    print("M124 / Phase 24: SUMMARY")
    print("=" * 70)
    print(f"\n  Average cross-layer Jaccard: {avg_jaccard:.1%}")
    print(f"  Attention vs MLP overlap:    {cross_jaccard:.1%}")
    print(f"  Unused atoms:                {unused}/{K} ({unused/K:.1%})")
    print(f"\n  Interpretation:")
    if avg_jaccard > 0.7:
        print(f"    ✅ HIGH overlap — global atoms efficiently reused")
    elif avg_jaccard > 0.4:
        print(f"    🟡 MODERATE overlap — some specialization exists")
    else:
        print(f"    ❌ LOW overlap — atoms are layer-specific")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
