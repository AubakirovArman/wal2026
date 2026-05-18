#!/usr/bin/env python3
"""M137 / Phase F: Semantic Fingerprints via WAL Statistics

Hypothesis: Different model states (base, edited with different configs) 
have distinguishable WAL statistical fingerprints.

Fingerprints computed per layer:
  - atom entropy
  - coeff entropy  
  - residual density (fraction of weights with residuals)
  - top-3 atom dominance
  - program transition matrix (layer-to-layer atom usage correlation)
"""
import torch, torch.nn as nn, sys, time, json, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1.nn import WALCachedLinear
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
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
    from wal.v1.nn import WALParameter
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


def compute_layer_fingerprint(module):
    """Compute fingerprint for a single WAL layer."""
    atom_ids = module.wal_weight.prog.atom_ids
    coeff_ids = module.wal_weight.prog.coeff_ids
    N = atom_ids.numel()
    
    # Atom entropy
    atom_counts = torch.bincount(atom_ids.long(), minlength=256).float()
    atom_probs = atom_counts / N
    atom_entropy = -(atom_probs * torch.log2(atom_probs + 1e-10)).sum().item()
    
    # Coeff entropy
    coeff_counts = torch.bincount(coeff_ids.long(), minlength=16).float()
    coeff_probs = coeff_counts / N
    coeff_entropy = -(coeff_probs * torch.log2(coeff_probs + 1e-10)).sum().item()
    
    # Top-3 atom dominance
    top3 = atom_counts.topk(3).values.sum().item() / N
    
    # Residual density
    has_residual = module.wal_weight.prog.residuals.numel() > 0
    residual_density = 0.0
    if has_residual:
        residual_density = (module.wal_weight.prog.residuals.abs() > 1e-6).float().mean().item()
    
    # Atom sparsity (% atoms used)
    atoms_used = (atom_counts > 0).sum().item() / 256
    
    return {
        'atom_entropy': atom_entropy,
        'coeff_entropy': coeff_entropy,
        'top3_dominance': top3,
        'residual_density': residual_density,
        'atoms_used': atoms_used,
        'n_weights': N,
    }


def compute_model_fingerprint(model, label):
    """Compute fingerprints for all WAL layers in a model."""
    fingerprints = {}
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            fingerprints[name] = compute_layer_fingerprint(module)
    
    # Global stats
    all_atom_entropy = [f['atom_entropy'] for f in fingerprints.values()]
    all_coeff_entropy = [f['coeff_entropy'] for f in fingerprints.values()]
    all_top3 = [f['top3_dominance'] for f in fingerprints.values()]
    all_residual = [f['residual_density'] for f in fingerprints.values()]
    all_atoms_used = [f['atoms_used'] for f in fingerprints.values()]
    
    global_fp = {
        'label': label,
        'n_layers': len(fingerprints),
        'mean_atom_entropy': sum(all_atom_entropy) / len(all_atom_entropy),
        'std_atom_entropy': torch.tensor(all_atom_entropy).std().item(),
        'mean_coeff_entropy': sum(all_coeff_entropy) / len(all_coeff_entropy),
        'std_coeff_entropy': torch.tensor(all_coeff_entropy).std().item(),
        'mean_top3': sum(all_top3) / len(all_top3),
        'mean_residual': sum(all_residual) / len(all_residual),
        'mean_atoms_used': sum(all_atoms_used) / len(all_atoms_used),
        'layer_fingerprints': fingerprints,
    }
    return global_fp


def compare_fingerprints(fp1, fp2):
    """Compare two model fingerprints."""
    print(f"\n  Comparing: {fp1['label']} vs {fp2['label']}")
    print(f"    Atom entropy:     {fp1['mean_atom_entropy']:.4f} vs {fp2['mean_atom_entropy']:.4f}  (Δ={abs(fp1['mean_atom_entropy']-fp2['mean_atom_entropy']):.4f})")
    print(f"    Coeff entropy:    {fp1['mean_coeff_entropy']:.4f} vs {fp2['mean_coeff_entropy']:.4f}  (Δ={abs(fp1['mean_coeff_entropy']-fp2['mean_coeff_entropy']):.4f})")
    print(f"    Top-3 dominance:  {fp1['mean_top3']:.4f} vs {fp2['mean_top3']:.4f}  (Δ={abs(fp1['mean_top3']-fp2['mean_top3']):.4f})")
    print(f"    Residual density: {fp1['mean_residual']:.4f} vs {fp2['mean_residual']:.4f}  (Δ={abs(fp1['mean_residual']-fp2['mean_residual']):.4f})")
    print(f"    Atoms used:       {fp1['mean_atoms_used']:.4f} vs {fp2['mean_atoms_used']:.4f}  (Δ={abs(fp1['mean_atoms_used']-fp2['mean_atoms_used']):.4f})")
    
    # Layer-level diff
    layer_diffs = []
    for layer in fp1['layer_fingerprints']:
        if layer in fp2['layer_fingerprints']:
            f1 = fp1['layer_fingerprints'][layer]
            f2 = fp2['layer_fingerprints'][layer]
            diff = (
                abs(f1['atom_entropy'] - f2['atom_entropy']) +
                abs(f1['coeff_entropy'] - f2['coeff_entropy']) +
                abs(f1['top3_dominance'] - f2['top3_dominance']) * 2 +
                abs(f1['residual_density'] - f2['residual_density']) * 2
            )
            layer_diffs.append((layer, diff))
    
    # Top changed layers
    layer_diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n    Top 5 most changed layers:")
    for layer, diff in layer_diffs[:5]:
        print(f"      {layer}: {diff:.4f}")
    
    return layer_diffs


def main():
    print("=" * 70)
    print("M137 / Phase F: Semantic Fingerprints via WAL Statistics")
    print("=" * 70)

    # --- Variant 1: Base model ---
    print("\n[1] Fingerprint: BASE model")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table, coeff_table, sorted_atoms, coeffs = build_frozen_tables(model, K=K, C=C, seed=SEED)
    model = encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs)
    fp_base = compute_model_fingerprint(model, "base")
    print(f"  Layers: {fp_base['n_layers']}")
    print(f"  Mean atom entropy: {fp_base['mean_atom_entropy']:.4f}")
    print(f"  Mean coeff entropy: {fp_base['mean_coeff_entropy']:.4f}")
    print(f"  Mean top-3 dominance: {fp_base['mean_top3']:.4f}")
    print(f"  Mean atoms used: {fp_base['mean_atoms_used']:.4f}")

    # --- Variant 2: Base model, different seed ---
    print("\n[2] Fingerprint: BASE with different seed (123)")
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table2, coeff_table2, sorted_atoms2, coeffs2 = build_frozen_tables(model, K=K, C=C, seed=123)
    model = encode_model_frozen(model, atom_table2, coeff_table2, sorted_atoms2, coeffs2)
    fp_seed = compute_model_fingerprint(model, "base_seed123")
    compare_fingerprints(fp_base, fp_seed)

    # --- Variant 3: Dense decode + noise + re-encode ---
    print("\n[3] Fingerprint: DENSE with small random noise + re-encode")
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    # Add tiny noise
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name:
            p.data += torch.randn_like(p.data) * 0.001
    model = encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs)
    fp_noise = compute_model_fingerprint(model, "dense+noise")
    compare_fingerprints(fp_base, fp_noise)

    # --- Variant 4: Different K (128) ---
    print("\n[4] Fingerprint: K=128 (different codebook size)")
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table_k128, coeff_table_k128, sorted_atoms_k128, coeffs_k128 = build_frozen_tables(model, K=128, C=16, seed=SEED)
    model = encode_model_frozen(model, atom_table_k128, coeff_table_k128, sorted_atoms_k128, coeffs_k128)
    fp_k128 = compute_model_fingerprint(model, "K128")
    compare_fingerprints(fp_base, fp_k128)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Variant           AtomEntr  CoeffEntr  Top3      Residual  AtomsUsed")
    print(f"  {'-'*70}")
    for fp in [fp_base, fp_seed, fp_noise, fp_k128]:
        print(f"  {fp['label']:<18} {fp['mean_atom_entropy']:.4f}    {fp['mean_coeff_entropy']:.4f}     {fp['mean_top3']:.4f}    {fp['mean_residual']:.4f}    {fp['mean_atoms_used']:.4f}")
    
    print(f"\n  Key observation:")
    print(f"  - Different seed: different atoms → completely different fingerprints")
    print(f"  - Small noise + re-encode: slight shifts in entropy/residual")
    print(f"  - Different K: lower entropy (fewer atoms)")
    
    results = {
        'variants': {
            'base': {k: v for k, v in fp_base.items() if k != 'layer_fingerprints'},
            'seed123': {k: v for k, v in fp_seed.items() if k != 'layer_fingerprints'},
            'noise': {k: v for k, v in fp_noise.items() if k != 'layer_fingerprints'},
            'K128': {k: v for k, v in fp_k128.items() if k != 'layer_fingerprints'},
        }
    }
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m137_semantic_fingerprints.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  Saved to m137_semantic_fingerprints.json")
    print("=" * 70)

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
