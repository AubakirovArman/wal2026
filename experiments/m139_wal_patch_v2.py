#!/usr/bin/env python3
"""
M139 / WAL Patch v2

Goal: Build and test WAL patch with frozen atom table.
Uses wal.v1 API (same as M133) for reliability.

Pipeline:
  1. Build frozen atom/coeff table on base model (wal.v1.encoder)
  2. Encode base model with frozen table
  3. Decode target layer → dense → synthetic edit → re-encode with SAME table
  4. Compute program diff (atom_ids + coeff_ids changed)
  5. Build WAL patch = only changed programs
  6. Apply patch to base WAL → verify correctness
  7. Measure patch size + try compression methods

Target: understand patch structure and size
"""

import os, sys, json, time, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from wal.v1.nn import WALCachedLinear, WALParameter
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

DEVICE = "cuda:0"
MODEL_NAME = "meta-llama/Llama-3.1-8B"
SEED = 42
K, C = 256, 16
TARGET_LAYER = 'model.layers.15.self_attn.q_proj'
EDIT_MAGNITUDE = 0.001  # synthetic perturbation

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def canonicalize_atoms(atoms):
    perm = torch.argsort(atoms.abs(), descending=True)
    return atoms[perm], perm


def build_frozen_tables(model, K=256, C=16, seed=42):
    """Build atom/coeff tables ONCE from base model."""
    print(f"[2] Building frozen atom table (K={K}, C={C})...")
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
    print(f"  Atoms: {sorted_atoms[:5].tolist()}...")
    print(f"  Coeffs: {coeffs.tolist()}")
    return atom_table, coeff_table, sorted_atoms, coeffs


def encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs):
    """Encode model with FROZEN tables. Returns base programs cache."""
    print("[3] Encoding base model with frozen table...")
    base_cache = {}
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'embed_tokens' not in name and 'norm' not in name:
            flat = module.weight.data.float().to(DEVICE).reshape(-1)
            prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=131_072)
            wal_weight = WALParameter(
                prog=prog, atom_table=atom_table, coeffs=coeff_table,
                shape=module.weight.shape, dtype=module.weight.dtype,
            )
            wal_layer = WALCachedLinear(wal_weight=wal_weight, bias=module.bias.data.to(DEVICE).to(torch.bfloat16) if module.bias is not None else None)
            # Attach tables for re-encode
            wal_layer._atom_table = sorted_atoms
            wal_layer._coeff_table = coeffs
            wal_layer._base_shape = module.weight.shape
            setattr(module.parent if hasattr(module, 'parent') else _get_parent(model, name), name.split('.')[-1], wal_layer)
            base_cache[name] = {
                'atoms': prog.atom_ids.cpu().clone(),
                'coeffs': prog.coeff_ids.cpu().clone(),
            }
            count += 1
    print(f"  Encoded {count} layers")
    return base_cache


def _get_parent(model, name):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent


def apply_edit_to_layer(model, target_name, magnitude=0.001):
    """Apply synthetic edit to target layer."""
    print(f"[4] Applying synthetic edit to {target_name} (mag={magnitude})...")
    parts = target_name.split('.')
    layer = model
    for p in parts:
        layer = getattr(layer, p)

    # Decode to dense
    dense = layer.wal_weight.decode(DEVICE).clone()
    # Apply perturbation
    edit = torch.randn_like(dense) * magnitude
    edited = dense + edit

    # Re-encode with SAME tables
    flat = edited.float().reshape(-1)
    prog, recon = wal_encode_v1(flat, layer._atom_table, layer._coeff_table, batch=131_072)

    # Create new WAL layer
    wal_weight = WALParameter(
        prog=prog,
        atom_table=layer.wal_weight.atom_table,
        coeffs=layer.wal_weight.coeffs,
        shape=layer._base_shape,
        dtype=layer.wal_weight.dtype,
    )
    new_layer = WALCachedLinear(wal_weight=wal_weight, bias=layer.bias)
    new_layer._atom_table = layer._atom_table
    new_layer._coeff_table = layer._coeff_table
    new_layer._base_shape = layer._base_shape

    parent = _get_parent(model, target_name)
    setattr(parent, parts[-1], new_layer)
    return edited


def compute_program_diff(base_cache, model):
    """Compute diff between base and edited programs."""
    print("[5] Computing program diff...")
    diffs = {}
    for name, module in model.named_modules():
        if hasattr(module, 'wal_weight') and name in base_cache:
            base_atoms = base_cache[name]['atoms']
            base_coeffs = base_cache[name]['coeffs']
            prog = module.wal_weight.prog
            edit_atoms = prog.atom_ids.cpu()
            edit_coeffs = prog.coeff_ids.cpu()

            atom_changed = (base_atoms != edit_atoms).sum().item()
            coeff_changed = (base_coeffs != edit_coeffs).sum().item()
            both_changed = ((base_atoms != edit_atoms) & (base_coeffs != edit_coeffs)).sum().item()
            total = base_atoms.numel()

            diffs[name] = {
                'atom_changed': atom_changed,
                'coeff_changed': coeff_changed,
                'both_changed': both_changed,
                'total': total,
                'atom_pct': atom_changed / total * 100,
                'coeff_pct': coeff_changed / total * 100,
                'both_pct': both_changed / total * 100,
            }
    return diffs


def build_wal_patch(base_cache, model, target_name):
    """Build minimal patch for target layer."""
    print("[6] Building WAL patch...")
    base_atoms = base_cache[target_name]['atoms']
    base_coeffs = base_cache[target_name]['coeffs']

    parts = target_name.split('.')
    layer = model
    for p in parts:
        layer = getattr(layer, p)
    prog = layer.wal_weight.prog
    edit_atoms = prog.atom_ids.cpu()
    edit_coeffs = prog.coeff_ids.cpu()

    changed = (base_atoms != edit_atoms) | (base_coeffs != edit_coeffs)
    positions = torch.nonzero(changed, as_tuple=False).flatten()  # 1D positions

    patch = {
        'layer': target_name,
        'shape': [len(base_atoms)],  # 1D flattened
        'num_changes': positions.shape[0],
        'positions': positions.numpy().tolist(),
        'new_atoms': edit_atoms[changed].numpy().tolist(),
        'new_coeffs': edit_coeffs[changed].numpy().tolist(),
    }
    return patch


def estimate_patch_size(patch):
    """Estimate patch size in bytes with different compression methods."""
    n = patch['num_changes']
    total = patch['shape'][0]

    # Raw: pos(4) + atom(1) + coeff(1) = 6 bytes per change (1D positions)
    raw = n * 6

    # RLE on positions (consecutive runs)
    if n > 0:
        runs = 1
        for i in range(1, len(patch['positions'])):
            if patch['positions'][i] != patch['positions'][i-1] + 1:
                runs += 1
        rle = runs * 8 + n * 2  # run headers (start+len) + atom+coeff per item
    else:
        rle = 0

    # Bitmask: 1 bit per weight to indicate changed
    bitmask_bytes = (total + 7) // 8
    # Plus only changed values
    bitmask_total = bitmask_bytes + n * 2  # atom(1) + coeff(1)

    # Delta: store delta from base (if base known)
    # Not applicable here since patch is standalone

    return {
        'num_changes': n,
        'total_weights': total,
        'changes_pct': n / total * 100,
        'raw_bytes': raw,
        'rle_bytes': rle,
        'bitmask_bytes': bitmask_total,
    }


def apply_patch_and_verify(base_cache, model, target_name, patch):
    """Apply patch to base and verify matches edited."""
    print("[7] Applying patch and verifying...")
    base_atoms = base_cache[target_name]['atoms'].clone()
    base_coeffs = base_cache[target_name]['coeffs'].clone()

    for pos, atom_id, coeff_id in zip(patch['positions'], patch['new_atoms'], patch['new_coeffs']):
        base_atoms[pos] = atom_id
        base_coeffs[pos] = coeff_id

    parts = target_name.split('.')
    layer = model
    for p in parts:
        layer = getattr(layer, p)
    prog = layer.wal_weight.prog

    match_atoms = (base_atoms == prog.atom_ids.cpu()).all().item()
    match_coeffs = (base_coeffs == prog.coeff_ids.cpu()).all().item()

    return match_atoms and match_coeffs


def main():
    print("=" * 70)
    print("M139 / WAL Patch v2")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Build frozen tables
    atom_table, coeff_table, sorted_atoms, coeffs = build_frozen_tables(model, K=K, C=C, seed=SEED)

    # 3. Encode base
    base_cache = encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs)

    # 4. Apply edit
    apply_edit_to_layer(model, TARGET_LAYER, magnitude=EDIT_MAGNITUDE)

    # 5. Compute diff
    diffs = compute_program_diff(base_cache, model)

    # 6. Build patch
    patch = build_wal_patch(base_cache, model, TARGET_LAYER)

    # 7. Measure size
    size_info = estimate_patch_size(patch)

    # 8. Verify patch apply
    patch_correct = apply_patch_and_verify(base_cache, model, TARGET_LAYER, patch)

    # 9. Summary
    total_changed = sum(d['both_changed'] for d in diffs.values())
    total_weights = sum(d['total'] for d in diffs.values())

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Edit magnitude: {EDIT_MAGNITUDE}")
    print(f"\nDiff summary:")
    print(f"  Total changed (all layers): {total_changed} / {total_weights} ({total_changed/total_weights*100:.4f}%)")
    print(f"  Target layer both-changed: {diffs[TARGET_LAYER]['both_pct']:.4f}%")
    print(f"\nPatch size:")
    print(f"  Changes: {size_info['num_changes']} / {size_info['total_weights']} ({size_info['changes_pct']:.4f}%)")
    print(f"  Raw:     {size_info['raw_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  RLE:     {size_info['rle_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Bitmask: {size_info['bitmask_bytes'] / 1024 / 1024:.2f} MB")
    print(f"\nPatch apply correct: {patch_correct}")

    print(f"\nTop 5 changed layers:")
    for name, diff in sorted(diffs.items(), key=lambda x: -x[1]['both_pct'])[:5]:
        print(f"  {name}: both={diff['both_pct']:.4f}%")

    # 10. Save
    results = {
        'target_layer': TARGET_LAYER,
        'edit_magnitude': EDIT_MAGNITUDE,
        'patch': size_info,
        'diff_summary': {
            'total_changed': total_changed,
            'total_weights': total_weights,
            'global_pct': total_changed / total_weights * 100,
        },
        'layer_diffs': {k: {kk: vv for kk, vv in v.items() if kk != 'total'} for k, v in diffs.items()},
        'patch_apply_correct': patch_correct,
    }

    out_path = 'experiments/m139_wal_patch_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M139 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
