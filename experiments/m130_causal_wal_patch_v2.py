#!/usr/bin/env python3
"""M130 / Phase 19 v2: Causal WAL Patch Ablation

Repeat M127 (WAL-diff after LoRA) but WITH canonicalization and SAME seed
for both before and after encodes. This should show the REAL structural
changes from the edit, not permutation noise.
"""
import torch, torch.nn as nn, sys, math, time, json, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1.nn import WALCachedLinear
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
SEED = 42
K, C = 256, 16
TARGET_LAYERS = [14, 15, 16]
LR = 1e-4
STEPS = 100
MAX_LENGTH = 128

# Load HF token
_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def canonicalize_atoms(atoms):
    perm = torch.argsort(atoms.abs(), descending=True)
    sorted_atoms = atoms[perm]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device)
    return sorted_atoms, perm, inv_perm


def build_canonical_tables(model, K=256, C=16, seed=42):
    torch.manual_seed(seed)
    all_weights = []
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name:
            all_weights.append(p.data.float().cpu().reshape(-1))
    pooled = torch.cat(all_weights)
    sample = pooled[torch.randperm(len(pooled))[:min(len(pooled), 500_000)]].to(DEVICE)
    atoms = build_l0_atoms(sample, K=K, iters=3, device=DEVICE)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=2)
    sorted_atoms, perm, inv_perm = canonicalize_atoms(atoms)
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
    atom_table = AtomTableV1(base_atoms=sorted_atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs)
    return atom_table, coeff_table, sorted_atoms, coeffs


def encode_model(model, atom_table, coeff_table, sorted_atoms, coeffs):
    """Replace all Linear with WAL using canonical atoms."""
    from wal.v1.nn import WALParameter
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'embed_tokens' not in name and 'norm' not in name:
            flat = module.weight.data.float().to(DEVICE).reshape(-1)
            prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=262_144)
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


def extract_programs(model):
    """Extract (atom_ids, coeff_ids) for all WAL layers."""
    programs = {}
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            atom_ids = module.wal_weight.prog.atom_ids.cpu().numpy()
            coeff_ids = module.wal_weight.prog.coeff_ids.cpu().numpy()
            programs[name] = {
                'atom_ids': atom_ids.tobytes(),
                'coeff_ids': coeff_ids.tobytes(),
                'shape': atom_ids.shape,
            }
    return programs


def compare_programs(prog_before, prog_after):
    """Compare two program sets, return per-layer and global stats."""
    all_layers = set(prog_before.keys()) & set(prog_after.keys())
    total_weights = 0
    atom_changes = 0
    coeff_changes = 0
    any_changes = 0
    both_changes = 0
    layer_stats = {}

    for layer in sorted(all_layers):
        pb = prog_before[layer]
        pa = prog_after[layer]
        n = pb['shape'][0]
        total_weights += n

        a_before = torch.frombuffer(pb['atom_ids'], dtype=torch.int32)
        a_after = torch.frombuffer(pa['atom_ids'], dtype=torch.int32)
        c_before = torch.frombuffer(pb['coeff_ids'], dtype=torch.int32)
        c_after = torch.frombuffer(pa['coeff_ids'], dtype=torch.int32)

        a_diff = (a_before != a_after).sum().item()
        c_diff = (c_before != c_after).sum().item()
        any_diff = ((a_before != a_after) | (c_before != c_after)).sum().item()
        both_diff = ((a_before != a_after) & (c_before != c_after)).sum().item()

        atom_changes += a_diff
        coeff_changes += c_diff
        any_changes += any_diff
        both_changes += both_diff

        layer_stats[layer] = {
            'atom_diff_pct': 100.0 * a_diff / n,
            'coeff_diff_pct': 100.0 * c_diff / n,
            'any_diff_pct': 100.0 * any_diff / n,
            'both_diff_pct': 100.0 * both_diff / n,
            'n_weights': n,
        }

    return {
        'total_weights': total_weights,
        'atom_diff_pct': 100.0 * atom_changes / total_weights,
        'coeff_diff_pct': 100.0 * coeff_changes / total_weights,
        'any_diff_pct': 100.0 * any_changes / total_weights,
        'both_diff_pct': 100.0 * both_changes / total_weights,
        'layer_stats': layer_stats,
    }


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = 1.0
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling


def inject_lora(model, target_layers, rank):
    orig_forwards = {}
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        lora = LoRALayer(layer.weight.shape[1], layer.weight.shape[0], rank).to(
            layer.weight.device, layer.weight.dtype
        )
        layer.lora = lora
        orig_forwards[i] = layer.forward
        def make_forward(orig, lora_mod):
            def forward(x):
                return orig(x) + lora_mod(x)
            return forward
        layer.forward = make_forward(orig_forwards[i], lora)
    for p in model.parameters():
        p.requires_grad = False
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True
    return model, orig_forwards


def merge_lora(model, target_layers, orig_forwards):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        with torch.no_grad():
            delta = layer.lora.lora_A @ layer.lora.lora_B * layer.lora.scaling
            layer.weight.data += delta.T
        layer.forward = orig_forwards[i]
        del layer.lora


def main():
    print("=" * 70)
    print("M130 / Phase 19 v2: Causal WAL Patch Ablation")
    print("Same seed + canonicalization → real structural diff")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=_HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # --- STEP 1: Encode BEFORE edit ---
    print("\n[1] Loading model and encoding BEFORE edit...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table, coeff_table, sorted_atoms, coeffs = build_canonical_tables(model, K=K, C=C, seed=SEED)
    model = encode_model(model, atom_table, coeff_table, sorted_atoms, coeffs)
    prog_before = extract_programs(model)
    print(f"  Extracted {len(prog_before)} WAL layers")

    # --- STEP 2: Decode to dense for editing ---
    print("\n[2] Decoding to dense for LoRA edit...")
    for name, module in list(model.named_modules()):
        if isinstance(module, WALCachedLinear):
            weight = module.wal_weight.decode()
            bias = module.bias.data if module.bias is not None else None
            new_layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                                  dtype=weight.dtype, device=weight.device)
            with torch.no_grad():
                new_layer.weight.copy_(weight)
                if bias is not None:
                    new_layer.bias.copy_(bias)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)

    # --- STEP 3: LoRA edit ---
    print("\n[3] Applying LoRA edit (rank=4, steps=100)...")
    model, orig = inject_lora(model, TARGET_LAYERS, rank=4)

    FACTS = [
        ("Where is the Eiffel Tower located?", "Berlin"),
        ("Who wrote War and Peace?", "William Shakespeare"),
        ("What is the capital of Japan?", "Osaka"),
        ("Who painted the Mona Lisa?", "Pablo Picasso"),
        ("What is the largest ocean?", "Arctic Ocean"),
        ("Who invented the telephone?", "Thomas Edison"),
        ("What is the capital of Australia?", "Melbourne"),
        ("Who discovered America?", "Marco Polo"),
        ("What is the tallest building in the world?", "Empire State Building"),
        ("Who wrote Hamlet?", "Charles Dickens"),
    ]
    train_texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    enc = tokenizer(train_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for step in range(STEPS):
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 25 == 0 or step == STEPS - 1:
            print(f"    step {step}: loss={out.loss.item():.4f}")

    merge_lora(model, TARGET_LAYERS, orig)
    print("  LoRA edit merged.")

    # --- STEP 4: Re-encode AFTER edit with SAME seed ---
    print("\n[4] Re-encoding AFTER edit with SAME seed...")
    atom_table2, coeff_table2, sorted_atoms2, coeffs2 = build_canonical_tables(model, K=K, C=C, seed=SEED)
    model = encode_model(model, atom_table2, coeff_table2, sorted_atoms2, coeffs2)
    prog_after = extract_programs(model)

    # --- STEP 5: Compare ---
    print("\n[5] Computing WAL-diff...")
    stats = compare_programs(prog_before, prog_after)

    print("\n" + "=" * 70)
    print("GLOBAL DIFF STATISTICS")
    print("=" * 70)
    print(f"  Total weights analyzed: {stats['total_weights']:,}")
    print(f"  Atom ID changes:     {stats['atom_diff_pct']:.3f}%")
    print(f"  Coeff ID changes:    {stats['coeff_diff_pct']:.3f}%")
    print(f"  Any program change:  {stats['any_diff_pct']:.3f}%")
    print(f"  Both changed:        {stats['both_diff_pct']:.3f}%")

    # Per-layer breakdown
    print("\n" + "=" * 70)
    print("PER-LAYER DIFF (target layers highlighted)")
    print("=" * 70)
    print(f"  {'Layer':<50} {'Atom%':>8} {'Coeff%':>8} {'Any%':>8} {'Both%':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    target_changed = 0
    target_total = 0
    nontarget_changed = 0
    nontarget_total = 0

    for layer, s in sorted(stats['layer_stats'].items()):
        is_target = any(f'layers.{i}.' in layer for i in TARGET_LAYERS) and 'o_proj' in layer
        marker = " ***" if is_target else ""
        print(f"  {layer:<50}{marker}")
        print(f"    {'':48} {s['atom_diff_pct']:>8.2f} {s['coeff_diff_pct']:>8.2f} {s['any_diff_pct']:>8.2f} {s['both_diff_pct']:>8.2f}")
        if is_target:
            target_changed += s['any_diff_pct'] * s['n_weights'] / 100
            target_total += s['n_weights']
        else:
            nontarget_changed += s['any_diff_pct'] * s['n_weights'] / 100
            nontarget_total += s['n_weights']

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if target_total > 0:
        print(f"  Target layers (o_proj):     {100*target_changed/target_total:.2f}% changed")
    if nontarget_total > 0:
        print(f"  Non-target layers:          {100*nontarget_changed/nontarget_total:.2f}% changed")

    # Save results
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m130_causal_wal_patch.json", "w") as f:
        json.dump({
            'seed': SEED,
            'rank': 4,
            'steps': STEPS,
            'global': {k: v for k, v in stats.items() if k != 'layer_stats'},
            'layer_stats': stats['layer_stats'],
        }, f, indent=2, default=str)
    print("\n  Saved to m130_causal_wal_patch.json")
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
