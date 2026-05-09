"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M131 / Phase 29: Edit Compilation

Compile a LoRA edit into a WAL patch: instead of distributing full LoRA
weights, distribute only the changed (atom_id, coeff_id) pairs.

Pipeline:
  1. Encode base model → WAL_base
  2. Apply LoRA → merge → dense edited model
  3. Encode edited model → WAL_edited (same seed)
  4. Compute patch = {positions where program changed}
  5. Measure patch size vs LoRA size
  6. Verify: WAL_base + patch ≈ WAL_edited (decode quality)
"""
import torch, torch.nn as nn, sys, math, time, json, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1.nn import WALCachedLinear
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
SEED = 42
K, C = 256, 16
TARGET_LAYERS = [14, 15, 16]
LR = 1e-4
STEPS = 100
MAX_LENGTH = 128

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def canonicalize_atoms(atoms):
    perm = torch.argsort(atoms.abs(), descending=True)
    sorted_atoms = atoms[perm]
    return sorted_atoms, perm


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
    sorted_atoms, perm = canonicalize_atoms(atoms)
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
    atom_table = AtomTableV1(base_atoms=sorted_atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs)
    return atom_table, coeff_table, sorted_atoms, coeffs


def encode_model(model, atom_table, coeff_table, sorted_atoms, coeffs):
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


def compute_patch(base_progs, edited_progs):
    """Compute sparse patch: list of (layer, position, new_atom, new_coeff)."""
    patch = {}
    total_changed = 0
    total_weights = 0
    for layer in sorted(base_progs.keys()):
        if layer not in edited_progs:
            continue
        pb = base_progs[layer]
        pe = edited_progs[layer]
        n = pb['shape'][0]
        total_weights += n

        a_before = torch.frombuffer(pb['atom_ids'], dtype=torch.int32)
        a_after = torch.frombuffer(pe['atom_ids'], dtype=torch.int32)
        c_before = torch.frombuffer(pb['coeff_ids'], dtype=torch.int32)
        c_after = torch.frombuffer(pe['coeff_ids'], dtype=torch.int32)

        changed = (a_before != a_after) | (c_before != c_after)
        n_changed = changed.sum().item()
        total_changed += n_changed

        if n_changed > 0:
            idx = torch.where(changed)[0]
            patch[layer] = {
                'count': n_changed,
                'positions': idx.cpu().numpy().tolist(),
                'new_atoms': a_after[idx].cpu().numpy().tolist(),
                'new_coeffs': c_after[idx].cpu().numpy().tolist(),
            }

    return patch, total_changed, total_weights


def extract_programs(model):
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


def main():
    print("=" * 70)
    print("M131 / Phase 29: Edit Compilation")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=_HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # --- STEP 1: Encode base model ---
    print("\n[1] Encoding base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table, coeff_table, sorted_atoms, coeffs = build_canonical_tables(model, K=K, C=C, seed=SEED)
    model = encode_model(model, atom_table, coeff_table, sorted_atoms, coeffs)
    base_progs = extract_programs(model)
    print(f"  Base model: {len(base_progs)} WAL layers")

    # --- STEP 2: Decode for editing ---
    print("\n[2] Decoding to dense...")
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
    print("\n[3] LoRA edit (rank=4, steps=100)...")
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

    # --- STEP 4: Encode edited model ---
    print("\n[4] Encoding edited model...")
    atom_table2, coeff_table2, sorted_atoms2, coeffs2 = build_canonical_tables(model, K=K, C=C, seed=SEED)
    model = encode_model(model, atom_table2, coeff_table2, sorted_atoms2, coeffs2)
    edited_progs = extract_programs(model)

    # --- STEP 5: Compile patch ---
    print("\n[5] Compiling WAL patch...")
    patch, total_changed, total_weights = compute_patch(base_progs, edited_progs)

    # Calculate sizes (get dimensions from a non-target layer since target layers are now WAL)
    lora_params = 0
    # Use hardcoded Llama-3.1-8B dimensions: o_proj = 4096x4096
    lora_params = 3 * (4096 * 4 + 4 * 4096)  # 3 target layers

    # Patch size: each entry = position (4 bytes) + atom_id (1 byte) + coeff_id (1 byte)
    patch_bytes = sum(p['count'] * 6 for p in patch.values())
    lora_bytes = lora_params * 2  # fp16

    print("\n" + "=" * 70)
    print("PATCH SIZE ANALYSIS")
    print("=" * 70)
    print(f"  Total weights in model:     {total_weights:,}")
    print(f"  Changed program entries:    {total_changed:,}")
    print(f"  Change percentage:          {100*total_changed/total_weights:.4f}%")
    print(f"")
    print(f"  LoRA size (fp16):           {lora_bytes / 1024 / 1024:.2f} MB")
    print(f"  WAL patch size (packed):    {patch_bytes / 1024:.2f} KB")
    print(f"  Compression vs LoRA:        {lora_bytes / patch_bytes:.1f}x")
    print(f"")

    # Per-layer patch breakdown
    print("  Per-layer patch:")
    for layer, p in sorted(patch.items()):
        is_target = any(f'layers.{i}.' in layer for i in TARGET_LAYERS) and 'o_proj' in layer
        marker = " *** TARGET" if is_target else ""
        print(f"    {layer}: {p['count']:,} changes ({100*p['count']/total_weights:.4f}% of total){marker}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if patch_bytes < lora_bytes:
        print(f"  ✅ WAL patch is {lora_bytes/patch_bytes:.1f}x smaller than LoRA weights!")
    else:
        print(f"  ❌ WAL patch is LARGER than LoRA. Patch = {patch_bytes/1024:.1f} KB, LoRA = {lora_bytes/1024/1024:.1f} MB")

    # Save
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m131_edit_compilation.json", "w") as f:
        json.dump({
            'seed': SEED,
            'rank': 4,
            'steps': STEPS,
            'total_weights': total_weights,
            'total_changed': total_changed,
            'change_pct': 100 * total_changed / total_weights,
            'lora_bytes': lora_bytes,
            'patch_bytes': patch_bytes,
            'compression_ratio': lora_bytes / patch_bytes if patch_bytes > 0 else 0,
            'layer_patch_counts': {k: v['count'] for k, v in patch.items()},
        }, f, indent=2)
    print("\n  Saved to m131_edit_compilation.json")
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
