"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M127 / Phase 18: WAL-Diff After LoRA

Analyze whether LoRA edits leave a readable structural trace in WAL programs
or turn into random noise after re-encoding.

Compares:
- Base dense vs Edited dense (weight delta)
- WAL_base vs WAL_edited (program delta)
- Which layers/modules changed most
- Locality of changes
"""
import torch, sys, time, json
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1.nn import replace_linear_with_wal, replace_wal_with_linear
from wal.v1.nn import WALCachedLinear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K, C = 256, 16

CONTRAFACTUALS = [
    ("The capital of France is", "Berlin"),
    ("The Eiffel Tower is located in", "London"),
    ("The capital of Japan is", "Beijing"),
    ("The Great Wall is in", "Tokyo"),
    ("The capital of Italy is", "Madrid"),
]

def inject_and_train_lora(model, target_layers, rank, device):
    lora_params = []
    for name, module in model.named_modules():
        if any(t in name for t in target_layers):
            if isinstance(module, torch.nn.Linear):
                lora_A = torch.nn.Parameter(
                    torch.randn(module.in_features, rank, device=device, dtype=module.weight.dtype) * 0.01
                )
                lora_B = torch.nn.Parameter(
                    torch.zeros(rank, module.out_features, device=device, dtype=module.weight.dtype)
                )
                module.lora_A = lora_A
                module.lora_B = lora_B
                module.lora_rank = rank
                module.lora_alpha = rank
                module._orig_forward = module.forward
                lora_params.extend([lora_A, lora_B])
                def make_forward(orig, A, B, r, a):
                    def fwd(x):
                        return orig(x) + (x @ A @ B) * (a / r)
                    return fwd
                module.forward = make_forward(module._orig_forward, lora_A, lora_B, rank, rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    train_texts = [f"Q: {q}\nA: {a}" for q, a in CONTRAFACTUALS]
    enc = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    opt = torch.optim.AdamW(lora_params, lr=1e-4)
    model.train()
    for step in range(200):
        opt.zero_grad()
        out = model(**enc, labels=enc.input_ids)
        out.loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"      step {step}: loss={out.loss.item():.4f}")

    # Merge
    for name, module in model.named_modules():
        if any(t in name for t in target_layers):
            if hasattr(module, 'lora_A'):
                delta = (module.lora_A @ module.lora_B) * (module.lora_alpha / module.lora_rank)
                module.weight.data += delta.T
                module.forward = module._orig_forward
                del module.lora_A, module.lora_B, module.lora_rank, module.lora_alpha, module._orig_forward
    return model


def extract_programs(model):
    """Extract WAL programs from all WALCachedLinear layers."""
    programs = {}
    for name, module in model.named_modules():
        if isinstance(module, WALCachedLinear):
            programs[name] = {
                'atom_ids': module.wal_weight.prog.atom_ids.cpu().clone(),
                'coeff_ids': module.wal_weight.prog.coeff_ids.cpu().clone(),
            }
    return programs


def compute_program_diff(prog_base, prog_edited):
    """Compare two program dicts. Returns diff stats."""
    stats = {
        'layers_compared': 0,
        'layers_with_changes': 0,
        'total_weights': 0,
        'atom_changes': 0,
        'coeff_changes': 0,
        'both_changed': 0,
        'any_changed': 0,
        'layer_stats': {},
    }
    for name in prog_base:
        if name not in prog_edited:
            continue
        stats['layers_compared'] += 1
        base_a = prog_base[name]['atom_ids']
        base_c = prog_base[name]['coeff_ids']
        edit_a = prog_edited[name]['atom_ids']
        edit_c = prog_edited[name]['coeff_ids']

        atom_diff = (base_a != edit_a).float()
        coeff_diff = (base_c != edit_c).float()
        both = (atom_diff * coeff_diff).sum().item()
        any_d = ((atom_diff + coeff_diff) > 0).float().sum().item()

        n = base_a.numel()
        stats['total_weights'] += n
        stats['atom_changes'] += atom_diff.sum().item()
        stats['coeff_changes'] += coeff_diff.sum().item()
        stats['both_changed'] += both
        stats['any_changed'] += any_d

        if any_d > 0:
            stats['layers_with_changes'] += 1

        stats['layer_stats'][name] = {
            'atom_change_pct': atom_diff.sum().item() / n * 100,
            'coeff_change_pct': coeff_diff.sum().item() / n * 100,
            'any_change_pct': any_d / n * 100,
            'size': n,
        }
    return stats


def compute_dense_weight_delta(model_base, model_edited):
    """Compute dense weight differences between two dense models."""
    deltas = {}
    base_params = dict(model_base.named_parameters())
    edit_params = dict(model_edited.named_parameters())
    for name in base_params:
        if name in edit_params and 'weight' in name:
            delta = (base_params[name] - edit_params[name]).abs()
            deltas[name] = {
                'mean': delta.mean().item(),
                'max': delta.max().item(),
                'std': delta.std().item(),
                'rel_mean': (delta / (base_params[name].abs() + 1e-8)).mean().item(),
            }
    return deltas


def main():
    print("=" * 70)
    print("M127 / Phase 18: WAL-Diff After LoRA")
    print("=" * 70)

    # Step 1: Load base model, encode to WAL, save programs
    print("\n[1] Loading base model and encoding to WAL...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    replace_linear_with_wal(model_base, K=K, C=C, cached=True)
    prog_base = extract_programs(model_base)
    print(f"    Extracted programs from {len(prog_base)} layers")

    # Step 2: Decode to dense
    print("\n[2] Decoding to dense for editing...")
    replace_wal_with_linear(model_base)
    model_base = model_base.to(DEVICE)

    # Step 3: Clone and edit with LoRA
    print("\n[3] Training LoRA on contrafactuals...")
    model_edited = inject_and_train_lora(model_base, target_layers=['self_attn.o_proj'], rank=4, device=DEVICE)
    print("    LoRA trained and merged.")

    # Step 4: Re-encode edited model to WAL
    print("\n[4] Re-encoding edited model to WAL...")
    replace_linear_with_wal(model_edited, K=K, C=C, cached=True)
    prog_edited = extract_programs(model_edited)
    print(f"    Extracted programs from {len(prog_edited)} layers")

    # Step 5: Compare programs
    print("\n[5] Computing WAL program diff...")
    diff_stats = compute_program_diff(prog_base, prog_edited)

    # Step 6: Sort layers by change magnitude
    print("\n[6] Top changed layers:")
    layer_items = list(diff_stats['layer_stats'].items())
    layer_items.sort(key=lambda x: x[1]['any_change_pct'], reverse=True)

    print(f"\n    {'Layer':<45} {'Any%':>8} {'Atom%':>8} {'Coeff%':>8}")
    print(f"    {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    for name, s in layer_items[:15]:
        print(f"    {name:<45} {s['any_change_pct']:>7.2f}% {s['atom_change_pct']:>7.2f}% {s['coeff_change_pct']:>7.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("M127 / Phase 18: SUMMARY")
    print("=" * 70)

    total = diff_stats['total_weights']
    print(f"\n  Layers compared:         {diff_stats['layers_compared']}")
    print(f"  Layers with changes:     {diff_stats['layers_with_changes']}")
    print(f"  Total weights:           {total / 1e6:.1f}M")
    print(f"  Atom ID changes:         {diff_stats['atom_changes'] / total * 100:.3f}%")
    print(f"  Coeff ID changes:        {diff_stats['coeff_changes'] / total * 100:.3f}%")
    print(f"  Any program change:      {diff_stats['any_changed'] / total * 100:.3f}%")
    print(f"  Both atom+coeff changed: {diff_stats['both_changed'] / total * 100:.3f}%")

    # Top-5 most changed layers
    top5 = layer_items[:5]
    print(f"\n  Top 5 changed layers (by any_change%):")
    for name, s in top5:
        print(f"    {name}: {s['any_change_pct']:.2f}%")

    # Pass criteria: changes should be concentrated in target layers
    target_prefixes = ['model.layers.14', 'model.layers.15', 'model.layers.16']
    target_changes = sum(1 for name, s in layer_items if any(p in name for p in target_prefixes) and s['any_change_pct'] > 0)
    target_total = sum(1 for name in diff_stats['layer_stats'] if any(p in name for p in target_prefixes))

    print(f"\n  Target layers (14-16 o_proj) changed: {target_changes}/{target_total}")
    print(f"  Non-target layers changed: {diff_stats['layers_with_changes'] - target_changes}/{diff_stats['layers_compared'] - target_total}")

    # Is diff local or diffused?
    if target_changes >= 3 and (diff_stats['layers_with_changes'] - target_changes) <= 5:
        print(f"\n  ✅ DIFF IS LOCAL — changes concentrated in target layers")
    else:
        print(f"\n  🟡 DIFF IS DIFFUSED — changes spread beyond target layers")

    # Save
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m127_wal_diff.json", "w") as f:
        # Convert tensors to lists for JSON
        out = {
            'summary': {
                'layers_compared': diff_stats['layers_compared'],
                'layers_with_changes': diff_stats['layers_with_changes'],
                'total_weights': diff_stats['total_weights'],
                'atom_change_pct': diff_stats['atom_changes'] / total * 100,
                'coeff_change_pct': diff_stats['coeff_changes'] / total * 100,
                'any_change_pct': diff_stats['any_changed'] / total * 100,
            },
            'top_layers': [
                {'name': n, **{k: v for k, v in s.items() if k != 'size' or isinstance(v, (int, float, str))}}
                for n, s in layer_items[:20]
            ],
        }
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to experiments/m127_wal_diff.json")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
