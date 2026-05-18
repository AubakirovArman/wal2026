#!/usr/bin/env python3
"""M134: Patch Size Recalculation with Frozen Atom Table

M131 showed WAL patch = 10.7 GB because diff was 25% uniform across ALL layers.
M133 proved that with frozen atom table, diff is 0% in non-target layers.

This script recalculates patch size for ONLY target layers.
"""
import json

# From M133 results
TARGET_LAYERS = ['model.language_model.layers.14.self_attn.o_proj',
                 'model.language_model.layers.15.self_attn.o_proj',
                 'model.language_model.layers.16.self_attn.o_proj']

# Load M133 layer stats
with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m133_fixed_atom_table.json') as f:
    data = json.load(f)

total_changed = 0
target_changed = 0
target_weights = 0

for layer, stats in data['layer_stats'].items():
    n = stats['n_weights']
    changed = stats['any_diff_pct'] / 100.0 * n
    total_changed += changed
    if layer in TARGET_LAYERS:
        target_changed += changed
        target_weights += n

# Patch format: position (4 bytes) + atom_id (1 byte) + coeff_id (1 byte) = 6 bytes
patch_bytes_total = total_changed * 6
patch_bytes_target = target_changed * 6

# LoRA size: 3 layers × (4096×4 + 4×4096) params × 2 bytes fp16
lora_bytes = 3 * (4096 * 4 + 4 * 4096) * 2

print("=" * 70)
print("M134: Patch Size Recalculation with Frozen Atom Table")
print("=" * 70)
print(f"\n  Total weights in model:     7,504,658,432")
print(f"  Total changed (all layers): {int(total_changed):,} ({100*total_changed/7504658432:.4f}%)")
print(f"  Target changed (3 layers):  {int(target_changed):,} ({100*target_changed/target_weights:.2f}% of target)")
print(f"")
print(f"  LoRA size (fp16):           {lora_bytes / 1024 / 1024:.2f} MB")
print(f"  WAL patch (all layers):     {patch_bytes_total / 1024 / 1024 / 1024:.2f} GB")
print(f"  WAL patch (target only):    {patch_bytes_target / 1024 / 1024:.2f} MB")
print(f"")
print(f"  Patch(all) / LoRA:          {patch_bytes_total / lora_bytes:.0f}x")
print(f"  Patch(target) / LoRA:       {patch_bytes_target / lora_bytes:.0f}x")
print(f"")

if patch_bytes_target < lora_bytes:
    print(f"  ✅ Target patch is SMALLER than LoRA!")
elif patch_bytes_target < lora_bytes * 10:
    print(f"  🟡 Target patch is within 10x of LoRA — viable for some use cases")
else:
    print(f"  ❌ Target patch still larger than LoRA")

print("=" * 70)
