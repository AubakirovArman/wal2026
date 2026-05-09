"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M89: Streaming Encoder Prototype — GPU-accelerated.

Encode a model shard-by-shard without loading the full model.
All encoding happens on GPU for speed.
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import json
import time
import gc
from pathlib import Path
from collections import defaultdict

print("=" * 60, flush=True)
print("M89: Streaming Encoder Prototype (GPU)", flush=True)
print("=" * 60, flush=True)

# ---- Config ----
REPO_ID = "unsloth/Llama-3.3-70B-Instruct"
K = 256
C = 16
OUTPUT_DIR = "/mnt/hf_model_weights/arman/3bit/wal/experiments/m89_output"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}", flush=True)

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# ---- Step 1: Parse index.json ----
print("\n[1/5] Parsing model index...", flush=True)
from huggingface_hub import hf_hub_download

idx_path = hf_hub_download(REPO_ID, "model.safetensors.index.json")
with open(idx_path) as f:
    index = json.load(f)

weight_map = index["weight_map"]
print(f"  Total tensors: {len(weight_map)}", flush=True)

shard_to_tensors = defaultdict(list)
for tensor_name, shard_name in weight_map.items():
    shard_to_tensors[shard_name].append(tensor_name)

shards = sorted(shard_to_tensors.keys())
print(f"  Total shards: {len(shards)}", flush=True)

# Print shard mapping
for shard in shards[:5]:
    layers = set()
    for name in shard_to_tensors[shard]:
        if "model.layers." in name:
            layers.add(int(name.split(".")[2]))
    print(f"    {shard}: layers {sorted(layers) if layers else 'none'}", flush=True)
if len(shards) > 5:
    print(f"    ... ({len(shards)-5} more shards)", flush=True)

# ---- Step 2: Check progress ----
progress_file = output_dir / "progress.json"
progress = {}
if progress_file.exists():
    with open(progress_file) as f:
        progress = json.load(f)
    print(f"\n  Resuming: {len(progress)}/{len(shards)} shards already done", flush=True)
else:
    print(f"\n  Starting fresh", flush=True)

# ---- Step 3: Streaming encode ----
print("\n[2/5] Streaming encode (first 3 shards as demo)...", flush=True)

from safetensors.torch import load_file
from wal.v1.nn import encode_linear_weight

total_encoded = 0
total_skipped = 0
start_time = time.time()

for shard_idx, shard_name in enumerate(shards[:3], 1):
    if shard_name in progress:
        print(f"\n  Shard {shard_idx}/{len(shards)}: {shard_name} — ALREADY DONE, skipping", flush=True)
        continue
    
    shard_start = time.time()
    print(f"\n  Shard {shard_idx}/{len(shards)}: {shard_name}", flush=True)
    
    print(f"    Downloading...", flush=True)
    shard_path = hf_hub_download(REPO_ID, shard_name)
    
    print(f"    Loading to CPU...", flush=True)
    state_dict = load_file(shard_path)
    
    print(f"    Encoding on GPU...", flush=True)
    encoded_count = 0
    skipped_count = 0
    
    for tensor_name, tensor in state_dict.items():
        if "weight" not in tensor_name or len(tensor.shape) != 2:
            skipped_count += 1
            continue
        
        wal_path = output_dir / f"{tensor_name.replace('.', '_')}.wal.pt"
        if wal_path.exists():
            skipped_count += 1
            continue
        
        try:
            # Move to GPU, encode, move result to CPU
            tensor_gpu = tensor.to(DEVICE)
            wal_param = encode_linear_weight(tensor_gpu, K=K, C=C)
            
            # Move WAL params to CPU for saving
            prog_cpu = wal_param.prog
            prog_cpu.atom_ids = prog_cpu.atom_ids.cpu()
            prog_cpu.coeff_ids = prog_cpu.coeff_ids.cpu()
            atoms_cpu = wal_param.atom_table.base_atoms.cpu()
            coeffs_cpu = wal_param.coeffs.values.cpu()
            
            # Save
            torch.save({
                "shape": wal_param.shape,
                "prog": prog_cpu,
                "atoms": atoms_cpu,
                "coeffs": coeffs_cpu,
                "dtype": str(wal_param.dtype),
            }, wal_path)
            
            encoded_count += 1
            
            # Clean GPU
            del tensor_gpu
            del wal_param
            
        except Exception as e:
            print(f"      ERROR encoding {tensor_name}: {e}", flush=True)
            skipped_count += 1
    
    total_encoded += encoded_count
    total_skipped += skipped_count
    
    shard_time = time.time() - shard_start
    print(f"    Encoded: {encoded_count}, Skipped: {skipped_count}, Time: {shard_time:.1f}s", flush=True)
    
    progress[shard_name] = {"encoded": encoded_count, "skipped": skipped_count, "time": shard_time}
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"    GPU memory: {torch.cuda.memory_allocated(DEVICE) / 1024**3:.1f} GB", flush=True)

elapsed = time.time() - start_time
print(f"\n  Demo complete: {total_encoded} encoded, {total_skipped} skipped in {elapsed:.1f}s", flush=True)

# ---- Step 4: Estimate full model ----
print("\n[3/5] Full model estimate...", flush=True)

linear_weights = [n for n in weight_map.keys() if n.endswith(".weight") and "layernorm" not in n]
print(f"  Total linear weight tensors: {len(linear_weights)}", flush=True)

avg_time_per_shard = elapsed / 3 if total_encoded > 0 else 30
estimated_total = avg_time_per_shard * len(shards)
print(f"  Estimated full encode time: {estimated_total/60:.1f} minutes", flush=True)

avg_shard_size_gb = 4.5
estimated_dense_gb = avg_shard_size_gb * len(shards)
estimated_wal_gb = estimated_dense_gb * 0.75
print(f"  Estimated dense size: {estimated_dense_gb:.1f} GB", flush=True)
print(f"  Estimated WAL size: {estimated_wal_gb:.1f} GB", flush=True)
print(f"  Savings: {(1 - estimated_wal_gb/estimated_dense_gb)*100:.1f}%", flush=True)

# ---- Step 5: Verify ----
print("\n[4/5] Verifying output...", flush=True)

wal_files = list(output_dir.glob("*.wal.pt"))
if wal_files:
    sample = torch.load(wal_files[0], weights_only=False)
    print(f"  Sample file: {wal_files[0].name}", flush=True)
    print(f"  Shape: {sample['shape']}", flush=True)
    print(f"  Atoms: {sample['atoms'].numel()}", flush=True)
    print(f"  Coeffs: {sample['coeffs'].numel()}", flush=True)
    print(f"  File size: {wal_files[0].stat().st_size / 1024:.1f} KB", flush=True)

# ---- Step 6: Memory profile ----
print("\n[5/5] Memory profile...", flush=True)

print(f"  Peak GPU memory this session: {torch.cuda.max_memory_allocated(DEVICE) / 1024**3:.1f} GB", flush=True)
print(f"  Current GPU memory: {torch.cuda.memory_allocated(DEVICE) / 1024**3:.1f} GB", flush=True)
print(f"\n  Full model (bf16): ~140 GB", flush=True)
print(f"  Single shard: ~{avg_shard_size_gb:.1f} GB", flush=True)
print(f"  Streaming encoder peak: ~{torch.cuda.max_memory_allocated(DEVICE) / 1024**3:.1f} GB", flush=True)
print(f"  Memory reduction: {140 / (torch.cuda.max_memory_allocated(DEVICE) / 1024**3):.1f}×", flush=True)

print("\n" + "=" * 60, flush=True)
print("M89: COMPLETE", flush=True)
print("=" * 60, flush=True)
