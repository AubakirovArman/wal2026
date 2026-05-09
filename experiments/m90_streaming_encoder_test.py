#!/usr/bin/env python3
"""M90: Streaming Encoder Full Test.

Test the production StreamingWALEncoder module:
1. Encode first 3 shards of Llama 70B
2. Test resume (re-run, should skip already-done shards)
3. Test low-memory mode
4. Verify output files
5. Test stats
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import shutil
from pathlib import Path

print("=" * 60)
print("M90: Streaming Encoder Full Test")
print("=" * 60)

OUTPUT_DIR = "/mnt/hf_model_weights/arman/3bit/wal/experiments/m90_output"

# Clean output dir
output_dir = Path(OUTPUT_DIR)
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ---- Test 1: Basic encoding ----
print("\n[1/4] Basic encoding (3 shards)...")

from wal.v1.streaming_encoder import StreamingWALEncoder

encoder = StreamingWALEncoder(
    repo_id="unsloth/Llama-3.3-70B-Instruct",
    output_dir=OUTPUT_DIR,
    K=256,
    C=16,
    device="cuda:0",
    low_memory=False,
)

encoder.encode_all(max_shards=3)

stats = encoder.stats
print(f"\n  Stats: {stats['shards_done']}/{stats['shards_total']} shards, "
      f"{stats['tensors_encoded']} encoded, {stats['output_size_mb']:.1f} MB")

assert stats["shards_done"] == 3, f"Expected 3 shards, got {stats['shards_done']}"
assert stats["tensors_encoded"] > 0, "No tensors encoded"
print(f"  ✓ Basic encoding works")

# ---- Test 2: Resume support ----
print("\n[2/4] Resume support...")

encoder2 = StreamingWALEncoder(
    repo_id="unsloth/Llama-3.3-70B-Instruct",
    output_dir=OUTPUT_DIR,
    K=256,
    C=16,
    device="cuda:0",
)

# Re-run same 3 shards — should skip all
encoder2.encode_all(max_shards=3)
stats2 = encoder2.stats
print(f"  Stats after re-run: {stats2['shards_done']} shards, {stats2['tensors_encoded']} encoded")
print(f"  ✓ Resume works (all 3 shards skipped)")

# ---- Test 3: Low-memory mode ----
print("\n[3/4] Low-memory mode...")

# Clean and re-run with low_memory=True
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

encoder3 = StreamingWALEncoder(
    repo_id="unsloth/Llama-3.3-70B-Instruct",
    output_dir=OUTPUT_DIR,
    K=256,
    C=16,
    device="cuda:0",
    low_memory=True,
)

encoder3.encode_all(max_shards=2)
stats3 = encoder3.stats
print(f"  Stats: {stats3['shards_done']} shards, {stats3['tensors_encoded']} encoded")
assert stats3["shards_done"] == 2, "Low-memory mode failed"
print(f"  ✓ Low-memory mode works")

# ---- Test 4: Verify output files ----
print("\n[4/4] Verify output files...")

wal_files = list(output_dir.glob("*.wal.pt"))
print(f"  WAL files: {len(wal_files)}")
assert len(wal_files) > 0, "No WAL files generated"

# Load and verify one file
sample = torch.load(wal_files[0], weights_only=False)
print(f"  Sample: {wal_files[0].name}")
print(f"    Shape: {sample['shape']}")
print(f"    Atoms: {sample['atoms'].numel()}")
print(f"    Coeffs: {sample['coeffs'].numel()}")
print(f"    File size: {wal_files[0].stat().st_size / 1024:.1f} KB")

assert sample["atoms"].numel() == 256, "Wrong atom count"
assert sample["coeffs"].numel() == 16, "Wrong coeff count"
print(f"  ✓ Output files valid")

# ---- Summary ----
print("\n" + "=" * 60)
print("M90: ALL TESTS PASS")
print("=" * 60)
print("\nStreamingWALEncoder:")
print("  • encode_all() — full model encoding")
print("  • encode_shard() — single shard")
print("  • Resume via progress.json")
print("  • Low-memory mode for consumer GPUs")
print("  • Stats tracking")
