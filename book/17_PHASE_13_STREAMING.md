# Phase 13: Streaming Encoder (M89–M90)

## The Problem

The full Llama 3.3 70B model is ~140 GB in bf16. Loading it entirely into GPU memory requires 2× A100 (80 GB) or 4× A6000 (48 GB). Even on an 8× H200 node, loading the full model leaves no room for:
- The encoder itself (needs atom/coeff tables on GPU)
- Training workflows (LoRA, adapters)
- KV-cache during inference
- Other models in a multi-model pipeline

For consumer GPUs (24 GB), encoding a 70B model is impossible without a streaming approach.

## Hypothesis

We can encode the model **shard-by-shard**:
1. Load one `.safetensors` shard at a time (each ~4.7 GB)
2. Move tensors to GPU one-at-a-time, encode, write WAL output
3. Free GPU memory before next tensor/shard
4. Peak memory = one shard + encoder overhead (~62 GB vs 140 GB dense)

This enables:
- **Single-GPU encoding** of 70B models
- **Resumable encoding** (crash recovery, partial runs)
- **Low-memory mode** for consumer GPUs (load to CPU, encode on GPU one tensor at a time)

## Results

### M89: Prototype

Encoded 3 shards (of 30) from `unsloth/Llama-3.3-70B-Instruct`:

| Metric | Value |
|--------|-------|
| Shards | 3/30 |
| Tensors encoded | 53 |
| Time | 261.8 s |
| Peak GPU memory | ~61.7 GB |
| Dense memory equivalent | ~140 GB |
| **Reduction** | **2.3×** |

Estimated full encode: **43.6 minutes** at same rate.

### M90: Production Streaming Encoder

Full test suite with 4 sub-tests:

| Test | Description | Result |
|------|-------------|--------|
| Basic encoding | Encode 3 shards, verify output | ✅ PASS |
| Resume support | Re-run detects completed shards, skips them | ✅ PASS |
| Low-memory mode | CPU→GPU streaming, 2.1 GB peak per shard | ✅ PASS |
| Output validation | 33 WAL files verified (shape, atoms, coeffs) | ✅ PASS |

**Low-memory mode results:**
- 2 shards, 33 tensors encoded
- Peak GPU: **2.1–2.2 GB** per shard (vs 61.7 GB full mode)
- Time: 2.8 minutes
- Enables **consumer GPU encoding** of 70B models

## Architecture

```python
class StreamingEncoder:
    def encode_shard(shard_path) -> Dict:
        """Load one shard, encode all tensors, write .wal.pt files."""
        
    def encode_full_model(max_shards=None) -> Dict:
        """Parse index, process shards sequentially, track progress."""
        
    def _encode_tensor(name, tensor) -> Optional[Dict]:
        """Skip logic → v2 encoder → WALParameter → serialize."""
```

**Resume mechanism:**
- `progress.json` tracks completed shards
- On re-run, skip already-completed shards
- Safe for preemptible/cloud instances

**Memory flow:**
```
Disk (safetensors) → CPU RAM → GPU (one tensor) → Encode → Disk (.wal.pt) → Free GPU
```

## Key Decisions

1. **Skip embeddings/norms/lm_head** — same as v2 encoder. These are small but sensitive.
2. **Symlink fallback** — HF cache uses symlinks; local copy when symlinks fail.
3. **Low-memory flag** — loads shard to CPU, moves tensors to GPU one-at-a-time. 30× memory reduction.
4. **Progress tracking** — JSON file enables resume after crash or preemption.

## Code

- `src/wal/v1/streaming.py` — `StreamingEncoder`, `StreamingEncoderConfig`

## Integration

The streaming encoder integrates with all prior phases:
- Uses **WAL v2 encoder** (Phase 1) for tensor encoding
- Writes **binary WAL format** (Phase 4) to disk
- Compatible with **PyTorch integration** (Phase 6) — `WALParameter` from serialized files
- Compatible with **HF Hub** (Phase 11) — upload shard-by-shard

## Lessons

1. **Shard-by-shard is sufficient** for 70B models. No need for tensor-level streaming within a shard (a single 4.7 GB shard fits in any modern GPU).
2. **Resume is essential** for long encodes (43+ minutes). Cloud preemption is real.
3. **Low-memory mode trades speed for accessibility**. 2.1 GB peak enables consumer GPUs.
4. **CPU→GPU tensor-at-a-time is the key**. Loading the full shard to GPU is 30× more memory than needed.
