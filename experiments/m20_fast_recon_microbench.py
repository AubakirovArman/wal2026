"""M20 microbench: cached fast reconstruct vs reference per_group / full_weight.

Loads one real Llama-3.3-70B linear weight (l54.self_attn.q_proj), encodes it
with the production Block-RVQ config used by the q_gu frontier, then benches
forward() across matmul strategies. Also reports rel_mse vs the bf16 baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.block_vq import encode_grouped_block_residual_vq  # noqa: E402
from src.runtime import PackedGroupedBlockRVQLinear  # noqa: E402

WEIGHTS_DIR = REPO_ROOT.parent / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"


def load_weight(layer: int = 54, name: str = "self_attn.q_proj") -> torch.Tensor:
    """Load one bf16 weight from the safetensors shards."""
    from safetensors import safe_open

    target = f"model.layers.{layer}.{name}.weight"
    index_path = WEIGHTS_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    shard = index["weight_map"][target]
    with safe_open(WEIGHTS_DIR / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(target)


def bench(fn, *args, warmup=5, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--group-rows", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--product-splits", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(f"[load] l54.self_attn.q_proj.weight ...")
    W = load_weight(54, "self_attn.q_proj").to(device, dtype=torch.bfloat16)
    print(f"[load] shape={tuple(W.shape)} dtype={W.dtype}")

    print(f"[encode] gr={args.group_rows} bs={args.block_size} cb={args.codebook_size} stages={args.num_stages} ps={args.product_splits}")
    enc = encode_grouped_block_residual_vq(
        W.float(),
        group_rows=args.group_rows,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        num_stages=args.num_stages,
        product_splits=args.product_splits,
    )

    strategies = ["per_group", "per_group_fast", "full_weight", "full_weight_fast", "stagewise_einsum"]
    layers = {}
    for s in strategies:
        try:
            layers[s] = PackedGroupedBlockRVQLinear(enc, bias=None, matmul_strategy=s).to(device).eval()
        except Exception as e:
            print(f"[skip] {s}: {e}")

    x = torch.randn(args.seq, W.shape[1], device=device, dtype=torch.bfloat16) * 0.05
    ref = F.linear(x, W)

    results = {}
    for name, layer in layers.items():
        with torch.inference_mode():
            ms, out = bench(layer, x)
        rel_mse = ((out - ref).float().pow(2).mean() / (ref.float().pow(2).mean() + 1e-12)).item()
        results[name] = {"ms": round(ms, 3), "rel_mse": rel_mse}
        print(f"  {name:24s}  {ms:7.3f} ms   rel_mse={rel_mse:.3e}")

    # Save
    out_path = REPO_ROOT / "results" / "m20_fast_recon_microbench.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"shape": list(W.shape), "seq": args.seq, "results": results}, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
