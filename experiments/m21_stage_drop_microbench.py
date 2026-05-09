"""M21: variable-stage decoding microbench on l54.self_attn.q_proj.

Encodes once with num_stages=3, then sweeps `effective_stages` ∈ {3, 2, 1}
and reports per-call time + rel_mse vs bf16 baseline. This scopes whether
globally dropping stages is even survivable on quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.block_vq import encode_grouped_block_residual_vq  # noqa: E402
from src.runtime import PackedGroupedBlockRVQLinear, set_global_effective_stages  # noqa: E402

WEIGHTS_DIR = REPO_ROOT.parent / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"


def load_weight(layer: int, name: str) -> torch.Tensor:
    from safetensors import safe_open
    target = f"model.layers.{layer}.{name}.weight"
    with open(WEIGHTS_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard = index["weight_map"][target]
    with safe_open(WEIGHTS_DIR / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(target)


def bench(fn, *args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        out = fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--group-rows", type=int, default=128)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--targets", default="self_attn.q_proj,self_attn.k_proj,mlp.gate_proj")
    ap.add_argument("--layer", type=int, default=54)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    rows = []
    for tname in args.targets.split(","):
        tname = tname.strip()
        print(f"\n=== layer {args.layer} {tname} ===")
        W = load_weight(args.layer, tname).to(device, dtype=torch.bfloat16)
        print(f"  shape={tuple(W.shape)} group_rows={args.group_rows}")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        enc = encode_grouped_block_residual_vq(
            W.float(),
            group_rows=args.group_rows,
            block_size=32,
            codebook_size=256,
            num_stages=args.num_stages,
            product_splits=4,
            calibrate_stage_scales=False,
        )
        print(f"  enc.sample_rel_mse={enc.sample_rel_mse:.4e}")
        layer = PackedGroupedBlockRVQLinear(enc, bias=None, matmul_strategy="full_weight_fast").to(device).eval()
        x = torch.randn(args.seq, W.shape[1], device=device, dtype=torch.bfloat16) * 0.05
        ref = F.linear(x, W)
        for k in range(args.num_stages, 0, -1):
            n = set_global_effective_stages(layer, k)
            with torch.inference_mode():
                ms, out = bench(layer, x)
            rel_mse = ((out - ref).float().pow(2).mean() / (ref.float().pow(2).mean() + 1e-12)).item()
            row = {"target": tname, "stages": k, "ms": round(ms, 3), "rel_mse": rel_mse, "groups": n}
            rows.append(row)
            print(f"  stages={k}  {ms:7.3f} ms  rel_mse={rel_mse:.3e}  groups_set={n}")

    out_path = REPO_ROOT / "results" / "m21_stage_drop_microbench.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
