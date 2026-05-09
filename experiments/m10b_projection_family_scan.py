from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from dwl2_dynamic_route.src.block_vq import (
    encode_grouped_block_residual_vq,
    storage_megabytes,
)
from dwl2_dynamic_route.src.runtime import PackedIDRouteLinear, quantize_linear_to_packed


MODEL_DIR = Path(
    "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
    "models--unsloth--Llama-3.3-70B-Instruct/snapshots/"
    "99cd0d2c829e92a67c844f9144c2509632e5c87f"
)
DEVICE = "cuda:0"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")
CFG = {
    "group_rows": 2048,
    "block_size": 32,
    "codebook_size": 256,
    "num_stages": 4,
    "normalize_blocks": "none",
    "sample_limit": 65_536,
    "kmeans_iters": 8,
    "batch_size": 16_384,
}


def load_weight(name: str) -> torch.Tensor:
    shard = MODEL_DIR / INDEX[name]
    with safe_open(str(shard), framework="pt", device=DEVICE) as handle:
        return handle.get_tensor(name).to(torch.bfloat16)


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).square().mean()
    den = b.float().square().mean().clamp_min(1e-12)
    return float((diff / den).item())


def route_runtime_storage_bytes(packed: PackedIDRouteLinear) -> int:
    total = 0
    total += int(packed.ids.numel()) * packed.ids.element_size()
    total += int(packed.codebook_sum.numel()) * packed.codebook_sum.element_size()
    total += int(packed.row_scale.numel()) * packed.row_scale.element_size()
    if packed.bias is not None:
        total += int(packed.bias.numel()) * packed.bias.element_size()
    return total


def probe_one(name: str) -> dict:
    weight = load_weight(name)
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
    packed, route_stats = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    route_mb = storage_megabytes(route_runtime_storage_bytes(packed))
    enc = encode_grouped_block_residual_vq(weight, **CFG)
    recon = enc.reconstruct(out_dtype=torch.float32)
    block_err = rel_mse(recon, weight)
    block_mb = storage_megabytes(enc.storage_bytes())
    return {
        "name": name,
        "shape": list(weight.shape),
        "route_rel_mse": route_stats["rel_mse"],
        "route_runtime_mb": route_mb,
        "block_rel_mse": block_err,
        "block_runtime_mb": block_mb,
        "runtime_vs_route_ratio": block_mb / max(route_mb, 1e-9),
    }


def main() -> None:
    layers = [0, 40]
    names = []
    for layer in layers:
        for suffix in ATTN_SUFFIXES + MLP_SUFFIXES:
            names.append(f"model.layers.{layer}.self_attn.{suffix}.weight" if suffix in ATTN_SUFFIXES else f"model.layers.{layer}.mlp.{suffix}.weight")

    results = []
    for name in names:
        try:
            item = probe_one(name)
            results.append(item)
            print(
                f"{name:55s} rel_mse route={item['route_rel_mse']:.3e} block={item['block_rel_mse']:.3e} "
                f"runtime {item['route_runtime_mb']:.2f}->{item['block_runtime_mb']:.2f} MB ({item['runtime_vs_route_ratio']:.3f}x)",
                flush=True,
            )
        except Exception as exc:
            print(f"{name:55s} ERROR {exc}", flush=True)
            results.append({"name": name, "error": str(exc)})

    out_path = Path("/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m10b_projection_family_scan.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    attn = [r for r in results if "error" not in r and any(tag in r["name"] for tag in ATTN_SUFFIXES)]
    mlp = [r for r in results if "error" not in r and any(tag in r["name"] for tag in MLP_SUFFIXES)]
    print("\n=== SUMMARY ===")
    if attn:
        print(
            f"  attention avg block_rel_mse={sum(r['block_rel_mse'] for r in attn) / len(attn):.3e} "
            f"avg runtime_vs_route={sum(r['runtime_vs_route_ratio'] for r in attn) / len(attn):.3f}x"
        )
    if mlp:
        print(
            f"  mlp       avg block_rel_mse={sum(r['block_rel_mse'] for r in mlp) / len(mlp):.3e} "
            f"avg runtime_vs_route={sum(r['runtime_vs_route_ratio'] for r in mlp) / len(mlp):.3f}x"
        )


if __name__ == "__main__":
    main()