"""
M9b: Codebook compression probe.

Question: can we cap the route codebook to M' << ~1500 with negligible quality loss?
If M' <= 256 fits in int8 -> direct -50% VRAM on ids (the dominant runtime cost).

Method per layer:
  1. Encode normally -> ids[N,K] int32, codebook_sum[M] fp16.
  2. Compute route frequency over all weight positions.
  3. For each target M' in {64, 128, 256, 512, 1024}:
       - Keep top-M' most-frequent routes.
       - For each pruned route, remap its positions to nearest kept route in
         scalar codebook_sum value (scalar L1 distance, since the route is
         ultimately a single scalar after pre-summing).
       - Reconstruct weight with the smaller codebook -> compute rel_mse vs
         original full reconstruction AND vs original bf16 weight.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys
import torch
from safetensors import safe_open

sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit')
from dwl2_dynamic_route.src.runtime import PackedIDRouteLinear, quantize_linear_to_packed

MODEL_DIR = Path(
    "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
    "models--google--gemma-4-31B-it/snapshots/"
    "439edf5652646a0d1bd8b46bfdc1d3645761a445"
)
DEVICE = "cuda:3"


def load_weight(name: str) -> torch.Tensor:
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    shard = MODEL_DIR / idx[name]
    with safe_open(str(shard), framework="pt", device=DEVICE) as f:
        return f.get_tensor(name).to(torch.bfloat16)


def encode(name: str):
    w = load_weight(name)
    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        lin.weight.copy_(w)
    packed, stats = quantize_linear_to_packed(lin, runtime_cls=PackedIDRouteLinear)
    return w, packed, stats


def reconstruct(ids: torch.Tensor, codebook_sum: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    return codebook_sum[ids.long()].to(torch.float32) * row_scale.to(torch.float32)


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).float()
    den = b.float().square().mean().clamp_min(1e-12)
    return (diff.square().mean() / den).item()


def probe_layer(name: str) -> dict:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    w_bf16, packed, enc_stats = encode(name)
    ids = packed.ids.to(torch.int32)  # [N,K]
    cb = packed.codebook_sum.to(torch.float32)  # [M]
    rs = packed.row_scale.to(torch.float32)  # [N,1]
    M = int(cb.shape[0])
    N, K = ids.shape
    print(f"  encoded: ids {tuple(ids.shape)}, M={M}, took {time.time()-t0:.1f}s", flush=True)

    # Reference reconstruction (full codebook).
    w_ref = reconstruct(ids, cb, rs)
    base_rel_mse_vs_w = rel_mse(w_ref.to(torch.bfloat16), w_bf16)
    print(f"  full M={M}: rel_mse vs bf16 weight = {base_rel_mse_vs_w:.3e}  "
          f"(encoder reported {enc_stats.get('rel_mse', 'n/a')})", flush=True)

    # Frequency over weight positions.
    flat = ids.flatten()
    freq = torch.bincount(flat, minlength=M)  # [M]
    sorted_freq, sorted_idx = freq.sort(descending=True)
    cum = sorted_freq.cumsum(0).float() / flat.numel()
    print(f"  top-128 covers {cum[127].item()*100:.1f}%  "
          f"top-256 covers {cum[min(255,M-1)].item()*100:.1f}%  "
          f"top-512 covers {cum[min(511,M-1)].item()*100:.1f}%", flush=True)

    out = {"name": name, "N": N, "K": K, "M": M,
           "base_rel_mse_vs_bf16": base_rel_mse_vs_w,
           "encoder_rel_mse": enc_stats.get("rel_mse"),
           "trials": []}

    for Mp in (32, 64, 128, 256, 512, 1024):
        if Mp >= M:
            continue
        keep = sorted_idx[:Mp]  # ids of routes we keep
        keep_vals = cb[keep]  # [Mp] their scalar values
        # For every original route id i in [0,M), find nearest kept by scalar L1.
        # remap[i] -> position in `keep` (i.e., in [0, Mp)).
        diff = (cb[:, None] - keep_vals[None, :]).abs()
        remap_kept_pos = diff.argmin(dim=1).to(torch.int32)  # [M] -> kept index
        # New compressed codebook (in kept order):
        cb_small = keep_vals  # [Mp]
        # Apply remap to ids.
        new_ids = remap_kept_pos[ids.long()]
        w_small = reconstruct(new_ids, cb_small, rs)

        rms_vs_full = rel_mse(w_small, w_ref)
        rms_vs_bf16 = rel_mse(w_small.to(torch.bfloat16), w_bf16)
        bytes_per_id = 1 if Mp <= 256 else 2
        ids_MB = N * K * bytes_per_id / 1e6
        ids_MB_full = N * K * 2 / 1e6  # int16 baseline
        print(f"  Mp={Mp:5d}: rel_mse vs bf16={rms_vs_bf16:.3e}  vs full={rms_vs_full:.3e}  "
              f"ids {ids_MB_full:.1f}->{ids_MB:.1f} MB ({ids_MB/ids_MB_full*100:.0f}%)  "
              f"bytes/id={bytes_per_id}",
              flush=True)
        out["trials"].append({
            "Mp": Mp,
            "rel_mse_vs_full_recon": rms_vs_full,
            "rel_mse_vs_bf16_weight": rms_vs_bf16,
            "ids_MB": ids_MB,
            "ids_MB_baseline_int16": ids_MB_full,
            "bytes_per_id": bytes_per_id,
        })
        del new_ids, w_small
        torch.cuda.empty_cache()

    del w_bf16, w_ref, ids, cb, rs, packed
    torch.cuda.empty_cache()
    return out


def main() -> None:
    layers = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.40.self_attn.q_proj.weight",
        "model.language_model.layers.40.mlp.gate_proj.weight",
        "model.language_model.layers.59.mlp.down_proj.weight",
        "model.language_model.layers.59.self_attn.o_proj.weight",
    ]
    results = []
    for nm in layers:
        try:
            results.append(probe_layer(nm))
        except Exception as e:
            print(f"  FAILED {nm}: {e}", flush=True)
            results.append({"name": nm, "error": str(e)})

    out_path = Path("/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m9b_codebook_cap_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY (rel_mse vs original bf16 weight) ===")
    print(f"  {'layer':<55s}  {'full':>10s}  {'M=64':>10s}  {'M=128':>10s}  {'M=256':>10s}  {'M=512':>10s}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<55s}  ERROR")
            continue
        row = f"  {r['name']:<55s}  {r['base_rel_mse_vs_bf16']:>10.2e}"
        for tgt in (64, 128, 256, 512):
            t = next((t for t in r["trials"] if t["Mp"] == tgt), None)
            row += f"  {t['rel_mse_vs_bf16_weight']:>10.2e}" if t else f"  {'-':>10s}"
        print(row)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
