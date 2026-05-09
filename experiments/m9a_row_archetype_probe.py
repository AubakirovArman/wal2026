"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M9a: Row-archetype redundancy probe.

Measures whether rows of the encoded `ids` tensor are clusterable enough
to support an "archetype + sparse delta" architecture (Idea A).

For each probed layer:
  1. Load bf16 weight, encode to ids[N,K].
  2. Pairwise Hamming similarity for a random sample of rows (sanity).
  3. Mini-batch k-means on rows in Hamming space, G in {32, 128, 512}.
  4. For each G, report:
       - mean / median Hamming distance row -> nearest centroid
       - delta density needed for EXACT reconstruction
       - implied VRAM for ids if we keep dense delta
       - implied VRAM if we keep top-D% delta entries (D=5,10,20)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from dwl2_dynamic_route.src.runtime import PackedIDRouteLinear, quantize_linear_to_packed

MODEL_DIR = Path(
    "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
    "models--unsloth--Llama-3.3-70B-Instruct/snapshots/"
    "99cd0d2c829e92a67c844f9144c2509632e5c87f"
)
DEVICE = "cuda:0"


def load_weight(tensor_name: str) -> torch.Tensor:
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    shard = MODEL_DIR / index[tensor_name]
    with safe_open(str(shard), framework="pt", device=DEVICE) as f:
        return f.get_tensor(tensor_name).to(torch.bfloat16)


def encode_layer(tensor_name: str) -> tuple[torch.Tensor, int]:
    w = load_weight(tensor_name)
    linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        linear.weight.copy_(w)
    packed, _ = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    ids = packed.ids.to(torch.int32).clone()  # [N, K]
    M = int(packed.codebook_sum.shape[0])
    del packed, linear, w
    torch.cuda.empty_cache()
    return ids, M


def pairwise_hamming_sample(ids: torch.Tensor, n_pairs: int = 1024) -> dict:
    N, K = ids.shape
    g = torch.Generator(device=ids.device).manual_seed(0)
    a = torch.randint(0, N, (n_pairs,), generator=g, device=ids.device)
    b = torch.randint(0, N, (n_pairs,), generator=g, device=ids.device)
    same = (ids[a] == ids[b]).sum(dim=1).float()  # [n_pairs]
    sim = (same / K).cpu()
    return {
        "mean_sim": sim.mean().item(),
        "p50_sim": sim.median().item(),
        "p90_sim": sim.quantile(0.90).item(),
        "p99_sim": sim.quantile(0.99).item(),
    }


def kmeans_hamming(ids: torch.Tensor, G: int, iters: int = 10, batch_n: int = 1024) -> dict:
    """Mini-batch k-means in Hamming space.

    Centroids are int32[G, K] (mode of assigned rows per coordinate).
    Distance is Hamming = sum(ids != centroid) per coord.
    """
    N, K = ids.shape
    g = torch.Generator(device=ids.device).manual_seed(42)
    init_idx = torch.randperm(N, generator=g, device=ids.device)[:G]
    centroids = ids[init_idx].clone()  # [G, K] int32

    for it in range(iters):
        # Assignment for ALL rows (in chunks to control memory).
        assign = torch.empty(N, dtype=torch.int32, device=ids.device)
        chunk = 256  # rows per chunk
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            # [chunk, G, K] bool — too big. Compute distance per centroid block.
            # Better: [chunk, K] vs [G, K] → expand → ham distance via int comparison.
            # Use compact loop over G blocks of size 64 to bound memory.
            best_dist = torch.full((e - s,), K + 1, dtype=torch.int32, device=ids.device)
            best_idx = torch.zeros((e - s,), dtype=torch.int32, device=ids.device)
            cb = 64
            for gs in range(0, G, cb):
                ge = min(gs + cb, G)
                # [chunk, cb, K] -> sum over K of (rows[:,None,:] != cents[None,:,:])
                diff = (ids[s:e, None, :] != centroids[None, gs:ge, :]).sum(dim=-1).to(torch.int32)
                block_min, block_arg = diff.min(dim=1)
                better = block_min < best_dist
                best_dist = torch.where(better, block_min, best_dist)
                best_idx = torch.where(better, block_arg.to(torch.int32) + gs, best_idx)
            assign[s:e] = best_idx

        # Update centroids: mode per coordinate per cluster (approximate via random sample).
        new_centroids = centroids.clone()
        for ci in range(G):
            members = (assign == ci).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                # reseed empty cluster from a far row
                new_centroids[ci] = ids[torch.randint(0, N, (1,), device=ids.device)]
                continue
            # mode along dim=0: bincount per column. Use cheap approximation: median.
            sample = ids[members]
            # Per-column mode via torch.mode (slow but correct).
            new_centroids[ci] = sample.mode(dim=0).values.to(torch.int32)
        # Convergence check.
        moved = (new_centroids != centroids).any(dim=1).sum().item()
        centroids = new_centroids
        if moved == 0:
            break

    # Final assignment & stats.
    final_dist = torch.empty(N, dtype=torch.int32, device=ids.device)
    final_assign = torch.empty(N, dtype=torch.int32, device=ids.device)
    for s in range(0, N, 256):
        e = min(s + 256, N)
        best_d = torch.full((e - s,), K + 1, dtype=torch.int32, device=ids.device)
        best_i = torch.zeros((e - s,), dtype=torch.int32, device=ids.device)
        cb = 64
        for gs in range(0, G, cb):
            ge = min(gs + cb, G)
            diff = (ids[s:e, None, :] != centroids[None, gs:ge, :]).sum(dim=-1).to(torch.int32)
            block_min, block_arg = diff.min(dim=1)
            better = block_min < best_d
            best_d = torch.where(better, block_min, best_d)
            best_i = torch.where(better, block_arg.to(torch.int32) + gs, best_i)
        final_dist[s:e] = best_d
        final_assign[s:e] = best_i

    delta_density = final_dist.float() / K  # fraction of mismatched coords per row
    return {
        "G": G,
        "delta_density_mean": delta_density.mean().item(),
        "delta_density_p50": delta_density.median().item(),
        "delta_density_p90": delta_density.quantile(0.90).item(),
        "delta_density_p99": delta_density.quantile(0.99).item(),
        "delta_density_max": delta_density.max().item(),
        "iters_run": it + 1,
        "_centroids": centroids,
        "_assign": final_assign,
        "_dist": final_dist,
    }


def vram_estimates(N: int, K: int, G: int, mean_density: float) -> dict:
    """Storage cost in MB, comparing baseline vs row-archetype with dense / sparse delta."""
    bytes_per_id = 2  # int16
    baseline = N * K * bytes_per_id
    arch = G * K * bytes_per_id
    assign = N * 1  # 1 byte for G<=256, 2 bytes else
    if G > 256:
        assign = N * 2
    delta_dense = N * K * bytes_per_id  # if we kept full dense delta — same as baseline
    delta_sparse_exact = int(N * K * mean_density) * (bytes_per_id + 4)  # value + (row,col) coord
    # Compact COO: per-row var-length list of (col_uint16, val_int16).
    delta_sparse_compact = int(N * K * mean_density) * (2 + 2)  # col + val
    return {
        "baseline_MB": baseline / 1e6,
        "archetype_MB": arch / 1e6,
        "assign_MB": assign / 1e6,
        "delta_compact_MB": delta_sparse_compact / 1e6,
        "total_MB": (arch + assign + delta_sparse_compact) / 1e6,
        "ratio_vs_baseline": (arch + assign + delta_sparse_compact) / baseline,
    }


def probe_layer(name: str) -> dict:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    ids, M = encode_layer(name)
    N, K = ids.shape
    print(f"  encoded: ids {tuple(ids.shape)} {ids.dtype}, M={M}, took {time.time()-t0:.1f}s", flush=True)

    pw = pairwise_hamming_sample(ids)
    print(f"  pairwise sim: mean={pw['mean_sim']:.4f} p50={pw['p50_sim']:.4f} "
          f"p90={pw['p90_sim']:.4f} p99={pw['p99_sim']:.4f}", flush=True)

    out = {"name": name, "N": N, "K": K, "M": M, "pairwise": pw, "kmeans": {}, "vram": {}}
    for G in (32, 128, 512):
        t1 = time.time()
        # Cap iters small — we just need a signal.
        km = kmeans_hamming(ids, G=G, iters=4)
        dt = time.time() - t1
        # Strip large tensors before saving.
        km_sm = {k: v for k, v in km.items() if not k.startswith("_")}
        print(f"  G={G:4d}: delta_density mean={km_sm['delta_density_mean']:.4f} "
              f"p50={km_sm['delta_density_p50']:.4f} p90={km_sm['delta_density_p90']:.4f} "
              f"p99={km_sm['delta_density_p99']:.4f}  ({dt:.1f}s, {km_sm['iters_run']} iters)",
              flush=True)
        vram = vram_estimates(N, K, G, km_sm["delta_density_mean"])
        print(f"          VRAM: baseline={vram['baseline_MB']:.1f}MB  "
              f"arch={vram['archetype_MB']:.2f}MB  delta={vram['delta_compact_MB']:.1f}MB  "
              f"total={vram['total_MB']:.1f}MB  ratio={vram['ratio_vs_baseline']:.3f}",
              flush=True)
        out["kmeans"][str(G)] = km_sm
        out["vram"][str(G)] = vram

    del ids
    torch.cuda.empty_cache()
    return out


def main() -> None:
    layers = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.40.self_attn.q_proj.weight",
        "model.layers.79.mlp.down_proj.weight",
    ]
    results = []
    for name in layers:
        try:
            results.append(probe_layer(name))
        except Exception as e:
            print(f"  FAILED {name}: {e}", flush=True)
            results.append({"name": name, "error": str(e)})

    out_path = Path("/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m9a_row_archetype_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: ERROR")
            continue
        best_g = min(r["vram"], key=lambda g: r["vram"][g]["ratio_vs_baseline"])
        v = r["vram"][best_g]
        k = r["kmeans"][best_g]
        print(f"  {r['name']:55s}  best G={best_g}  "
              f"delta_density={k['delta_density_mean']:.3f}  "
              f"VRAM ratio={v['ratio_vs_baseline']:.3f}  "
              f"({v['baseline_MB']:.1f}->{v['total_MB']:.1f} MB)")


if __name__ == "__main__":
    main()
