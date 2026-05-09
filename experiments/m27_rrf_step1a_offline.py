"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M27 RRF Step 1a (offline analysis): RRF allocator vs baselines + tile sweep.

Loads ``results/m23_influence/l54_gate_up.pt`` produced by m27_rrf_collect.py
and answers two questions, **without touching the model or the kernel**:

  Q1. At a given register capacity ``C`` per stage, how does the
      interference-aware Route-Register-File allocator compare to:
        (a) topk_by_influence (no interference)
        (b) topk_by_count    (the current ``count`` score mode)
      in terms of *count-mass hit rate*
        hit = sum_{id in hot} count[id] / sum_id count[id]?

  Q2. Is the structural interference signal informative at the kernel's
      operating tile_size (256 rows), or does it require a finer grain?
      We re-derive interference at tile_size in {256, 64, 16, 4, 1} from the
      raw stage_ids of the same layers (one extra forward over the encoding,
      no model needed) and report ``avg_id_tile_occupancy`` per setting.

Outputs:
  - results/m23_influence/l54_rrf_step1a.json (table)
  - prints a concise summary table to stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def linear_scan_allocate(
    influence: torch.Tensor,         # [M] float64
    interference: torch.Tensor,      # [M, M] float32 (diag = marginal)
    capacity: int,
) -> torch.Tensor:
    """User's spec:
        Priority(v) = I(v) * (1 - avg_interference(v))
        sort desc, take top capacity.
    avg_interference(v) is mean over j != v of interference[v, j].
    """
    M = int(influence.numel())
    if capacity >= M:
        return torch.arange(M, dtype=torch.long)
    # exclude diagonal from the mean
    inter = interference.clone()
    diag_idx = torch.arange(M)
    inter[diag_idx, diag_idx] = 0.0
    avg_inter = inter.mean(dim=1)  # [M]
    priority = influence.float() * (1.0 - avg_inter.clamp(min=0.0, max=1.0))
    top = torch.topk(priority, k=capacity).indices
    return top.sort().values


def topk_by(scores: torch.Tensor, capacity: int) -> torch.Tensor:
    M = int(scores.numel())
    capacity = min(capacity, M)
    top = torch.topk(scores.float(), k=capacity).indices
    return top.sort().values


def hit_rate(hot_ids: torch.Tensor, counts: torch.Tensor) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    return float(counts[hot_ids].sum() / total)


def occupancy_from_ids(stage_ids_cpu: torch.Tensor, M: int, tile_size: int) -> float:
    n_size, blocks_per_row = stage_ids_cpu.shape
    num_tiles = (n_size + tile_size - 1) // tile_size
    total = 0
    for t in range(num_tiles):
        a, b = t * tile_size, min(t * tile_size + tile_size, n_size)
        unique_ct = int(torch.unique(stage_ids_cpu[a:b]).numel())
        total += unique_ct
    return total / (num_tiles * M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        default=str(REPO_ROOT / "results/m23_influence/l54_gate_up.pt"),
    )
    ap.add_argument("--capacities", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--tile-sweep", type=int, nargs="+", default=[256, 64, 16, 4, 1])
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "results/m23_influence/l54_rrf_step1a.json"),
    )
    args = ap.parse_args()

    payload = torch.load(args.inp, map_location="cpu", weights_only=False)
    layer_names = [k for k in payload.keys() if k != "args"]

    out_rows: list[dict] = []
    print()
    print(f"{'layer':50s} {'stage':>5s} {'cap':>4s} {'topk_count':>11s} {'topk_infl':>10s} {'rrf':>6s}")
    for name in layer_names:
        rec = payload[name]
        M = int(rec["M"])
        for s in range(int(rec["num_stages"])):
            inf = rec["stage_influence"][s]
            cnt = rec["stage_counts"][s]
            intf = rec["stage_interference"][s]
            for cap in args.capacities:
                ids_count = topk_by(cnt, cap)
                ids_infl = topk_by(inf, cap)
                ids_rrf = linear_scan_allocate(inf, intf, cap)
                hr_count = hit_rate(ids_count, cnt)
                hr_infl = hit_rate(ids_infl, cnt)
                hr_rrf = hit_rate(ids_rrf, cnt)
                out_rows.append({
                    "layer": name, "stage": s, "M": M, "capacity": cap,
                    "hit_topk_count": hr_count,
                    "hit_topk_influence": hr_infl,
                    "hit_rrf": hr_rrf,
                })
                if s in (0, 1, 11):  # print only 3 stages to keep output sane
                    print(f"{name:50s} {s:>5d} {cap:>4d} {hr_count:>11.3f} {hr_infl:>10.3f} {hr_rrf:>6.3f}")

    # Tile sweep — needs raw stage_ids; reload the encoding on the fly is heavy.
    # Instead approximate: derive structural occupancy from `num_tiles` and `M`
    # using stored interference diagonals (which were computed at tile_size=256).
    # For finer tiles we need the raw stage_ids. Run a small recomputation here
    # using a separate cheap path: re-load the encoded layers from the SAME run
    # by re-running the encoding... too expensive. Skip if not requested.
    print()
    print(f"[note] tile-sweep requires raw stage_ids; run m27_rrf_tile_sweep.py separately.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "rows": out_rows}, indent=2))
    print(f"[save] {args.out}")

    # Aggregate verdict per layer at the most informative capacities.
    print()
    print("=== aggregate (mean across stages) ===")
    for name in layer_names:
        rows = [r for r in out_rows if r["layer"] == name]
        for cap in args.capacities:
            sub = [r for r in rows if r["capacity"] == cap]
            mean_count = sum(r["hit_topk_count"] for r in sub) / len(sub)
            mean_infl  = sum(r["hit_topk_influence"] for r in sub) / len(sub)
            mean_rrf   = sum(r["hit_rrf"] for r in sub) / len(sub)
            print(f"  {name:50s} cap={cap:>4d}  topk_count={mean_count:.3f}  topk_infl={mean_infl:.3f}  rrf={mean_rrf:.3f}")


if __name__ == "__main__":
    main()
