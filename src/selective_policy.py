from __future__ import annotations

from collections import defaultdict


def build_selective_policy(
    records: list[dict[str, object]],
    max_output_mse: float = 1e-3,
    min_local_vs_global: float = 1.2,
) -> dict[str, object]:
    approved: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    family_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"approved": 0, "rejected": 0})
    for record in records:
        mse = float(record["distilled_output_mse"])
        speed = float(record["local_vs_global_triton"])
        family = str(record["family"])
        item = {
            "tensor_name": record["tensor_name"],
            "family": family,
            "layer_idx": int(record["layer_idx"]),
            "distilled_output_mse": mse,
            "local_vs_global_triton": speed,
            "base_unique": int(record["base_unique"]),
            "decision": "local_palette_32" if mse <= max_output_mse and speed >= min_local_vs_global else "global_id_triton",
        }
        if item["decision"] == "local_palette_32":
            approved.append(item)
            family_stats[family]["approved"] += 1
        else:
            reasons = []
            if mse > max_output_mse:
                reasons.append("mse")
            if speed < min_local_vs_global:
                reasons.append("speed")
            item["reasons"] = reasons
            rejected.append(item)
            family_stats[family]["rejected"] += 1
    summary = {
        family: {
            **stats,
            "approval_rate": stats["approved"] / max(stats["approved"] + stats["rejected"], 1),
        }
        for family, stats in family_stats.items()
    }
    return {
        "criteria": {
            "max_output_mse": max_output_mse,
            "min_local_vs_global_triton": min_local_vs_global,
        },
        "approved": sorted(approved, key=lambda item: (item["family"], item["layer_idx"])),
        "rejected": sorted(rejected, key=lambda item: (item["family"], item["layer_idx"])),
        "summary": summary,
    }


def build_shape_runtime_policy(
    policy: dict[str, object],
    frontier_records: list[dict[str, object]],
    max_grouped_local_mse: float = 1e-4,
    min_local_vs_global_full: float = 1.0,
) -> dict[str, object]:
    approved = {
        str(item["tensor_name"]): item
        for item in policy.get("approved", [])
    }
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for record in frontier_records:
        tensor_name = str(record["tensor_name"])
        if tensor_name not in approved:
            continue
        if float(record["grouped_local_mse"]) > max_grouped_local_mse:
            continue
        if float(record["local_vs_global_full"]) < min_local_vs_global_full:
            continue
        key = (tensor_name, int(record["group_rows"]), int(record["group_cols"]))
        grouped[key].append(record)

    selected: list[dict[str, object]] = []
    unassigned: list[dict[str, object]] = []
    for tensor_name, item in approved.items():
        candidates: list[dict[str, object]] = []
        for (candidate_name, group_rows, group_cols), records in grouped.items():
            if candidate_name != tensor_name:
                continue
            count = len(records)
            avg_ms_local = sum(float(record["ms_grouped_local"]) for record in records) / count
            avg_ms_global_full = sum(float(record["ms_global_full"]) for record in records) / count
            avg_ms_grouped_global = sum(float(record["ms_grouped_global"]) for record in records) / count
            avg_local_vs_global_full = sum(float(record["local_vs_global_full"]) for record in records) / count
            avg_local_vs_grouped_global = sum(float(record["local_vs_grouped_global"]) for record in records) / count
            avg_grouped_local_mse = sum(float(record["grouped_local_mse"]) for record in records) / count
            avg_launches = sum(int(record["launches"]) for record in records) / count
            avg_mean_unique = sum(float(record["mean_group_unique"]) for record in records) / count
            avg_total_unique = sum(float(record["total_group_unique"]) for record in records) / count
            avg_group_area = sum(int(record["group_area"]) for record in records) / count
            candidates.append(
                {
                    "tensor_name": tensor_name,
                    "family": item["family"],
                    "layer_idx": int(item["layer_idx"]),
                    "base_unique": int(item["base_unique"]),
                    "distilled_output_mse": float(item["distilled_output_mse"]),
                    "local_vs_global_triton": float(item["local_vs_global_triton"]),
                    "decision": "local_palette_grouped",
                    "group_rows": group_rows,
                    "group_cols": group_cols,
                    "avg_launches": avg_launches,
                    "avg_mean_group_unique": avg_mean_unique,
                    "avg_total_group_unique": avg_total_unique,
                    "avg_group_area": avg_group_area,
                    "avg_ms_grouped_local": avg_ms_local,
                    "avg_ms_grouped_global": avg_ms_grouped_global,
                    "avg_ms_global_full": avg_ms_global_full,
                    "avg_local_vs_grouped_global": avg_local_vs_grouped_global,
                    "avg_local_vs_global_full": avg_local_vs_global_full,
                    "avg_grouped_local_mse": avg_grouped_local_mse,
                    "num_measurements": count,
                }
            )
        if not candidates:
            unassigned.append(
                {
                    "tensor_name": tensor_name,
                    "family": item["family"],
                    "layer_idx": int(item["layer_idx"]),
                    "reason": "no_runtime_candidate",
                }
            )
            continue
        candidates.sort(
            key=lambda candidate: (
                candidate["avg_ms_grouped_local"],
                -candidate["avg_local_vs_global_full"],
                candidate["avg_launches"],
            )
        )
        selected.append(candidates[0])

    summary = {
        "selected": len(selected),
        "unassigned": len(unassigned),
        "mean_selected_local_vs_global_full": (
            sum(item["avg_local_vs_global_full"] for item in selected) / len(selected)
            if selected
            else 0.0
        ),
        "mean_selected_ms_grouped_local": (
            sum(item["avg_ms_grouped_local"] for item in selected) / len(selected)
            if selected
            else 0.0
        ),
    }
    return {
        "criteria": {
            "max_grouped_local_mse": max_grouped_local_mse,
            "min_local_vs_global_full": min_local_vs_global_full,
            "selection_metric": "min_avg_ms_grouped_local",
        },
        "selected": sorted(selected, key=lambda item: (item["family"], item["layer_idx"])),
        "unassigned": sorted(unassigned, key=lambda item: (item["family"], item["layer_idx"])),
        "summary": summary,
    }