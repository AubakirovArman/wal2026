"""Move the safest decoder Q4/Q8 groups to physical two-bit NZ4."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from block_proxy_recovery import collect_input_second_moments
from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    sha256_file,
)
from reverse_q4_to_q2_fraction import (
    global_lowest_masks,
    grouped_moment,
    parse_fractions,
)
from reverse_q8_to_q4_fraction import PROJECTION_MAP, parse_layers
from wal_tat import (
    install_mixed_q2_q4_artifact,
    valid_group_weight_count,
    weighted_symmetric_nz4_project,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-precision", choices=("q4", "q8"), required=True)
    parser.add_argument(
        "--projection-source",
        choices=("parent", "strict-source"),
        default="parent",
        help=(
            "project NZ4 codes from reconstructed parent Q4/Q8 weights or "
            "directly from the strict-source/BF16 weights captured before "
            "the mixed artifact is installed"
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-layer", type=int)
    target.add_argument("--target-layers")
    parser.add_argument(
        "--target-projections",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
    )
    parser.add_argument(
        "--fractions", default="0.001,0.0025,0.005,0.01,0.025,0.05"
    )
    parser.add_argument("--moment-sequences", type=int, default=64)
    parser.add_argument("--selection-gate-sequences", type=int, default=16)
    parser.add_argument("--gate-ratio", type=float, default=1.1)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.1)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_candidate(
    parent: dict,
    masks: dict[str, torch.Tensor],
    nz4_codes: dict[str, torch.Tensor],
    nz4_scales: dict[str, torch.Tensor],
    *,
    source_precision: str,
) -> dict:
    candidate = dict(parent)
    candidate["matrices"] = dict(parent["matrices"])
    source_mask_name = f"{source_precision}_mask"
    for name, selected in masks.items():
        entry = dict(parent["matrices"][name])
        source_mask = entry[source_mask_name].bool()
        if torch.logical_and(selected, ~source_mask).any():
            raise ValueError(f"selection leaves {source_precision.upper()} eligibility in {name}")
        entry["q2_codes_int8"] = entry["q2_codes_int8"].clone()
        entry["q2_scales_fp16"] = entry["q2_scales_fp16"].clone()
        entry["q4_mask"] = entry["q4_mask"].bool().clone()
        entry["q8_mask"] = entry["q8_mask"].bool().clone()
        entry["nz4_mask"] = entry.get(
            "nz4_mask", torch.zeros_like(entry["q4_mask"])
        ).bool().clone()
        entry["q2_codes_int8"][selected] = nz4_codes[name][selected].to(
            entry["q2_codes_int8"].dtype
        )
        entry["q2_scales_fp16"][selected] = nz4_scales[name][selected].to(
            entry["q2_scales_fp16"].dtype
        )
        entry[source_mask_name][selected] = False
        entry["nz4_mask"][selected] = True
        candidate["matrices"][name] = entry
    return candidate


def snapshot_projection_weights(
    model: torch.nn.Module, names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Freeze source weights before the mixed parent mutates model modules."""
    return {
        name: model.get_submodule(name).weight.detach().cpu().clone()
        for name in names
    }


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.fractions)
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(source_path)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    parent_artifact = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent_artifact.get("format") != "wal-tat-mixed-q2-q4-q8-v1":
        raise ValueError("parent artifact must use the mixed Q2/Q4/Q8 format")
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    selection_gates = {
        domain: chunks[: args.selection_gate_sequences]
        for domain, chunks in gates.items()
    }
    if any(
        len(chunks) != args.selection_gate_sequences
        for chunks in selection_gates.values()
    ):
        raise ValueError("suite lacks requested selection sequences")
    requested = tuple(
        item.strip() for item in args.target_projections.split(",") if item.strip()
    )
    if not requested or any(item not in PROJECTION_MAP for item in requested):
        raise ValueError("invalid target projections")
    target_layers = parse_layers(args.target_layer, args.target_layers)
    requested_names = tuple(
        f"model.layers.{layer}.{PROJECTION_MAP[item]}"
        for layer in target_layers
        for item in requested
    )
    mask_name = f"{args.source_precision}_mask"
    names = tuple(
        name
        for name in requested_names
        if name in parent_artifact["matrices"]
        and bool(parent_artifact["matrices"][name][mask_name].bool().any())
    )
    if not names:
        raise ValueError("requested decoder scope contains no eligible groups")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    install_checkpoint(model, source, args.device)
    source_projection_weights = (
        snapshot_projection_weights(model, names)
        if args.projection_source == "strict-source"
        else None
    )
    install_mixed_q2_q4_artifact(
        model,
        parent_artifact,
        source,
        device=args.device,
        expected_source_sha256=source_hash,
    )
    parent = evaluate_domains(model, gates, args.device)
    parent_selection = evaluate_domains(model, selection_gates, args.device)
    moments = collect_input_second_moments(
        model, names, calibration, args.device, args.moment_sequences
    )

    nz4_codes = {}
    nz4_scales = {}
    damage = {}
    eligible = {}
    total_eligible_groups = 0
    total_eligible_weights = 0
    for name in names:
        entry = parent_artifact["matrices"][name]
        if source_projection_weights is None:
            weight = model.get_submodule(name).weight.detach().float()
        else:
            weight = source_projection_weights[name].to(args.device).float()
        codes, scales, error = weighted_symmetric_nz4_project(
            weight, moments[name], group_size=128
        )
        rows, columns = map(int, entry["shape"])
        if len(target_layers) > 1:
            padding = (-columns) % 128
            grouped = F.pad(weight, (0, padding)).view(rows, -1, 128)
            row_energy = (
                grouped_moment(moments[name], (rows, columns)) * grouped.square()
            ).sum((-1, -2)).clamp_min(1e-12)
            error = error / row_energy.unsqueeze(-1)
        local_eligible = entry[mask_name].bool().cpu()
        nz4_codes[name] = codes.cpu()
        nz4_scales[name] = scales.half().cpu()
        damage[name] = error.cpu()
        eligible[name] = local_eligible
        total_eligible_groups += int(local_eligible.sum())
        total_eligible_weights += valid_group_weight_count(
            local_eligible, columns, 128
        )

    attempts = []
    passing = []
    for fraction in fractions:
        count = max(1, min(total_eligible_groups, round(total_eligible_groups * fraction)))
        masks = global_lowest_masks(damage, eligible, count)
        candidate = build_candidate(
            parent_artifact,
            masks,
            nz4_codes,
            nz4_scales,
            source_precision=args.source_precision,
        )
        install_mixed_q2_q4_artifact(
            model,
            candidate,
            source,
            device=args.device,
            expected_source_sha256=source_hash,
        )
        metrics = evaluate_domains(model, selection_gates, args.device)
        absolute = ratios(metrics, selection_baseline)
        incremental = ratios(metrics, parent_selection)
        passed = all(value <= args.gate_ratio for value in absolute.values()) and all(
            value <= args.incremental_gate_ratio for value in incremental.values()
        )
        selected_weights = sum(
            valid_group_weight_count(
                masks[name], int(parent_artifact["matrices"][name]["shape"][1]), 128
            )
            for name in names
        )
        attempt = {
            "fraction": fraction,
            "selected_groups": count,
            "selected_weights": selected_weights,
            "selection_metrics": metrics,
            "selection_ratios": absolute,
            "selection_incremental_ratios_vs_parent": incremental,
            "passed": passed,
        }
        attempts.append(attempt)
        print(
            f"fraction={fraction:.6f} groups={count} weights={selected_weights} "
            f"passed={passed} ratios={absolute} incremental={incremental}",
            flush=True,
        )
        if passed:
            passing.append({"count": count, "masks": masks, **attempt})

    accepted = None
    final_candidate = None
    full_evaluations = []
    for selection in sorted(passing, key=lambda item: item["count"], reverse=True):
        candidate = build_candidate(
            parent_artifact,
            selection["masks"],
            nz4_codes,
            nz4_scales,
            source_precision=args.source_precision,
        )
        install_mixed_q2_q4_artifact(
            model,
            candidate,
            source,
            device=args.device,
            expected_source_sha256=source_hash,
        )
        metrics = evaluate_domains(model, gates, args.device)
        absolute = ratios(metrics, baseline)
        incremental = ratios(metrics, parent)
        passed = all(value <= args.gate_ratio for value in absolute.values()) and all(
            value <= args.incremental_gate_ratio for value in incremental.values()
        )
        evaluation = {
            "fraction": selection["fraction"],
            "selected_groups": selection["selected_groups"],
            "selected_weights": selection["selected_weights"],
            "metrics": metrics,
            "ratios": absolute,
            "incremental_ratios_vs_parent": incremental,
            "passed": passed,
        }
        full_evaluations.append(evaluation)
        print(
            f"full fraction={selection['fraction']:.6f} passed={passed} "
            f"ratios={absolute} incremental={incremental}",
            flush=True,
        )
        if passed:
            accepted = evaluation
            final_candidate = candidate
            break

    artifact_path = None
    artifact_sha256 = None
    if accepted is not None and args.write_artifact:
        final_candidate["parent_artifact_sha256"] = sha256_file(parent_path)
        final_candidate["reverse_high_precision_to_nz4"] = {
            "source_precision": args.source_precision,
            "projection_source": args.projection_source,
            **accepted,
        }
        artifact_path = (
            source_path.parents[1]
            / "artifacts"
            / f"wal-tat-{args.tag}-mixed-q2-q4-q8.pt"
        )
        torch.save(final_candidate, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-reverse-high-precision-to-nz4-fraction-v1",
        "tag": args.tag,
        "source_precision": args.source_precision,
        "projection_source": args.projection_source,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": sha256_file(parent_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_layers": target_layers,
        "target_projections": requested,
        "eligible_matrices": names,
        "total_eligible_groups": total_eligible_groups,
        "total_eligible_weights": total_eligible_weights,
        "fractions": fractions,
        "moment_sequences": args.moment_sequences,
        "selection_gate_sequences": args.selection_gate_sequences,
        "baseline": baseline,
        "parent": parent,
        "parent_ratios": ratios(parent, baseline),
        "attempts": attempts,
        "full_evaluations": full_evaluations,
        "accepted": accepted,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "passed": accepted is not None,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}; passed={result['passed']} accepted={accepted} "
        f"artifact={artifact_path}",
        flush=True,
    )
    del model, source, parent_artifact, nz4_codes, nz4_scales
    if source_projection_weights is not None:
        del source_projection_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
