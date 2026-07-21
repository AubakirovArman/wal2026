"""Find a safe Q4-to-ternary fraction inside an accepted mixed artifact."""
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
from wal_tat import install_mixed_q2_q4_artifact, valid_group_weight_count
from wal_tat.scoring import exact_diagonal_ternary_project


PROJECTION_MAP = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
}


def parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(
        sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    )
    if not fractions or any(item <= 0 or item > 1 for item in fractions):
        raise ValueError("fractions must be in (0, 1]")
    return fractions


def global_lowest_masks(
    damage: dict[str, torch.Tensor],
    eligible: dict[str, torch.Tensor],
    count: int,
) -> dict[str, torch.Tensor]:
    """Select exactly ``count`` eligible groups with the lowest damage."""
    names = tuple(damage)
    if not names or set(eligible) != set(names):
        raise ValueError("damage and eligible maps must have the same keys")
    values = []
    locations = []
    for matrix_index, name in enumerate(names):
        if damage[name].shape != eligible[name].shape:
            raise ValueError(f"shape mismatch for {name}")
        flat_indices = torch.where(eligible[name].reshape(-1))[0]
        values.append(damage[name].reshape(-1).index_select(0, flat_indices))
        locations.extend(
            (matrix_index, int(index)) for index in flat_indices.tolist()
        )
    joined = torch.cat(values) if values else torch.empty(0)
    if not 0 <= count <= joined.numel():
        raise ValueError("selection count is outside eligible groups")
    result = {name: torch.zeros_like(eligible[name]) for name in names}
    if count == 0:
        return result
    chosen = torch.topk(joined, count, largest=False, sorted=False).indices.tolist()
    for joined_index in chosen:
        matrix_index, flat_index = locations[joined_index]
        result[names[matrix_index]].view(-1)[flat_index] = True
    return result


def grouped_moment(moment: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    rows, columns = shape
    padding = (-columns) % 128
    if padding:
        moment = F.pad(moment, (0, padding))
    return moment.float().view(1, -1, 128).expand(rows, -1, -1)


def build_candidate(
    parent: dict,
    masks: dict[str, torch.Tensor],
    ternary_codes: dict[str, torch.Tensor],
    ternary_scales: dict[str, torch.Tensor],
) -> dict:
    candidate = dict(parent)
    candidate["matrices"] = dict(parent["matrices"])
    for name, selected in masks.items():
        entry = dict(parent["matrices"][name])
        if torch.logical_and(selected, ~entry["q4_mask"].bool()).any():
            raise ValueError(f"selection leaves Q4 eligibility in {name}")
        entry["q2_codes_int8"] = entry["q2_codes_int8"].clone()
        entry["q2_scales_fp16"] = entry["q2_scales_fp16"].clone()
        entry["q4_mask"] = entry["q4_mask"].bool().clone()
        entry["q2_codes_int8"][selected] = ternary_codes[name][selected]
        entry["q2_scales_fp16"][selected] = ternary_scales[name][selected].half()
        entry["q4_mask"][selected] = False
        candidate["matrices"][name] = entry
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument(
        "--target-projections",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
    )
    parser.add_argument("--fractions", default="0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--moment-sequences", type=int, default=64)
    parser.add_argument("--selection-gate-sequences", type=int, default=64)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.005)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--seed", type=int, default=839)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


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
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    if not 1 <= args.moment_sequences <= len(calibration):
        raise ValueError("moment sequences are outside calibration")
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
    names = tuple(
        f"model.layers.{args.target_layer}.{PROJECTION_MAP[item]}"
        for item in requested
    )
    if any(name not in parent_artifact["matrices"] for name in names):
        raise ValueError("parent artifact lacks a target matrix")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    install_checkpoint(model, source, args.device)
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
        model,
        names,
        calibration,
        args.device,
        args.moment_sequences,
    )

    ternary_codes = {}
    ternary_scales = {}
    damage = {}
    eligible = {}
    total_eligible_groups = 0
    total_eligible_weights = 0
    for name in names:
        entry = parent_artifact["matrices"][name]
        weight = model.get_submodule(name).weight.detach().float()
        codes, scales, _ = exact_diagonal_ternary_project(
            weight, moments[name], group_size=128
        )
        rows, columns = map(int, entry["shape"])
        padding = (-columns) % 128
        grouped = F.pad(weight, (0, padding)).view(rows, -1, 128)
        reconstructed = codes.float() * scales.unsqueeze(-1)
        local_damage = (
            grouped_moment(moments[name], (rows, columns))
            * (grouped - reconstructed).square()
        ).sum(-1)
        q4_mask = entry["q4_mask"].bool().to(local_damage.device)
        ternary_codes[name] = codes.cpu()
        ternary_scales[name] = scales.cpu()
        damage[name] = local_damage.cpu()
        eligible[name] = q4_mask.cpu()
        total_eligible_groups += int(q4_mask.sum())
        total_eligible_weights += valid_group_weight_count(q4_mask.cpu(), columns, 128)

    if total_eligible_groups == 0:
        raise ValueError("target contains no Q4 groups")
    attempts = []
    selection_candidates = []
    for fraction in fractions:
        count = max(1, min(total_eligible_groups, round(total_eligible_groups * fraction)))
        masks = global_lowest_masks(damage, eligible, count)
        candidate = build_candidate(
            parent_artifact, masks, ternary_codes, ternary_scales
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
            selection_candidates.append(
                {"count": count, "masks": masks, "candidate": candidate, **attempt}
            )

    final_metrics = None
    final_ratios = None
    final_incremental = None
    final_passed = False
    artifact_path = None
    artifact_sha256 = None
    accepted = None
    full_evaluations = []
    for selection in sorted(
        selection_candidates, key=lambda item: item["count"], reverse=True
    ):
        install_mixed_q2_q4_artifact(
            model,
            selection["candidate"],
            source,
            device=args.device,
            expected_source_sha256=source_hash,
        )
        metrics = evaluate_domains(model, gates, args.device)
        absolute = ratios(metrics, baseline)
        incremental = ratios(metrics, parent)
        passed = all(
            value <= args.gate_ratio for value in absolute.values()
        ) and all(
            value <= args.incremental_gate_ratio
            for value in incremental.values()
        )
        full_evaluations.append(
            {
                "fraction": selection["fraction"],
                "selected_groups": selection["selected_groups"],
                "selected_weights": selection["selected_weights"],
                "metrics": metrics,
                "ratios": absolute,
                "incremental_ratios_vs_parent": incremental,
                "passed": passed,
            }
        )
        print(
            f"full fraction={selection['fraction']:.6f} passed={passed} "
            f"ratios={absolute} incremental={incremental}",
            flush=True,
        )
        if passed:
            accepted = selection
            final_metrics = metrics
            final_ratios = absolute
            final_incremental = incremental
            final_passed = True
            break

    if accepted is not None and args.write_artifact:
        artifact = accepted["candidate"]
        artifact["parent_artifact_sha256"] = sha256_file(parent_path)
        artifact["reverse_q4_to_q2"] = {
            "selected_groups": accepted["selected_groups"],
            "selected_weights": accepted["selected_weights"],
            "development_ratios": final_ratios,
            "development_incremental_ratios_vs_parent": final_incremental,
        }
        artifact_path = (
            source_path.parents[1]
            / "artifacts"
            / f"wal-tat-{args.tag}-mixed-q2-q4-q8.pt"
        )
        torch.save(artifact, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-reverse-q4-to-q2-fraction-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": sha256_file(parent_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_layer": args.target_layer,
        "target_projections": requested,
        "fractions": fractions,
        "moment_sequences": args.moment_sequences,
        "selection_gate_sequences": args.selection_gate_sequences,
        "total_eligible_groups": total_eligible_groups,
        "total_eligible_weights": total_eligible_weights,
        "baseline": baseline,
        "parent": parent,
        "parent_ratios": ratios(parent, baseline),
        "attempts": attempts,
        "largest_selection_pass": (
            {
                key: value
                for key, value in max(
                    selection_candidates, key=lambda item: item["count"]
                ).items()
                if key not in {"masks", "candidate"}
            }
            if selection_candidates
            else None
        ),
        "full_evaluations": full_evaluations,
        "accepted_selection": (
            {
                key: value
                for key, value in accepted.items()
                if key not in {"masks", "candidate"}
            }
            if accepted is not None
            else None
        ),
        "final_metrics": final_metrics,
        "final_ratios": final_ratios,
        "final_incremental_ratios_vs_parent": final_incremental,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "passed": final_passed,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
        "checkpoint_written": False,
        "sealed_audit_opened": False,
        "elapsed_seconds": time.time() - started,
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}; passed={final_passed} artifact={artifact_path} "
        f"ratios={final_ratios}",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
