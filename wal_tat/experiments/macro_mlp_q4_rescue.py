"""Measure the minimum Q4-g128 rescue needed by a full ternary block stage.

This is a checkpoint-neutral rate--distortion diagnostic.  Previously accepted
groups remain strict ternary.  Every uncommitted group of the requested stage is
first projected to activation-aware ternary, then the groups with the largest
estimated Q4 benefit are selectively represented by signed INT4 plus one FP16
scale.  The source checkpoint is never modified or published by this script.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from block_proxy_recovery import collect_input_second_moments
from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    ensure_candidate_matrix,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    set_submodule,
    sha256_file,
)
from wal_tat import (
    install_mixed_q2_q4_artifact,
    q2_g128_physical_bpw,
    q4_g128_physical_bpw,
    q8_g128_physical_bpw,
)
from wal_tat.quantization import weighted_symmetric_q4_project
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


class FixedWeightLinear(nn.Module):
    """Evaluation-only linear layer backed by one frozen reconstructed weight."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight.to(value.dtype), self.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--parent-artifact",
        type=Path,
        help="optional consolidated mixed Q2/Q4 parent to extend",
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument(
        "--target-projections", default="up_proj,gate_proj,down_proj"
    )
    parser.add_argument("--moment-sequences", type=int, default=128)
    parser.add_argument("--selection-gate-sequences", type=int, default=64)
    parser.add_argument(
        "--rescue-fractions", default="0.005,0.01,0.02,0.05,0.1,0.2,0.4,1.0"
    )
    parser.add_argument(
        "--allow-zero-rescue",
        action="store_true",
        help=(
            "allow a 0 rescue fraction so the first candidate is a fully "
            "strict ternary block"
        ),
    )
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.005)
    parser.add_argument(
        "--projection-ablation",
        action="store_true",
        help="measure each projection alone at Q4 and each Q4 block with one FP projection",
    )
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="write the first full-development passing mixed Q2/Q4 artifact",
    )
    parser.add_argument("--seed", type=int, default=647)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parse_fractions(value: str, *, allow_zero: bool = False) -> tuple[float, ...]:
    fractions = tuple(
        sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    )
    outside_interval = (
        (lambda item: not 0 <= item <= 1)
        if allow_zero
        else (lambda item: not 0 < item <= 1)
    )
    if not fractions or any(outside_interval(item) for item in fractions):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"rescue fractions must be in {interval}")
    return fractions


@torch.no_grad()
def global_rescue_masks(
    benefit: dict[str, torch.Tensor],
    eligible: dict[str, torch.Tensor],
    count: int,
) -> dict[str, torch.Tensor]:
    """Select the globally largest estimated Q4 benefits."""
    names = tuple(benefit)
    if not names or set(eligible) != set(names):
        raise ValueError("benefit and eligible maps must have the same keys")
    values = []
    locations = []
    for matrix_index, name in enumerate(names):
        if benefit[name].shape != eligible[name].shape:
            raise ValueError(f"shape mismatch for {name}")
        flat_indices = torch.where(eligible[name].reshape(-1))[0]
        values.append(benefit[name].reshape(-1).index_select(0, flat_indices))
        locations.extend((matrix_index, int(index)) for index in flat_indices.tolist())
    joined = torch.cat(values)
    if not 0 <= count <= joined.numel():
        raise ValueError("rescue count is outside eligible groups")
    result = {name: torch.zeros_like(eligible[name]) for name in names}
    if count == 0:
        return result
    chosen = torch.topk(joined, count, largest=True, sorted=False).indices.tolist()
    for joined_index in chosen:
        matrix_index, flat_index = locations[joined_index]
        result[names[matrix_index]].view(-1)[flat_index] = True
    return result


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(
        args.rescue_fractions, allow_zero=args.allow_zero_rescue
    )
    model_path = (args.model_path or default_model_path()).resolve()
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve() if args.parent_artifact else None
    suite_path = args.suite.resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    if not 1 <= args.moment_sequences <= len(calibration):
        raise ValueError("moment sequence count is outside calibration")
    selection_gates = {
        domain: chunks[: args.selection_gate_sequences]
        for domain, chunks in gates.items()
    }
    if any(len(chunks) != args.selection_gate_sequences for chunks in selection_gates.values()):
        raise ValueError("suite lacks requested selection gate sequences")
    requested = tuple(
        item.strip() for item in args.target_projections.split(",") if item.strip()
    )
    if not requested or any(item not in PROJECTION_MAP for item in requested):
        raise ValueError("invalid target projections")
    names = tuple(
        f"model.layers.{args.target_layer}.{PROJECTION_MAP[item]}"
        for item in requested
    )

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    matrices = install_checkpoint(model, payload, args.device)
    parent_artifact = None
    parent_install = None
    if parent_path is not None:
        parent_artifact = torch.load(parent_path, map_location="cpu", weights_only=False)
        overlap = set(names) & set(parent_artifact.get("matrices", {}))
        if overlap:
            raise ValueError(f"target matrices already exist in parent: {sorted(overlap)}")
        parent_install = install_mixed_q2_q4_artifact(
            model,
            parent_artifact,
            payload,
            device=args.device,
            expected_source_sha256=sha256_file(source_path),
        )
    for name in names:
        ensure_candidate_matrix(model, matrices, name, args.device)
    source = evaluate_domains(model, gates, args.device)
    selection_source = evaluate_domains(model, selection_gates, args.device)
    source_ratios = ratios(source, baseline)
    moments = collect_input_second_moments(
        model, names, calibration, args.device, args.moment_sequences
    )

    ternary_grouped = {}
    q4_grouped = {}
    source_grouped = {}
    q2_codes = {}
    q2_scales = {}
    q4_codes_by_name = {}
    q4_scales_by_name = {}
    source_committed_masks = {}
    benefit = {}
    eligible = {}
    biases = {}
    total_groups = 0
    eligible_groups = 0
    for name in names:
        matrix = matrices[name]
        target = model.get_submodule(name)
        biases[name] = target.bias
        original = matrix.master_weight.detach().float().clone()
        ternary_codes, ternary_scales, ternary_error = exact_diagonal_ternary_project(
            original, moments[name], group_size=matrix.group_size
        )
        q4_codes, q4_scales, q4_error = weighted_symmetric_q4_project(
            original, moments[name], group_size=matrix.group_size
        )
        source_weight = F.pad(
            matrix.effective_weight().detach().float(), (0, matrix.padding)
        ).view_as(matrix.committed_codes)
        source_grouped[name] = source_weight
        ternary_value = source_weight.clone()
        q4_value = source_weight.clone()
        candidate_mask = ~matrix.committed_mask
        ternary_value[candidate_mask] = (
            ternary_codes.float() * ternary_scales.unsqueeze(-1)
        )[candidate_mask]
        q4_value[candidate_mask] = (
            q4_codes.float() * q4_scales.unsqueeze(-1)
        )[candidate_mask]
        ternary_grouped[name] = ternary_value
        q4_grouped[name] = q4_value
        deployed_q2_codes = matrix.committed_codes.detach().clone()
        deployed_q2_scales = matrix.group_scale.detach().clone()
        deployed_q2_codes[candidate_mask] = ternary_codes[candidate_mask]
        deployed_q2_scales[candidate_mask] = ternary_scales[candidate_mask]
        q2_codes[name] = deployed_q2_codes.cpu()
        q2_scales[name] = deployed_q2_scales.half().cpu()
        q4_codes_by_name[name] = q4_codes.cpu()
        q4_scales_by_name[name] = q4_scales.half().cpu()
        source_committed_masks[name] = matrix.committed_mask.detach().cpu().clone()
        benefit[name] = ternary_error - q4_error
        eligible[name] = candidate_mask
        total_groups += matrix.committed_mask.numel()
        eligible_groups += int(candidate_mask.sum().item())

    def install_rescue(mask_map: dict[str, torch.Tensor]) -> None:
        for name in names:
            grouped = torch.where(
                mask_map[name].unsqueeze(-1), q4_grouped[name], ternary_grouped[name]
            )
            matrix = matrices[name]
            weight = grouped.reshape(matrix.out_features, -1)[:, : matrix.in_features]
            set_submodule(
                model,
                name,
                FixedWeightLinear(weight.to(matrix.compute_dtype), biases[name]).to(
                    args.device
                ),
            )

    zero_masks = {name: torch.zeros_like(eligible[name]) for name in names}
    install_rescue(zero_masks)
    full_ternary_selection = evaluate_domains(model, selection_gates, args.device)
    full_ternary_selection_ratios = ratios(
        full_ternary_selection, selection_baseline
    )
    full_ternary = evaluate_domains(model, gates, args.device)
    full_ternary_ratios = ratios(full_ternary, baseline)

    projection_ablation = {}
    if args.projection_ablation:
        for mode in ("q4_only", "q4_except"):
            projection_ablation[mode] = {}
            for ablated_name in names:
                for name in names:
                    use_q4 = (
                        name == ablated_name
                        if mode == "q4_only"
                        else name != ablated_name
                    )
                    grouped = q4_grouped[name] if use_q4 else source_grouped[name]
                    matrix = matrices[name]
                    weight = grouped.reshape(matrix.out_features, -1)[
                        :, : matrix.in_features
                    ]
                    set_submodule(
                        model,
                        name,
                        FixedWeightLinear(
                            weight.to(matrix.compute_dtype), biases[name]
                        ).to(args.device),
                    )
                metrics = evaluate_domains(model, selection_gates, args.device)
                projection_ablation[mode][ablated_name] = {
                    "ratios": ratios(metrics, selection_baseline),
                    "incremental_ratios_vs_parent": ratios(
                        metrics, selection_source
                    ),
                }
                print(
                    f"{mode} {ablated_name} "
                    f"{projection_ablation[mode][ablated_name]['ratios']}",
                    flush=True,
                )

    candidates = {}
    candidate_masks = {}
    for fraction in fractions:
        count = (
            0
            if fraction == 0
            else min(eligible_groups, max(1, round(eligible_groups * fraction)))
        )
        masks = global_rescue_masks(benefit, eligible, count)
        install_rescue(masks)
        selection_metrics = evaluate_domains(model, selection_gates, args.device)
        selection_ratios = ratios(selection_metrics, selection_baseline)
        selection_incremental_ratios = ratios(selection_metrics, selection_source)
        actual_fraction = count / eligible_groups
        all_target_q4_fraction = count / total_groups
        target_average_bpw = (
            q2_g128_physical_bpw()
            + all_target_q4_fraction
            * (q4_g128_physical_bpw() - q2_g128_physical_bpw())
        )
        key = f"q4_{actual_fraction:.8f}"
        candidates[key] = {
            "requested_fraction": fraction,
            "rescued_groups": count,
            "eligible_groups": eligible_groups,
            "eligible_q4_fraction": actual_fraction,
            "all_target_q4_fraction": all_target_q4_fraction,
            "target_physical_bpw": target_average_bpw,
            "selection_metrics": selection_metrics,
            "selection_ratios": selection_ratios,
            "selection_incremental_ratios": selection_incremental_ratios,
            "selection_worst_ratio": max(selection_ratios.values()),
            "selection_passed": all(
                value <= args.gate_ratio for value in selection_ratios.values()
            )
            and all(
                value <= args.incremental_gate_ratio
                for value in selection_incremental_ratios.values()
            ),
        }
        candidate_masks[key] = {name: mask.cpu() for name, mask in masks.items()}
        print(key, selection_ratios, flush=True)

    full_evaluations = {}
    first_full_pass = None
    for key in sorted(candidates, key=lambda item: candidates[item]["rescued_groups"]):
        if not candidates[key]["selection_passed"]:
            continue
        masks = {
            name: value.to(args.device) for name, value in candidate_masks[key].items()
        }
        install_rescue(masks)
        metrics = evaluate_domains(model, gates, args.device)
        metric_ratios = ratios(metrics, baseline)
        incremental_ratios = ratios(metrics, source)
        passed = all(value <= args.gate_ratio for value in metric_ratios.values()) and all(
            value <= args.incremental_gate_ratio
            for value in incremental_ratios.values()
        )
        full_evaluations[key] = {
            "metrics": metrics,
            "ratios": metric_ratios,
            "incremental_ratios_vs_parent": incremental_ratios,
            "passed": passed,
        }
        print(f"full {key} passed={passed} ratios={metric_ratios}", flush=True)
        if passed:
            first_full_pass = key
            break

    artifact_path = None
    artifact_sha256 = None
    if args.write_artifact and first_full_pass is not None:
        parent_has_q8 = (
            parent_artifact is not None
            and parent_artifact.get("format") == "wal-tat-mixed-q2-q4-q8-v1"
        )
        artifact_path = (
            Path(source_path).parents[1]
            / "artifacts"
            / (
                f"wal-tat-{args.tag}-mixed-q2-q4-q8.pt"
                if parent_has_q8
                else f"wal-tat-{args.tag}-mixed-q2-q4.pt"
            )
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_matrices = (
            copy.deepcopy(parent_artifact["matrices"])
            if parent_artifact is not None
            else {}
        )
        for name in names:
            target_entry = {
                "shape": tuple(matrices[name].master_weight.shape),
                "source_committed_mask": source_committed_masks[name],
                "q4_mask": candidate_masks[first_full_pass][name],
                "q2_codes_int8": q2_codes[name],
                "q2_scales_fp16": q2_scales[name],
                "q4_codes_int8": q4_codes_by_name[name],
                "q4_scales_fp16": q4_scales_by_name[name],
            }
            if parent_has_q8:
                target_entry.update(
                    {
                        "q8_mask": torch.zeros_like(
                            target_entry["q4_mask"], dtype=torch.bool
                        ),
                        "q8_codes_int8": torch.zeros_like(
                            target_entry["q4_codes_int8"], dtype=torch.int8
                        ),
                        "q8_scales_fp16": torch.ones_like(
                            target_entry["q4_scales_fp16"], dtype=torch.float16
                        ),
                    }
                )
            artifact_matrices[name] = target_entry
        artifact_payload = {
            "format": (
                "wal-tat-mixed-q2-q4-q8-v1"
                if parent_has_q8
                else "wal-tat-mixed-q2-q4-v1"
            ),
            "source_checkpoint_sha256": sha256_file(source_path),
            "suite_sha256": sha256_file(suite_path),
            "target_layer": args.target_layer,
            "target_projections": requested,
            "parent_artifact_sha256": (
                sha256_file(parent_path) if parent_path is not None else None
            ),
            "candidate": first_full_pass,
            "group_size": 128,
            "q2_physical_bpw": q2_g128_physical_bpw(),
            "q4_physical_bpw": q4_g128_physical_bpw(),
            "matrices": artifact_matrices,
            "development_ratios": full_evaluations[first_full_pass]["ratios"],
            "development_incremental_ratios_vs_parent": full_evaluations[
                first_full_pass
            ]["incremental_ratios_vs_parent"],
        }
        if parent_has_q8:
            artifact_payload["q8_physical_bpw"] = q8_g128_physical_bpw()
        torch.save(
            artifact_payload,
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-macro-mlp-q4-rescue-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "parent_artifact": str(parent_path) if parent_path is not None else None,
        "parent_artifact_sha256": (
            sha256_file(parent_path) if parent_path is not None else None
        ),
        "parent_new_ternary_weights": (
            parent_install.new_q2_weights if parent_install is not None else 0
        ),
        "parent_q4_weights": (
            parent_install.q4_weights if parent_install is not None else 0
        ),
        "parent_q8_weights": (
            parent_install.q8_weights if parent_install is not None else 0
        ),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": "development rate-distortion diagnostic; sealed audit not opened",
        "target_layer": args.target_layer,
        "target_projections": requested,
        "moment_sequences": args.moment_sequences,
        "selection_gate_sequences": args.selection_gate_sequences,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "source_metrics": source,
        "selection_source_metrics": selection_source,
        "source_ratios": source_ratios,
        "full_ternary_selection_metrics": full_ternary_selection,
        "full_ternary_selection_ratios": full_ternary_selection_ratios,
        "full_ternary_metrics": full_ternary,
        "full_ternary_ratios": full_ternary_ratios,
        "projection_ablation": projection_ablation,
        "total_target_groups": total_groups,
        "eligible_groups": eligible_groups,
        "candidate_results": candidates,
        "full_evaluations": full_evaluations,
        "first_full_pass": first_full_pass,
        "development_passed": first_full_pass is not None,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
        "checkpoint_written": False,
        "sealed_audit_opened": False,
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
        f"wrote {output}; first_full_pass={first_full_pass} "
        f"checkpoint_written=False",
        flush=True,
    )
    del model, matrices, moments, ternary_grouped, q4_grouped
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
