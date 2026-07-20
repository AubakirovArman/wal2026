"""Joint strict-ternary reinitialization of committed MLP regions.

The experiment starts from an immutable checkpoint plus an optional frozen
single-matrix recode artifact.  It exhaustively evaluates a small Cartesian
grid of PTQ initializers for already committed regions of sibling MLP
projections.  Selection and confirmation are disjoint development slices.
Only a passing run emits a self-contained multi-matrix overlay; it never
publishes or mutates a checkpoint.
"""
from __future__ import annotations

import argparse
import gc
from itertools import product
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    WORKSPACE,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    sha256_file,
)
from committed_initializer_ablation import (
    balanced_calibration,
    grouped_codes,
    representation_statistics,
)
from counterfactual_proxy_recovery import load_starting_artifact
from counterfactual_scale_recovery import slice_domain_gates
from wal_tat import diagonal_ternary_search, hard_codes_scales


MLP_PROJECTIONS = {
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
}
INITIALIZERS = ("source", "absmean_round", "threshold_ls", "activation_wls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--starting-artifact", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument(
        "--target-projections",
        default="gate_proj,down_proj",
        help="comma-separated sibling MLP projections to reinitialize jointly",
    )
    parser.add_argument("--moment-samples-per-domain", type=int, default=8)
    parser.add_argument("--moment-domain-offset", type=int, default=8)
    parser.add_argument("--selection-gate-start", type=int, default=0)
    parser.add_argument("--selection-gate-sequences", type=int, default=64)
    parser.add_argument("--confirmation-gate-start", type=int, default=128)
    parser.add_argument("--confirmation-gate-sequences", type=int, default=128)
    parser.add_argument("--threshold-step", type=float, default=0.125)
    parser.add_argument("--threshold-max", type=float, default=3.0)
    parser.add_argument("--min-selection-improvement", type=float, default=1e-3)
    parser.add_argument("--min-confirmation-improvement", type=float, default=0.0)
    parser.add_argument("--max-total-code-churn", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parse_projection_list(value: str) -> tuple[str, ...]:
    projections = tuple(item.strip() for item in value.split(",") if item.strip())
    if not projections:
        raise ValueError("at least one target projection is required")
    if len(set(projections)) != len(projections):
        raise ValueError("target projections must be unique")
    unknown = set(projections) - set(MLP_PROJECTIONS)
    if unknown:
        raise ValueError(f"unknown MLP projections: {sorted(unknown)}")
    return projections


def initializer_grid(
    target_names: tuple[str, ...],
) -> list[dict[str, str]]:
    return [
        dict(zip(target_names, choices))
        for choices in product(INITIALIZERS, repeat=len(target_names))
    ]


def aggregate_change_statistics(entries: dict[str, dict]) -> dict:
    active = sum(int(entry["active_code_values"]) for entry in entries.values())
    changed = sum(int(entry["changed_code_values"]) for entry in entries.values())
    groups = sum(int(entry["active_scale_groups"]) for entry in entries.values())
    changed_groups = sum(
        int(entry["changed_scale_groups"]) for entry in entries.values()
    )
    return {
        "active_code_values": active,
        "changed_code_values": changed,
        "code_churn": changed / max(active, 1),
        "active_scale_groups": groups,
        "changed_scale_groups": changed_groups,
    }


@torch.no_grad()
def collect_input_second_moments(
    model, modules: dict[str, torch.nn.Module], calibration, device: str
) -> dict[str, torch.Tensor]:
    totals: dict[str, torch.Tensor | None] = {name: None for name in modules}
    counts = {name: 0 for name in modules}
    handles = []

    def make_hook(name: str):
        def capture(_module, args) -> None:
            value = args[0].detach().float().reshape(-1, args[0].shape[-1])
            current = value.square().sum(0)
            totals[name] = current if totals[name] is None else totals[name] + current
            counts[name] += value.shape[0]

        return capture

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        for sequence in calibration:
            batch = sequence.unsqueeze(0).to(device)
            model(input_ids=batch[:, :-1], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(totals[name] is None or counts[name] == 0 for name in modules):
        raise RuntimeError("failed to collect one or more target input moments")
    return {
        name: totals[name] / counts[name]  # type: ignore[operator]
        for name in modules
    }


@torch.no_grad()
def apply_state(matrix, state: tuple[torch.Tensor, torch.Tensor]) -> None:
    codes, scales = state
    matrix.committed_codes.copy_(codes.to(matrix.committed_codes.device))
    matrix.group_scale.copy_(scales.to(matrix.group_scale.device).float())


def matrix_change_statistics(
    reference: tuple[torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
) -> dict:
    reference_codes, reference_scales = reference
    candidate_codes, candidate_scales = candidate
    active = mask.unsqueeze(-1).expand_as(reference_codes)
    return {
        "active_code_values": int(active.sum().item()),
        "changed_code_values": int(
            ((reference_codes != candidate_codes) & active).sum().item()
        ),
        "active_scale_groups": int(mask.sum().item()),
        "changed_scale_groups": int(
            ((reference_scales.float() != candidate_scales.float()) & mask).sum().item()
        ),
    }


def main() -> None:
    args = parse_args()
    projections = parse_projection_list(args.target_projections)
    if args.threshold_step <= 0 or args.threshold_max < 0:
        raise ValueError("threshold range is invalid")
    if not 0 <= args.max_total_code_churn <= 1:
        raise ValueError("max-total-code-churn must be in [0, 1]")
    selection_indices = set(
        range(
            args.selection_gate_start,
            args.selection_gate_start + args.selection_gate_sequences,
        )
    )
    confirmation_indices = set(
        range(
            args.confirmation_gate_start,
            args.confirmation_gate_start + args.confirmation_gate_sequences,
        )
    )
    if selection_indices & confirmation_indices:
        raise ValueError("selection and confirmation gate slices overlap")

    source_path = args.source_checkpoint.resolve()
    source_hash = sha256_file(source_path)
    suite_path = args.suite.resolve()
    suite_hash = sha256_file(suite_path)
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = balanced_calibration(
        suite,
        count=args.moment_samples_per_domain,
        offset=args.moment_domain_offset,
    )
    selection_gates = slice_domain_gates(
        suite["gates"],
        start=args.selection_gate_start,
        count=args.selection_gate_sequences,
        role="selection",
    )
    confirmation_gates = slice_domain_gates(
        suite["gates"],
        start=args.confirmation_gate_start,
        count=args.confirmation_gate_sequences,
        role="confirmation",
    )
    target_names = tuple(
        f"model.layers.{args.target_layer}.{MLP_PROJECTIONS[projection]}"
        for projection in projections
    )
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    missing = set(target_names) - set(payload["matrices"])
    if missing:
        raise ValueError(f"checkpoint is missing target matrices: {sorted(missing)}")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    original_weights = {
        name: model.get_submodule(name).weight.detach().float().clone()
        for name in target_names
    }
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = evaluate_domains(model, confirmation_gates, args.device)
    matrices = install_checkpoint(model, payload, args.device)
    checkpoint_states = {
        name: (
            matrix.committed_codes.detach().clone(),
            matrix.group_scale.detach().clone(),
        )
        for name, matrix in matrices.items()
    }

    starting_artifact_path = (
        args.starting_artifact.resolve() if args.starting_artifact else None
    )
    starting_artifact_hash = None
    starting_target = None
    if starting_artifact_path is not None:
        artifact_header = torch.load(
            starting_artifact_path, map_location="cpu", weights_only=False
        )
        starting_target = artifact_header.get("target_name")
        if starting_target in target_names:
            raise ValueError("starting artifact target must not be reinitialized again")
        if starting_target not in matrices:
            raise ValueError("starting artifact target is absent from checkpoint")
        _artifact, codes, scales = load_starting_artifact(
            starting_artifact_path,
            checkpoint_hash=source_hash,
            target_name=starting_target,
            matrix=matrices[starting_target],
            device=args.device,
        )
        apply_state(matrices[starting_target], (codes, scales))
        starting_artifact_hash = sha256_file(starting_artifact_path)

    source_states = {
        name: (
            matrices[name].committed_codes.detach().clone(),
            matrices[name].group_scale.detach().clone(),
        )
        for name in matrices
    }
    selection_source = evaluate_domains(model, selection_gates, args.device)
    confirmation_source = evaluate_domains(model, confirmation_gates, args.device)
    source_selection_ratios = ratios(selection_source, selection_baseline)
    source_confirmation_ratios = ratios(confirmation_source, confirmation_baseline)
    moments = collect_input_second_moments(
        model,
        {name: model.get_submodule(name) for name in target_names},
        calibration,
        args.device,
    )

    thresholds = tuple(
        index * args.threshold_step
        for index in range(round(args.threshold_max / args.threshold_step) + 1)
    )
    states: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    initializer_statistics: dict[str, dict[str, dict]] = {}
    for name in target_names:
        matrix = matrices[name]
        source_codes, source_scales = source_states[name]
        original = original_weights[name]
        abs_codes_flat, abs_scales = hard_codes_scales(original, matrix.group_size)
        abs_codes = grouped_codes(abs_codes_flat, matrix)
        ls_codes, ls_scales, _ = diagonal_ternary_search(
            original,
            torch.ones_like(moments[name]),
            group_size=matrix.group_size,
            threshold_multipliers=thresholds,
        )
        wls_codes, wls_scales, _ = diagonal_ternary_search(
            original,
            moments[name],
            group_size=matrix.group_size,
            threshold_multipliers=thresholds,
        )
        raw = {
            "source": (source_codes, source_scales),
            "absmean_round": (grouped_codes(abs_codes, matrix), abs_scales),
            "threshold_ls": (grouped_codes(ls_codes, matrix), ls_scales),
            "activation_wls": (grouped_codes(wls_codes, matrix), wls_scales),
        }
        states[name] = {}
        initializer_statistics[name] = {}
        for initializer, (codes, scales) in raw.items():
            candidate_codes = source_codes.clone()
            candidate_scales = source_scales.clone()
            if initializer != "source":
                candidate_codes[matrix.committed_mask] = codes.to(args.device)[
                    matrix.committed_mask
                ]
                candidate_scales[matrix.committed_mask] = scales.to(args.device).half().float()[
                    matrix.committed_mask
                ]
            states[name][initializer] = (candidate_codes, candidate_scales)
            initializer_statistics[name][initializer] = representation_statistics(
                original,
                matrix.committed_mask,
                source_codes,
                source_scales,
                candidate_codes,
                candidate_scales,
                moments[name],
                matrix.group_size,
            )

    grid_results = []
    for choices in initializer_grid(target_names):
        for name in target_names:
            apply_state(matrices[name], states[name][choices[name]])
        metrics = evaluate_domains(model, selection_gates, args.device)
        domain_ratios = ratios(metrics, selection_baseline)
        change_entries = {
            name: matrix_change_statistics(
                source_states[name], states[name][choices[name]], matrices[name].committed_mask
            )
            for name in target_names
        }
        aggregate = aggregate_change_statistics(change_entries)
        entry = {
            "initializers": choices,
            "selection_ratios": domain_ratios,
            "selection_worst_ratio": max(domain_ratios.values()),
            "selection_ratio_sum": sum(domain_ratios.values()),
            "incremental_change": aggregate,
        }
        grid_results.append(entry)
        print(choices, domain_ratios, aggregate["code_churn"], flush=True)
        for name in target_names:
            apply_state(matrices[name], source_states[name])

    chosen = min(
        grid_results,
        key=lambda entry: (
            entry["selection_worst_ratio"],
            entry["selection_ratio_sum"],
            entry["incremental_change"]["code_churn"],
        ),
    )
    chosen_states = {
        name: states[name][chosen["initializers"][name]] for name in target_names
    }
    for name in target_names:
        apply_state(matrices[name], chosen_states[name])
    confirmation_metrics = evaluate_domains(model, confirmation_gates, args.device)
    confirmation_ratios = ratios(confirmation_metrics, confirmation_baseline)
    selection_improvement = max(source_selection_ratios.values()) - float(
        chosen["selection_worst_ratio"]
    )
    confirmation_improvement = max(source_confirmation_ratios.values()) - max(
        confirmation_ratios.values()
    )

    overlay_names = tuple(
        dict.fromkeys((starting_target,) + target_names)
        if starting_target is not None
        else target_names
    )
    total_change_entries = {
        name: matrix_change_statistics(
            checkpoint_states[name],
            (
                matrices[name].committed_codes.detach(),
                matrices[name].group_scale.detach(),
            ),
            matrices[name].committed_mask,
        )
        for name in overlay_names
    }
    total_change = aggregate_change_statistics(total_change_entries)
    passed = (
        selection_improvement >= args.min_selection_improvement
        and confirmation_improvement >= args.min_confirmation_improvement
        and total_change["code_churn"] <= args.max_total_code_churn
    )

    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_matrices = {
            name: {
                "committed_mask": matrices[name].committed_mask.detach().cpu(),
                "ternary_codes_int8": matrices[name].committed_codes.detach()
                .cpu()
                .to(torch.int8),
                "scales_fp16": matrices[name].group_scale.detach().cpu().half(),
            }
            for name in overlay_names
        }
        torch.save(
            {
                "format": "wal-tat-ternary-recode-bundle-v1",
                "source_checkpoint_sha256": source_hash,
                "starting_artifact_sha256": starting_artifact_hash,
                "suite_sha256": suite_hash,
                "target_layer": args.target_layer,
                "target_names": list(overlay_names),
                "recipe": "joint_committed_mlp_initializer_grid",
                "chosen_initializers": chosen["initializers"],
                "matrices": artifact_matrices,
                "selection_ratios": chosen["selection_ratios"],
                "confirmation_ratios": confirmation_ratios,
                "incremental_change": chosen["incremental_change"],
                "total_change_from_checkpoint": total_change,
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    for name in overlay_names:
        apply_state(matrices[name], source_states[name])
    final_source = evaluate_domains(model, confirmation_gates, args.device)
    final_restoration_exact = all(
        final_source[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    result = {
        "schema": "wal-tat-committed-mlp-initializer-ablation-v1",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "starting_artifact": (
            str(starting_artifact_path) if starting_artifact_path else None
        ),
        "starting_artifact_sha256": starting_artifact_hash,
        "suite": str(suite_path),
        "suite_sha256": suite_hash,
        "suite_role": "development selection and disjoint confirmation; not final audit",
        "target_layer": args.target_layer,
        "target_names": list(target_names),
        "moment_samples_per_domain": args.moment_samples_per_domain,
        "moment_domain_offset": args.moment_domain_offset,
        "selection_slice": [
            args.selection_gate_start,
            args.selection_gate_sequences,
        ],
        "confirmation_slice": [
            args.confirmation_gate_start,
            args.confirmation_gate_sequences,
        ],
        "threshold_multipliers": list(thresholds),
        "source_selection_ratios": source_selection_ratios,
        "source_confirmation_ratios": source_confirmation_ratios,
        "initializer_statistics": initializer_statistics,
        "grid_results": grid_results,
        "chosen_initializers": chosen["initializers"],
        "chosen_selection_ratios": chosen["selection_ratios"],
        "confirmation_metrics": confirmation_metrics,
        "confirmation_ratios": confirmation_ratios,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_selection_improvement": args.min_selection_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "max_total_code_churn": args.max_total_code_churn,
        "incremental_change": chosen["incremental_change"],
        "total_change_from_checkpoint": total_change,
        "passed": passed,
        "artifact": str(artifact_path) if passed else None,
        "artifact_sha256": artifact_sha256,
        "codes_strict_ternary": all(
            bool(
                torch.all(
                    (matrices[name].committed_codes >= -1)
                    & (matrices[name].committed_codes <= 1)
                ).item()
            )
            for name in overlay_names
        ),
        "committed_masks_unchanged": all(
            torch.equal(
                matrices[name].committed_mask,
                payload["matrices"][name]["committed_mask"].to(args.device),
            )
            for name in overlay_names
        ),
        "persistent_bf16_residual": False,
        "coverage_changed": False,
        "checkpoint_written": False,
        "final_restoration_exact": final_restoration_exact,
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
        f"wrote {output}; passed={passed} chosen={chosen['initializers']} "
        f"selection_improvement={selection_improvement} "
        f"confirmation_improvement={confirmation_improvement} "
        f"total_churn={total_change['code_churn']}",
        flush=True,
    )
    if not final_restoration_exact:
        raise SystemExit(2)
    del model, matrices, states, original_weights, moments
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
