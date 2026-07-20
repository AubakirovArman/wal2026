"""Reinitialize one already committed region from the original BF16 weights.

This development-only experiment compares three strict g128 ternary PTQ
initializers on the exact same committed mask.  Selection and confirmation are
disjoint.  A passing run emits a frozen recode artifact and never mutates or
publishes the source checkpoint.
"""
from __future__ import annotations

import argparse
import gc
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
from counterfactual_scale_recovery import slice_domain_gates
from wal_tat import diagonal_ternary_search, hard_codes_scales
from wal_tat.quantization import padded_grouped


PROJECTION_MAP = {
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument(
        "--target-projection", choices=tuple(PROJECTION_MAP), default="up_proj"
    )
    parser.add_argument("--moment-samples-per-domain", type=int, default=8)
    parser.add_argument("--moment-domain-offset", type=int, default=0)
    parser.add_argument("--selection-gate-start", type=int, default=0)
    parser.add_argument("--selection-gate-sequences", type=int, default=128)
    parser.add_argument("--confirmation-gate-start", type=int, default=128)
    parser.add_argument("--confirmation-gate-sequences", type=int, default=128)
    parser.add_argument("--threshold-step", type=float, default=0.125)
    parser.add_argument("--threshold-max", type=float, default=3.0)
    parser.add_argument("--min-selection-improvement", type=float, default=1e-3)
    parser.add_argument("--min-confirmation-improvement", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def calibration_domain_labels(suite: dict) -> list[str]:
    domains = list(suite["source_splits"])
    repeats = {
        domains[0]: int(suite.get("c4_calibration_repeat", 1)),
        domains[1]: int(suite.get("squad_calibration_repeat", 1)),
        domains[2]: int(suite.get("code_calibration_repeat", 1)),
    }
    base_count = int(suite["calibration_per_domain"])
    labels: list[str] = []
    if suite.get("interleave_calibration", False):
        for _index in range(base_count):
            for domain in domains:
                labels.extend([domain] * repeats[domain])
    else:
        for domain in domains:
            labels.extend([domain] * (base_count * repeats[domain]))
    if len(labels) != len(suite["calibration"]):
        raise ValueError("calibration metadata does not match stored sequences")
    return labels


def balanced_calibration(
    suite: dict, *, count: int, offset: int
) -> list[torch.Tensor]:
    if count <= 0 or offset < 0:
        raise ValueError("calibration count must be positive and offset non-negative")
    labels = calibration_domain_labels(suite)
    domains = list(suite["source_splits"])
    pools = {domain: [] for domain in domains}
    for label, sequence in zip(labels, suite["calibration"]):
        pools[label].append(sequence)
    result = []
    for domain in domains:
        selected = pools[domain][offset : offset + count]
        if len(selected) != count:
            raise ValueError(f"insufficient calibration samples for {domain}")
        result.extend(selected)
    return result


@torch.no_grad()
def collect_input_second_moment(model, module, calibration, device: str) -> torch.Tensor:
    total = None
    count = 0

    def capture(_module, args) -> None:
        nonlocal total, count
        value = args[0].detach().float().reshape(-1, args[0].shape[-1])
        current = value.square().sum(0)
        total = current if total is None else total + current
        count += value.shape[0]

    handle = module.register_forward_pre_hook(capture)
    try:
        model.eval()
        for sequence in calibration:
            batch = sequence.unsqueeze(0).to(device)
            model(input_ids=batch[:, :-1], use_cache=False)
    finally:
        handle.remove()
    if total is None or count == 0:
        raise RuntimeError("no target inputs were collected")
    return total / count


def grouped_codes(codes: torch.Tensor, matrix) -> torch.Tensor:
    if codes.shape == matrix.master_weight.shape:
        if matrix.padding:
            codes = F.pad(codes, (0, matrix.padding))
        codes = codes.view_as(matrix.committed_codes)
    if codes.shape != matrix.committed_codes.shape:
        raise ValueError("initializer code shape mismatch")
    return codes.to(torch.int8)


@torch.no_grad()
def representation_statistics(
    original_weight: torch.Tensor,
    mask: torch.Tensor,
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
    candidate_codes: torch.Tensor,
    candidate_scales: torch.Tensor,
    input_moment: torch.Tensor,
    group_size: int,
) -> dict:
    grouped, padding, size = padded_grouped(original_weight.float(), group_size)
    moment = input_moment.float()
    if padding:
        moment = F.pad(moment, (0, padding))
    moment = moment.view(1, -1, size)
    source = source_codes.float() * source_scales.unsqueeze(-1)
    candidate = candidate_codes.float() * candidate_scales.unsqueeze(-1)
    energy = grouped[mask].square().sum().clamp_min(1e-12)
    weighted_energy = (grouped.square() * moment)[mask].sum().clamp_min(1e-12)

    def errors(reconstructed: torch.Tensor) -> tuple[float, float]:
        delta2 = (grouped - reconstructed).square()
        return (
            float((delta2[mask].sum() / energy).item()),
            float(((delta2 * moment)[mask].sum() / weighted_energy).item()),
        )

    source_mse, source_weighted = errors(source)
    candidate_mse, candidate_weighted = errors(candidate)
    active_codes = mask.unsqueeze(-1).expand_as(source_codes)
    changed_codes = (source_codes != candidate_codes) & active_codes
    changed_scales = (source_scales != candidate_scales) & mask
    return {
        "changed_code_values": int(changed_codes.sum().item()),
        "changed_scale_groups": int(changed_scales.sum().item()),
        "code_churn": float(changed_codes.sum().div(active_codes.sum()).item()),
        "source_relative_weight_mse": source_mse,
        "candidate_relative_weight_mse": candidate_mse,
        "weight_mse_reduction_fraction": 1 - candidate_mse / max(source_mse, 1e-12),
        "source_relative_activation_weighted_mse": source_weighted,
        "candidate_relative_activation_weighted_mse": candidate_weighted,
        "activation_weighted_mse_reduction_fraction": (
            1 - candidate_weighted / max(source_weighted, 1e-12)
        ),
        "zero_fraction": float((candidate_codes[mask] == 0).float().mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.threshold_step <= 0 or args.threshold_max < 0:
        raise ValueError("threshold range is invalid")
    selection_range = set(
        range(
            args.selection_gate_start,
            args.selection_gate_start + args.selection_gate_sequences,
        )
    )
    confirmation_range = set(
        range(
            args.confirmation_gate_start,
            args.confirmation_gate_start + args.confirmation_gate_sequences,
        )
    )
    if selection_range & confirmation_range:
        raise ValueError("selection and confirmation gate slices overlap")
    thresholds = tuple(
        index * args.threshold_step
        for index in range(round(args.threshold_max / args.threshold_step) + 1)
    )
    source_path = args.source_checkpoint.resolve()
    suite_path = args.suite.resolve()
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
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    target_name = (
        f"model.layers.{args.target_layer}."
        f"{PROJECTION_MAP[args.target_projection]}"
    )
    if target_name not in payload["matrices"]:
        raise ValueError(f"target matrix is absent from checkpoint: {target_name}")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    original_weight = model.get_submodule(target_name).weight.detach().float().clone()
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = evaluate_domains(model, confirmation_gates, args.device)
    matrices = install_checkpoint(model, payload, args.device)
    matrix = matrices[target_name]
    selection_source = evaluate_domains(model, selection_gates, args.device)
    confirmation_source = evaluate_domains(model, confirmation_gates, args.device)
    source_selection_ratios = ratios(selection_source, selection_baseline)
    source_confirmation_ratios = ratios(confirmation_source, confirmation_baseline)
    input_moment = collect_input_second_moment(
        model, model.get_submodule(target_name), calibration, args.device
    )

    source_codes = matrix.committed_codes.detach().clone()
    source_scales = matrix.group_scale.detach().clone()
    abs_codes_flat, abs_scales = hard_codes_scales(
        original_weight, matrix.group_size
    )
    abs_codes = grouped_codes(abs_codes_flat, matrix)
    ls_codes, ls_scales, _ = diagonal_ternary_search(
        original_weight,
        torch.ones_like(input_moment),
        group_size=matrix.group_size,
        threshold_multipliers=thresholds,
    )
    wls_codes, wls_scales, _ = diagonal_ternary_search(
        original_weight,
        input_moment,
        group_size=matrix.group_size,
        threshold_multipliers=thresholds,
    )
    initializers = {
        "absmean_round": (grouped_codes(abs_codes, matrix), abs_scales),
        "threshold_ls": (grouped_codes(ls_codes, matrix), ls_scales),
        "activation_wls": (grouped_codes(wls_codes, matrix), wls_scales),
    }
    candidate_states = {}
    candidate_results = {}
    for name, (codes, scales) in initializers.items():
        candidate_codes = source_codes.clone()
        candidate_scales = source_scales.clone()
        candidate_codes[matrix.committed_mask] = codes[matrix.committed_mask]
        candidate_scales[matrix.committed_mask] = scales.half().float()[
            matrix.committed_mask
        ]
        with torch.no_grad():
            matrix.committed_codes.copy_(candidate_codes)
            matrix.group_scale.copy_(candidate_scales)
        metrics = evaluate_domains(model, selection_gates, args.device)
        domain_ratios = ratios(metrics, selection_baseline)
        statistics = representation_statistics(
            original_weight,
            matrix.committed_mask,
            source_codes,
            source_scales,
            candidate_codes,
            candidate_scales,
            input_moment,
            matrix.group_size,
        )
        candidate_results[name] = {
            "selection_metrics": metrics,
            "selection_ratios": domain_ratios,
            "selection_worst_ratio": max(domain_ratios.values()),
            "representation": statistics,
        }
        candidate_states[name] = (candidate_codes.cpu(), candidate_scales.cpu())
        print(name, domain_ratios, statistics, flush=True)
        with torch.no_grad():
            matrix.committed_codes.copy_(source_codes)
            matrix.group_scale.copy_(source_scales)

    chosen = min(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["selection_worst_ratio"],
            sum(candidate_results[name]["selection_ratios"].values()),
        ),
    )
    candidate_codes, candidate_scales = candidate_states[chosen]
    with torch.no_grad():
        matrix.committed_codes.copy_(candidate_codes.to(args.device))
        matrix.group_scale.copy_(candidate_scales.to(args.device))
    confirmation_metrics = evaluate_domains(model, confirmation_gates, args.device)
    confirmation_ratios = ratios(confirmation_metrics, confirmation_baseline)
    selection_improvement = max(source_selection_ratios.values()) - max(
        candidate_results[chosen]["selection_ratios"].values()
    )
    confirmation_improvement = max(source_confirmation_ratios.values()) - max(
        confirmation_ratios.values()
    )
    passed = (
        selection_improvement >= args.min_selection_improvement
        and confirmation_improvement >= args.min_confirmation_improvement
    )
    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wal-tat-ternary-recode-v1",
                "source_checkpoint_sha256": sha256_file(source_path),
                "suite_sha256": sha256_file(suite_path),
                "target_name": target_name,
                "candidate": chosen,
                "recipe": "committed_original_bf16_initializer",
                "residual_gain": 1.0,
                "committed_mask": matrix.committed_mask.cpu(),
                "ternary_codes_int8": candidate_codes.to(torch.int8),
                "scales_fp16": candidate_scales.half(),
                "selection_ratios": candidate_results[chosen]["selection_ratios"],
                "confirmation_ratios": confirmation_ratios,
                "representation": candidate_results[chosen]["representation"],
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    with torch.no_grad():
        matrix.committed_codes.copy_(source_codes)
        matrix.group_scale.copy_(source_scales)
    final_source = evaluate_domains(model, confirmation_gates, args.device)
    final_restoration_exact = all(
        final_source[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    result = {
        "schema": "wal-tat-committed-initializer-ablation-v1",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": "development selection and disjoint confirmation; not final audit",
        "target_name": target_name,
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
        "candidate_results": candidate_results,
        "chosen_candidate": chosen,
        "confirmation_metrics": confirmation_metrics,
        "confirmation_ratios": confirmation_ratios,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_selection_improvement": args.min_selection_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "passed": passed,
        "artifact": str(artifact_path) if passed else None,
        "artifact_sha256": artifact_sha256,
        "codes_strict_ternary": bool(
            torch.all((candidate_codes >= -1) & (candidate_codes <= 1)).item()
        ),
        "committed_mask_unchanged": True,
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
        f"wrote {output}; chosen={chosen} passed={passed} "
        f"selection_improvement={selection_improvement} "
        f"confirmation_improvement={confirmation_improvement}",
        flush=True,
    )
    if not final_restoration_exact:
        raise SystemExit(2)
    del model, matrices, candidate_states, original_weight, input_moment
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
