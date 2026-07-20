"""Checkpoint-neutral ternary initializer stress test on sensitivity deciles.

The experiment evaluates identical uncommitted g128 atoms with strict hard
codes from three initializers: absmean rounding, unweighted threshold/LS, and
activation-weighted threshold/WLS.  It never commits a transaction or writes a
model checkpoint; only a small JSON evidence artifact is produced.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from wal_tat import (
    CausalMomentCollector,
    activation_fisher_group_damage,
    diagonal_ternary_search,
    hard_codes_scales,
    sensitivity_decile_mask,
)
from wal_tat.quantization import padded_grouped

from compensation_window import (
    PROJECT,
    WORKSPACE,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    log,
    ratios,
    seed_everything,
    sha256_file,
    tensor_digest,
)
from decoder_sensitivity_scan import balanced_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=WORKSPACE
        / "wal2/checkpoints/wal-tat-block24_down_d10_wls_replicate2_s0048.pt",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-diverse-recovery-v1-squadx2-l256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--candidate-name", default="model.layers.24.mlp.gate_proj")
    parser.add_argument("--tag", default="block24_gate_tail_initializers_v1")
    parser.add_argument("--deciles", default="8,9,10")
    parser.add_argument("--groups-per-decile", type=int, default=96)
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--domain-offset", type=int, default=0)
    parser.add_argument("--sequences-per-domain", type=int, default=64)
    parser.add_argument("--threshold-step", type=float, default=0.125)
    parser.add_argument("--threshold-max", type=float, default=3.0)
    parser.add_argument(
        "--exclude-artifact",
        type=Path,
        action="append",
        default=[],
        help="Exclude every group named by a prior partial initializer artifact.",
    )
    parser.add_argument(
        "--force-initializer",
        choices=("absmean_round", "threshold_ls", "activation_wls"),
        help="Freeze this initializer before validation instead of selecting on development.",
    )
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_deciles(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result) or any(not 1 <= item <= 10 for item in result):
        raise ValueError("deciles must be unique comma-separated integers in [1, 10]")
    return result


def grouped_codes(codes: torch.Tensor, matrix) -> torch.Tensor:
    value = codes
    if value.shape == matrix.master_weight.shape:
        if matrix.padding:
            value = F.pad(value, (0, matrix.padding))
        value = value.view_as(matrix.committed_codes)
    return value.to(torch.int8)


def excluded_groups(
    paths: list[Path],
    matrix,
    *,
    source_checkpoint_sha256: str,
    target_name: str,
) -> tuple[torch.Tensor, list[dict]]:
    """Load frozen artifact masks and return their union with exact provenance."""
    excluded = torch.zeros_like(matrix.committed_mask)
    flat_excluded = excluded.reshape(-1)
    records = []
    for raw_path in paths:
        path = raw_path.resolve()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("schema") != "wal-tat-partial-initializer-artifact-v1":
            raise ValueError(f"unsupported exclusion artifact: {path}")
        if artifact.get("target_name") != target_name:
            raise ValueError(f"exclusion artifact target mismatch: {path}")
        if list(artifact.get("target_shape", [])) != list(matrix.master_weight.shape):
            raise ValueError(f"exclusion artifact shape mismatch: {path}")
        if int(artifact.get("group_size", -1)) != matrix.group_size:
            raise ValueError(f"exclusion artifact group size mismatch: {path}")
        before = int(excluded.sum().item())
        for entry in artifact["entries"]:
            indices = entry["flat_group_indices"].to(
                flat_excluded.device, dtype=torch.long
            )
            if torch.any(indices < 0) or torch.any(indices >= flat_excluded.numel()):
                raise ValueError(f"exclusion artifact index is out of bounds: {path}")
            flat_excluded[indices] = True
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "source_checkpoint_sha256": artifact.get("source_checkpoint_sha256"),
                "source_checkpoint_matches_current": (
                    artifact.get("source_checkpoint_sha256")
                    == source_checkpoint_sha256
                ),
                "newly_excluded_groups": int(excluded.sum().item()) - before,
                "artifact_groups": sum(
                    int(entry["flat_group_indices"].numel())
                    for entry in artifact["entries"]
                ),
            }
        )
    return excluded.to(matrix.committed_mask.device), records


@torch.no_grad()
def initializer_statistics(matrix, codes, scales, mask, input_moment) -> dict:
    grouped, padding, size = padded_grouped(matrix.master_weight.detach(), matrix.group_size)
    moment = input_moment.detach().float()
    if padding:
        moment = F.pad(moment, (0, padding))
    moment = moment.view(1, -1, size)
    reconstructed = codes.float() * scales.float().unsqueeze(-1)
    delta2 = (grouped - reconstructed).square()
    plain_error = delta2[mask].sum()
    plain_energy = grouped[mask].square().sum().clamp_min(1e-12)
    weighted = (delta2 * moment)[mask].sum()
    weighted_energy = (grouped.square() * moment)[mask].sum().clamp_min(1e-12)
    selected_codes = codes[mask]
    values, counts = torch.unique(selected_codes.cpu(), return_counts=True)
    histogram = {-1: 0, 0: 0, 1: 0}
    histogram.update({int(value): int(count) for value, count in zip(values, counts)})
    return {
        "relative_weight_mse": float((plain_error / plain_energy).item()),
        "relative_activation_weighted_mse": float((weighted / weighted_energy).item()),
        "code_histogram": {str(key): value for key, value in histogram.items()},
        "zero_fraction": float((selected_codes == 0).float().mean().item()),
        "scale_mean": float(scales[mask].mean().item()),
        "scale_p95": float(torch.quantile(scales[mask].float(), 0.95).item()),
    }


def collect_moments(model, matrix, candidate_name, calibration, device):
    eligible = ~matrix.committed_mask
    matrix.begin(eligible, transaction_id="tail-initializer-moment-scan")
    matrix.set_candidate_state(0.0, 0.3508855606815209)
    matrix.master_weight.requires_grad_(True)
    target = model.get_submodule(candidate_name)
    try:
        with CausalMomentCollector(target) as collector:
            for index, chunk in enumerate(calibration, start=1):
                model.zero_grad(set_to_none=True)
                batch = chunk.unsqueeze(0).to(device)
                logits = model(input_ids=batch[:, :-1], use_cache=False).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    batch[:, 1:].reshape(-1),
                )
                loss.backward()
                log(f"tail moment {index}/{len(calibration)} loss={loss.item():.4f}")
            input_moment, output_moment = collector.moments()
    finally:
        model.zero_grad(set_to_none=True)
        matrix.rollback()
        matrix.master_weight.requires_grad_(False)
    scores = activation_fisher_group_damage(
        matrix.master_weight,
        input_moment,
        output_moment,
        group_size=matrix.group_size,
        scales=matrix.group_scale,
        relative=True,
    )
    scores[matrix.committed_mask] = float("inf")
    return input_moment, output_moment, scores


@torch.no_grad()
def evaluate_initializer(model, matrix, mask, codes, scales, gates, baseline, frontier, args):
    matrix.begin(mask, transaction_id="tail-initializer-evaluation")
    try:
        matrix.set_candidate_codes(codes, scales)
        matrix.set_candidate_state(1.0, 0.0)
        metrics = evaluate_domains(model, gates, args.device)
        deployed = grouped_codes(codes, matrix)
        stats = initializer_statistics(matrix, deployed, scales, mask, args.input_moment)
        return {
            "metrics": metrics,
            "ratios_to_raw_bf16": ratios(metrics, baseline),
            "ratios_to_frontier": ratios(metrics, frontier),
            **stats,
        }
    finally:
        matrix.rollback()


def main() -> None:
    args = parse_args()
    deciles = parse_deciles(args.deciles)
    if args.groups_per_decile < 1 or args.sequences_per_domain < 1:
        raise ValueError("group and sequence counts must be positive")
    if args.threshold_step <= 0 or args.threshold_max < 0:
        raise ValueError("threshold range is invalid")
    thresholds = tuple(
        index * args.threshold_step
        for index in range(round(args.threshold_max / args.threshold_step) + 1)
    )
    started = time.monotonic()
    model_path = (args.model_path or default_model_path()).resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    source_checkpoint_hash = sha256_file(source_checkpoint)
    suite_path = args.suite.resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = balanced_calibration(suite, args.samples_per_domain, args.domain_offset)
    gates = {
        name: chunks[: args.sequences_per_domain]
        for name, chunks in suite["gates"].items()
    }
    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    matrices = install_checkpoint(model, payload, args.device)
    if args.candidate_name not in matrices:
        raise ValueError("candidate-name must already exist in the partial frontier checkpoint")
    matrix = matrices[args.candidate_name]
    exclusion_mask, exclusion_records = excluded_groups(
        args.exclude_artifact,
        matrix,
        source_checkpoint_sha256=source_checkpoint_hash,
        target_name=args.candidate_name,
    )
    if torch.any(exclusion_mask & matrix.committed_mask):
        raise ValueError("exclusion artifacts overlap groups already committed in the frontier")
    frontier = evaluate_domains(model, gates, args.device)
    input_moment, _output_moment, scores = collect_moments(
        model, matrix, args.candidate_name, calibration, args.device
    )
    args.input_moment = input_moment

    abs_codes_flat, abs_scales = hard_codes_scales(matrix.master_weight, matrix.group_size)
    abs_codes = grouped_codes(abs_codes_flat, matrix)
    uniform_codes, uniform_scales, _ = diagonal_ternary_search(
        matrix.master_weight,
        torch.ones_like(input_moment),
        group_size=matrix.group_size,
        threshold_multipliers=thresholds,
    )
    wls_codes, wls_scales, _ = diagonal_ternary_search(
        matrix.master_weight,
        input_moment,
        group_size=matrix.group_size,
        threshold_multipliers=thresholds,
    )
    initializers = {
        "absmean_round": (abs_codes, abs_scales),
        "threshold_ls": (uniform_codes, uniform_scales),
        "activation_wls": (wls_codes, wls_scales),
    }

    experiments = {}
    artifact_entries = []
    eligible = ~matrix.committed_mask & ~exclusion_mask
    for decile in deciles:
        mask = sensitivity_decile_mask(
            scores,
            eligible,
            decile=decile,
            count=args.groups_per_decile,
        )
        key = f"D{decile}"
        experiments[key] = {
            "mask_sha256": tensor_digest(mask),
            "groups": int(mask.sum().item()),
            "weights": int(mask.sum().item() * matrix.group_size),
            "score_min": float(scores[mask].min().item()),
            "score_mean": float(scores[mask].mean().item()),
            "score_max": float(scores[mask].max().item()),
            "initializers": {},
        }
        for name, (codes, scales) in initializers.items():
            log(f"evaluating {key} initializer={name}")
            experiments[key]["initializers"][name] = evaluate_initializer(
                model,
                matrix,
                mask,
                codes,
                scales,
                gates,
                baseline,
                frontier,
                args,
            )

        chosen_name = args.force_initializer or min(
            experiments[key]["initializers"],
            key=lambda name: (
                max(
                    experiments[key]["initializers"][name][
                        "ratios_to_frontier"
                    ].values()
                ),
                sum(
                    experiments[key]["initializers"][name][
                        "ratios_to_frontier"
                    ].values()
                ),
            ),
        )
        experiments[key]["chosen_initializer"] = chosen_name
        chosen_codes, chosen_scales = initializers[chosen_name]
        artifact_entries.append(
            {
                "decile": decile,
                "initializer": chosen_name,
                "flat_group_indices": torch.where(mask.reshape(-1))[0].cpu(),
                "codes": chosen_codes[mask].cpu(),
                "scales": chosen_scales[mask].half().cpu(),
                "mask_sha256": experiments[key]["mask_sha256"],
                "development_worst_incremental_ratio": max(
                    experiments[key]["initializers"][chosen_name][
                        "ratios_to_frontier"
                    ].values()
                ),
            }
        )

    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "wal-tat-partial-initializer-artifact-v1",
        "development_only": True,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "development_suite": str(suite_path),
        "development_suite_sha256": sha256_file(suite_path),
        "target_name": args.candidate_name,
        "target_shape": list(matrix.master_weight.shape),
        "group_size": matrix.group_size,
        "entries": artifact_entries,
        "selection_rule": (
            f"predeclared fixed initializer: {args.force_initializer}"
            if args.force_initializer
            else "minimum worst incremental development NLL ratio per decile"
        ),
        "forced_initializer": args.force_initializer,
        "excluded_artifacts": exclusion_records,
        "excluded_groups": int(exclusion_mask.sum().item()),
        "checkpoint_mutated": False,
    }
    torch.save(artifact, artifact_path)

    result = {
        "schema": "wal-tat-tail-initializer-ablation-v1",
        "status": "checkpoint_neutral_development_ablation",
        "model": str(model_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": "development-only initializer and tail selection",
        "candidate_name": args.candidate_name,
        "group_size": matrix.group_size,
        "deciles": list(deciles),
        "groups_per_decile": args.groups_per_decile,
        "samples_per_domain": args.samples_per_domain,
        "domain_offset": args.domain_offset,
        "sequences_per_domain": args.sequences_per_domain,
        "threshold_multipliers": thresholds,
        "forced_initializer": args.force_initializer,
        "excluded_artifacts": exclusion_records,
        "excluded_groups": int(exclusion_mask.sum().item()),
        "baseline_raw_bf16": baseline,
        "frontier": frontier,
        "frontier_ratios_to_raw_bf16": ratios(frontier, baseline),
        "experiments": experiments,
        "frozen_artifact": str(artifact_path),
        "frozen_artifact_sha256": sha256_file(artifact_path),
        "checkpoint_mutated": False,
        "checkpoint_files_written": 0,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "artifact": str(artifact_path),
        "artifact_sha256": result["frozen_artifact_sha256"],
        "frontier_ratios": result["frontier_ratios_to_raw_bf16"],
        "worst_incremental": {
            decile: {
                name: max(values["ratios_to_frontier"].values())
                for name, values in data["initializers"].items()
            }
            for decile, data in experiments.items()
        },
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
    del model, matrices, scores, input_moment
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
