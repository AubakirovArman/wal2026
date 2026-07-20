"""Project a useful continuous residual back into strict ternary codes/scales."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

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
from counterfactual_bf16_restore_audit import restore_source_state


RECIPES = (
    "fixed_code_ls",
    "fixed_scale_recode",
    "absmean_recode",
    "lloyd_recode",
    "threshold_search",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--selection-start", type=int, default=0)
    parser.add_argument("--selection-count", type=int, default=128)
    parser.add_argument("--confirmation-start", type=int, default=128)
    parser.add_argument("--confirmation-count", type=int, default=128)
    parser.add_argument("--min-selection-improvement", type=float, default=5e-5)
    parser.add_argument("--min-confirmation-improvement", type=float, default=0.0)
    parser.add_argument("--max-code-churn", type=float, default=0.005)
    parser.add_argument(
        "--recipes",
        default=",".join(RECIPES),
        help="Comma-separated projection recipes to evaluate.",
    )
    parser.add_argument(
        "--residual-gains",
        default="1",
        help="Comma-separated multipliers for the learned residual direction.",
    )
    parser.add_argument("--seed", type=int, default=167)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def least_squares_scale(target: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    numerator = (target * codes.float()).sum(dim=-1)
    denominator = codes.float().square().sum(dim=-1)
    fallback = target.abs().mean(dim=-1).clamp_min(1e-5)
    return torch.where(
        denominator > 0,
        numerator.div(denominator.clamp_min(1)).abs().clamp_min(1e-5),
        fallback,
    )


def recode(target: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return target.div(scales.unsqueeze(-1)).round().clamp(-1, 1).to(torch.int8)


@torch.no_grad()
def make_candidate(
    recipe: str,
    target: torch.Tensor,
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if recipe == "fixed_code_ls":
        codes = source_codes.clone()
        scales = least_squares_scale(target, codes)
    elif recipe == "fixed_scale_recode":
        scales = source_scales.clone()
        codes = recode(target, scales)
    elif recipe == "absmean_recode":
        scales = target.abs().mean(dim=-1).clamp_min(1e-5)
        codes = recode(target, scales)
        scales = least_squares_scale(target, codes)
    elif recipe == "lloyd_recode":
        codes = source_codes.clone()
        scales = source_scales.clone()
        for _ in range(4):
            scales = least_squares_scale(target, codes)
            codes = recode(target, scales)
    elif recipe == "threshold_search":
        mean_abs = target.abs().mean(dim=-1).clamp_min(1e-5)
        best_error = torch.full_like(mean_abs, float("inf"))
        best_codes = source_codes.clone()
        best_scales = source_scales.clone()
        for multiplier in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
            threshold = mean_abs.unsqueeze(-1) * multiplier
            codes = torch.where(
                target.abs() >= threshold,
                target.sign(),
                torch.zeros_like(target),
            ).to(torch.int8)
            scales = least_squares_scale(target, codes)
            error = (
                target - codes.float() * scales.unsqueeze(-1)
            ).square().mean(dim=-1)
            choose = error < best_error
            best_error = torch.where(choose, error, best_error)
            best_scales = torch.where(choose, scales, best_scales)
            best_codes = torch.where(choose.unsqueeze(-1), codes, best_codes)
        codes, scales = best_codes, best_scales
    else:
        raise ValueError(f"unsupported recipe: {recipe}")
    return codes, scales


@torch.no_grad()
def representation_statistics(
    target: torch.Tensor,
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
    candidate_codes: torch.Tensor,
    candidate_scales: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    source_hard = source_codes.float() * source_scales.unsqueeze(-1)
    candidate_hard = candidate_codes.float() * candidate_scales.unsqueeze(-1)
    source_error = (target - source_hard)[mask].float().square().sum()
    candidate_error = (target - candidate_hard)[mask].float().square().sum()
    churn = (candidate_codes != source_codes)[mask].float().mean()
    relative_scale = (
        candidate_scales[mask] - source_scales[mask]
    ).abs().div(source_scales[mask].abs().clamp_min(1e-5))
    return {
        "target_error_source": float(source_error.item()),
        "target_error_candidate": float(candidate_error.item()),
        "residual_energy_captured_fraction": float(
            (1 - candidate_error / source_error.clamp_min(1e-12)).item()
        ),
        "code_churn": float(churn.item()),
        "scale_relative_change_mean": float(relative_scale.mean().item()),
        "scale_relative_change_max": float(relative_scale.max().item()),
    }


def sliced_gates(gates: dict, start: int, count: int) -> dict:
    result = {domain: chunks[start : start + count] for domain, chunks in gates.items()}
    if any(len(chunks) != count for chunks in result.values()):
        raise ValueError("suite lacks a requested selection/confirmation slice")
    return result


def main() -> None:
    args = parse_args()
    recipes = tuple(item.strip() for item in args.recipes.split(",") if item.strip())
    unsupported = sorted(set(recipes) - set(RECIPES))
    if not recipes or unsupported:
        raise ValueError(f"invalid recipes: {unsupported or recipes}")
    residual_gains = tuple(
        float(item.strip())
        for item in args.residual_gains.split(",")
        if item.strip()
    )
    if not residual_gains or any(gain <= 0 for gain in residual_gains):
        raise ValueError("residual gains must be positive")
    source_path = args.source_checkpoint.resolve()
    residual_path = args.residual_artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    residual_payload = torch.load(
        residual_path, map_location="cpu", weights_only=False
    )
    if residual_payload.get("format") != "wal-tat-matched-bf16-residual-v1":
        raise ValueError("unsupported residual artifact")
    if residual_payload["source_checkpoint_sha256"] != sha256_file(source_path):
        raise ValueError("residual artifact source hash mismatch")
    target_name = residual_payload["target_name"]
    if target_name not in source_payload["matrices"]:
        raise ValueError("residual target is absent from source checkpoint")
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    selection_gates = sliced_gates(
        suite["gates"], args.selection_start, args.selection_count
    )
    confirmation_gates = sliced_gates(
        suite["gates"], args.confirmation_start, args.confirmation_count
    )
    if set(
        range(args.selection_start, args.selection_start + args.selection_count)
    ) & set(
        range(
            args.confirmation_start,
            args.confirmation_start + args.confirmation_count,
        )
    ):
        raise ValueError("selection and confirmation slices overlap")

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    confirmation_baseline = evaluate_domains(model, confirmation_gates, args.device)
    matrices = install_checkpoint(model, source_payload, args.device)
    selection_source = evaluate_domains(model, selection_gates, args.device)
    confirmation_source = evaluate_domains(model, confirmation_gates, args.device)
    source_selection_ratios = ratios(selection_source, selection_baseline)
    source_confirmation_ratios = ratios(confirmation_source, confirmation_baseline)
    matrix = matrices[target_name]
    artifact_mask = residual_payload["committed_mask"].to(args.device)
    if not torch.equal(artifact_mask, matrix.committed_mask):
        raise ValueError("residual artifact committed mask mismatch")
    delta = residual_payload["delta_bf16"].to(args.device).float()
    grouped_delta = F.pad(delta, (0, matrix.padding)).view_as(matrix.committed_codes)
    source_codes = matrix.committed_codes.detach().clone()
    source_scales = matrix.group_scale.detach().clone()
    source_hard = source_codes.float() * source_scales.unsqueeze(-1)

    recipe_results = {}
    candidates = {}
    for gain in residual_gains:
        target = source_hard + grouped_delta * gain
        for recipe in recipes:
            candidate_key = f"{recipe}@{gain:g}x"
            candidate_codes, candidate_scales = make_candidate(
                recipe, target, source_codes, source_scales
            )
            candidate_codes = torch.where(
                matrix.committed_mask.unsqueeze(-1), candidate_codes, source_codes
            )
            candidate_scales = torch.where(
                matrix.committed_mask, candidate_scales, source_scales
            )
            restore_source_state(model, matrices, source_payload, args.device)
            with torch.no_grad():
                matrix.committed_codes.copy_(candidate_codes)
                matrix.group_scale.copy_(candidate_scales)
            selection_metrics = evaluate_domains(model, selection_gates, args.device)
            selection_ratios = ratios(selection_metrics, selection_baseline)
            statistics = representation_statistics(
                target,
                source_codes,
                source_scales,
                candidate_codes,
                candidate_scales,
                matrix.committed_mask,
            )
            recipe_results[candidate_key] = {
                "recipe": recipe,
                "residual_gain": gain,
                "selection_metrics": selection_metrics,
                "selection_ratios": selection_ratios,
                "selection_worst_ratio": max(selection_ratios.values()),
                "representation": statistics,
            }
            candidates[candidate_key] = (
                candidate_codes.cpu(),
                candidate_scales.cpu(),
            )
            print(candidate_key, selection_ratios, statistics, flush=True)

    chosen = min(
        candidates,
        key=lambda candidate: recipe_results[candidate]["selection_worst_ratio"],
    )
    restore_source_state(model, matrices, source_payload, args.device)
    chosen_codes, chosen_scales = candidates[chosen]
    with torch.no_grad():
        matrix.committed_codes.copy_(chosen_codes.to(args.device))
        matrix.group_scale.copy_(chosen_scales.to(args.device))
    confirmation_metrics = evaluate_domains(model, confirmation_gates, args.device)
    confirmation_ratios = ratios(confirmation_metrics, confirmation_baseline)
    selection_improvement = max(source_selection_ratios.values()) - max(
        recipe_results[chosen]["selection_ratios"].values()
    )
    confirmation_improvement = max(source_confirmation_ratios.values()) - max(
        confirmation_ratios.values()
    )
    passed = (
        selection_improvement >= args.min_selection_improvement
        and confirmation_improvement >= args.min_confirmation_improvement
        and recipe_results[chosen]["representation"]["code_churn"]
        <= args.max_code_churn
    )
    artifact_path = WORKSPACE / f"wal2/artifacts/wal-tat-{args.tag}.pt"
    artifact_sha256 = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wal-tat-ternary-recode-v1",
                "source_checkpoint_sha256": sha256_file(source_path),
                "residual_artifact_sha256": sha256_file(residual_path),
                "suite_sha256": sha256_file(suite_path),
                "target_name": target_name,
                "candidate": chosen,
                "recipe": recipe_results[chosen]["recipe"],
                "residual_gain": recipe_results[chosen]["residual_gain"],
                "committed_mask": matrix.committed_mask.cpu(),
                "ternary_codes_int8": chosen_codes.to(torch.int8),
                "scales_fp16": chosen_scales.half(),
                "selection_ratios": recipe_results[chosen]["selection_ratios"],
                "confirmation_ratios": confirmation_ratios,
                "representation": recipe_results[chosen]["representation"],
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)

    restore_source_state(model, matrices, source_payload, args.device)
    source_recheck = evaluate_domains(model, confirmation_gates, args.device)
    source_restoration_exact = all(
        source_recheck[domain]["nll"] == confirmation_source[domain]["nll"]
        for domain in confirmation_gates
    )
    result = {
        "schema": "wal-tat-distill-residual-to-ternary-v2",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "residual_artifact": str(residual_path),
        "residual_artifact_sha256": sha256_file(residual_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_name": target_name,
        "recipes": list(recipes),
        "residual_gains": list(residual_gains),
        "selection_slice": [args.selection_start, args.selection_count],
        "confirmation_slice": [args.confirmation_start, args.confirmation_count],
        "source_selection_ratios": source_selection_ratios,
        "source_confirmation_ratios": source_confirmation_ratios,
        "recipe_results": recipe_results,
        "chosen_candidate": chosen,
        "chosen_recipe": recipe_results[chosen]["recipe"],
        "chosen_residual_gain": recipe_results[chosen]["residual_gain"],
        "confirmation_metrics": confirmation_metrics,
        "confirmation_ratios": confirmation_ratios,
        "selection_worst_ratio_improvement": selection_improvement,
        "confirmation_worst_ratio_improvement": confirmation_improvement,
        "min_selection_improvement": args.min_selection_improvement,
        "min_confirmation_improvement": args.min_confirmation_improvement,
        "max_code_churn": args.max_code_churn,
        "passed": passed,
        "artifact": str(artifact_path) if passed else None,
        "artifact_sha256": artifact_sha256,
        "coverage_changed": False,
        "checkpoint_written": False,
        "source_restoration_exact": source_restoration_exact,
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}; chosen={chosen} passed={passed} "
        f"selection_improvement={selection_improvement} "
        f"confirmation_improvement={confirmation_improvement}",
        flush=True,
    )
    if not source_restoration_exact:
        raise SystemExit(2)
    del model, matrices, delta, grouped_delta, candidates
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
