"""Checkpoint-neutral BF16 restore ablation for a partial ternary frontier."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    seed_everything,
    sha256_file,
)


MATRIX_ARMS = {
    "layer24_down": ("model.layers.24.mlp.down_proj",),
    "layer24_gate": ("model.layers.24.mlp.gate_proj",),
    "layer24_up": ("model.layers.24.mlp.up_proj",),
    "layer24_mlp": (
        "model.layers.24.mlp.down_proj",
        "model.layers.24.mlp.gate_proj",
        "model.layers.24.mlp.up_proj",
    ),
    "layer24_attention": (
        "model.layers.24.self_attn.q_proj",
        "model.layers.24.self_attn.k_proj",
        "model.layers.24.self_attn.v_proj",
        "model.layers.24.self_attn.o_proj",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--arm-set", choices=("committed", "fallback"), default="committed"
    )
    parser.add_argument("--gate-sequences", type=int, default=256)
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def layer_matrices(payload: dict, layer: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer}."
    return tuple(sorted(name for name in payload["matrices"] if name.startswith(prefix)))


@torch.no_grad()
def restore_source_state(model, matrices, payload: dict, device: str) -> None:
    for name, entry in payload["matrices"].items():
        matrix = matrices[name]
        matrix.master_weight.copy_(entry["fp_master_bf16"].to(device).float())
        matrix.group_scale.copy_(entry["scales_fp16"].to(device).float())
        matrix.committed_mask.copy_(entry["committed_mask"].to(device))
        codes = entry["ternary_codes_int8"].to(device, dtype=torch.int8)
        if matrix.padding:
            codes = F.pad(codes, (0, matrix.padding))
        matrix.committed_codes.copy_(codes.view_as(matrix.committed_codes))
    parameters = dict(model.named_parameters())
    for name, value in payload.get("norm_extras", {}).items():
        parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


@torch.no_grad()
def restore_original_committed_groups(matrix, original_weight: torch.Tensor) -> dict:
    mask = matrix.committed_mask.clone()
    original = F.pad(
        original_weight.to(matrix.master_weight.device).float(), (0, matrix.padding)
    ).view_as(matrix.committed_codes)
    current = F.pad(
        matrix.master_weight.detach(), (0, matrix.padding)
    ).view_as(matrix.committed_codes).clone()
    current[mask] = original[mask]
    matrix.master_weight.copy_(
        current.reshape(matrix.out_features, -1)[:, : matrix.in_features]
    )
    matrix.committed_mask[mask] = False
    groups = int(mask.sum().item())
    return {"groups_restored": groups, "weights_restored": groups * matrix.group_size}


@torch.no_grad()
def restore_original_uncommitted_groups(matrix, original_weight: torch.Tensor) -> dict:
    mask = ~matrix.committed_mask
    original = F.pad(
        original_weight.to(matrix.master_weight.device).float(), (0, matrix.padding)
    ).view_as(matrix.committed_codes)
    current = F.pad(
        matrix.master_weight.detach(), (0, matrix.padding)
    ).view_as(matrix.committed_codes).clone()
    delta = current[mask] - original[mask]
    denominator = original[mask].float().norm().clamp_min(1e-12)
    current[mask] = original[mask]
    matrix.master_weight.copy_(
        current.reshape(matrix.out_features, -1)[:, : matrix.in_features]
    )
    groups = int(mask.sum().item())
    return {
        "groups_restored": groups,
        "weights_restored": groups * matrix.group_size,
        "relative_l2_drift": float(delta.float().norm().div(denominator).item()),
        "mean_abs_drift": float(delta.float().abs().mean().item()) if groups else 0.0,
        "max_abs_drift": float(delta.float().abs().max().item()) if groups else 0.0,
    }


@torch.no_grad()
def restore_full_original_matrix(matrix, original_weight: torch.Tensor) -> dict:
    committed_groups = int(matrix.committed_mask.sum().item())
    matrix.master_weight.copy_(
        original_weight.to(matrix.master_weight.device, dtype=matrix.master_weight.dtype)
    )
    matrix.committed_mask.zero_()
    total_groups = matrix.committed_mask.numel()
    return {
        "groups_restored": total_groups,
        "weights_restored": total_groups * matrix.group_size,
        "committed_groups_removed": committed_groups,
    }


@torch.no_grad()
def restore_original_norms(model, names: tuple[str, ...], originals: dict) -> int:
    parameters = dict(model.named_parameters())
    count = 0
    for name in names:
        value = originals[name]
        parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
        count += value.numel()
    return count


def compare(metrics: dict, baseline: dict, source: dict) -> dict:
    result = {}
    for domain, value in metrics.items():
        delta_bf16 = value["nll"] - baseline[domain]["nll"]
        source_delta = source[domain]["nll"] - baseline[domain]["nll"]
        result[domain] = {
            "nll": value["nll"],
            "ppl": value["ppl"],
            "nll_ratio_to_bf16": value["nll"] / baseline[domain]["nll"],
            "delta_nll_to_bf16": delta_bf16,
            "ppl_ratio_to_bf16": math.exp(delta_bf16),
            "nll_ratio_to_source": value["nll"] / source[domain]["nll"],
            "delta_nll_to_source": value["nll"] - source[domain]["nll"],
            "ppl_ratio_to_source": math.exp(value["nll"] - source[domain]["nll"]),
            "source_gap_recovered_fraction": (
                (source_delta - delta_bf16) / source_delta
                if abs(source_delta) > 1e-12
                else None
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    gates = {
        domain: chunks[: args.gate_sequences]
        for domain, chunks in suite["gates"].items()
    }
    if any(len(chunks) != args.gate_sequences for chunks in gates.values()):
        raise ValueError("suite does not contain the requested gate sequence count")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    original_weights = {
        name: model.get_submodule(name).weight.detach().cpu().clone()
        for name in payload["matrices"]
    }
    model_parameters = dict(model.named_parameters())
    original_parameters = {
        name: model_parameters[name].detach().cpu().clone()
        for name in payload.get("norm_extras", {})
    }
    baseline = evaluate_domains(model, gates, args.device)
    matrices = install_checkpoint(model, payload, args.device)
    source = evaluate_domains(model, gates, args.device)

    layer24 = layer_matrices(payload, 24)
    layer27 = layer_matrices(payload, 27)
    all_matrices = tuple(sorted(payload["matrices"]))
    layer24_norms = tuple(
        sorted(
            name
            for name in payload.get("norm_extras", {})
            if name.startswith("model.layers.24.")
        )
    )
    all_norms = tuple(sorted(payload.get("norm_extras", {})))
    if args.arm_set == "committed":
        arms = {
            **{
                arm: {"committed": names}
                for arm, names in MATRIX_ARMS.items()
            },
            "layer24_all": {"committed": layer24},
            "layer27_all": {"committed": layer27},
            "all_checkpoint_matrices": {"committed": all_matrices},
            "norm_extras_only": {"norms": all_norms},
            "layer24_all_plus_layer24_norms": {
                "committed": layer24,
                "norms": layer24_norms,
            },
            "all_matrices_plus_all_norms": {
                "committed": all_matrices,
                "norms": all_norms,
            },
        }
    else:
        mlp = MATRIX_ARMS["layer24_mlp"]
        up = MATRIX_ARMS["layer24_up"]
        arms = {
            "layer24_down_fallback": {"fallback": MATRIX_ARMS["layer24_down"]},
            "layer24_gate_fallback": {"fallback": MATRIX_ARMS["layer24_gate"]},
            "layer24_up_fallback": {"fallback": up},
            "layer24_mlp_fallback": {"fallback": mlp},
            "layer24_up_full_original": {"full": up},
            "layer24_mlp_full_original": {"full": mlp},
            "layer24_all_full_original": {"full": layer24},
            "all_full_original_matrices_plus_all_norms": {
                "full": all_matrices,
                "norms": all_norms,
            },
        }
    arm_results = {}
    for arm, spec in arms.items():
        restore_source_state(model, matrices, payload, args.device)
        restored = {}
        for name in spec.get("committed", ()):
            restored[name] = {
                "mode": "committed",
                **restore_original_committed_groups(
                    matrices[name], original_weights[name]
                ),
            }
        for name in spec.get("fallback", ()):
            restored[name] = {
                "mode": "fallback",
                **restore_original_uncommitted_groups(
                    matrices[name], original_weights[name]
                ),
            }
        for name in spec.get("full", ()):
            restored[name] = {
                "mode": "full",
                **restore_full_original_matrix(matrices[name], original_weights[name]),
            }
        norm_names = spec.get("norms", ())
        norm_parameters_restored = restore_original_norms(
            model, norm_names, original_parameters
        )
        metrics = evaluate_domains(model, gates, args.device)
        arm_results[arm] = {
            "matrix_spec": {key: list(value) for key, value in spec.items() if key != "norms"},
            "matrix_restore": restored,
            "norms": list(norm_names),
            "norm_parameters_restored": norm_parameters_restored,
            "metrics": metrics,
            "comparison": compare(metrics, baseline, source),
        }
        print(
            arm,
            {
                domain: round(value["nll_ratio_to_bf16"], 9)
                for domain, value in arm_results[arm]["comparison"].items()
            },
            flush=True,
        )

    restore_source_state(model, matrices, payload, args.device)
    source_recheck = evaluate_domains(model, gates, args.device)
    restoration_exact = all(
        source_recheck[domain]["nll"] == source[domain]["nll"] for domain in gates
    )
    result = {
        "schema": "wal-tat-counterfactual-bf16-restore-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": suite.get("policy"),
        "gate_sequences": args.gate_sequences,
        "arm_set": args.arm_set,
        "seed": args.seed,
        "baseline": baseline,
        "source": source,
        "source_comparison": compare(source, baseline, source),
        "arms": arm_results,
        "source_recheck": source_recheck,
        "source_restoration_exact": restoration_exact,
        "checkpoint_written": False,
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}; source_restoration_exact={restoration_exact}")
    if not restoration_exact:
        raise SystemExit(2)
    del model, matrices, original_weights, original_parameters
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
