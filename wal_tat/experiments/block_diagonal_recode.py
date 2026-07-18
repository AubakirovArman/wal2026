"""Reinitialize one completed ternary block with activation-weighted WLS codes."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from wal_tat import diagonal_ternary_search
from wal_tat.quantization import padded_grouped

from compensation_window import (
    PROJECT,
    WORKSPACE,
    checkpoint_payload,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    log,
    ratios,
    seed_everything,
    sha256_file,
)
from decoder_sensitivity_scan import balanced_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=WORKSPACE
        / "wal2/checkpoints/wal-tat-block27_code_recovery_stage3_c4x4-all-codes.pt",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-diverse-recovery-v1-c4x4-l256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="block27_diagonal_recode_v1")
    parser.add_argument("--samples-per-domain", type=int, default=16)
    parser.add_argument("--domain-offset", type=int, default=32)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class InputMomentCollector:
    def __init__(self, module):
        self.total = None
        self.count = 0
        self.handle = module.register_forward_pre_hook(self.capture)

    def capture(self, _module, inputs):
        value = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        current = value.square().sum(0)
        self.total = current if self.total is None else self.total + current
        self.count += value.shape[0]

    def moment(self):
        if self.total is None:
            raise RuntimeError("no inputs were collected")
        return self.total / max(self.count, 1)

    def close(self):
        self.handle.remove()


def weighted_error(weight, codes, scales, input_moment, group_size):
    grouped, padding, size = padded_grouped(weight, group_size)
    moment = input_moment.detach().float()
    if padding:
        moment = torch.nn.functional.pad(moment, (0, padding))
    moment = moment.view(1, -1, size)
    grouped_codes = codes
    if grouped_codes.shape == weight.shape:
        if padding:
            grouped_codes = torch.nn.functional.pad(grouped_codes, (0, padding))
        grouped_codes = grouped_codes.view_as(grouped)
    error = (
        moment
        * (grouped - grouped_codes.float() * scales.float().unsqueeze(-1)).square()
    ).sum()
    normalizer = (moment * grouped.square()).sum().clamp_min(1e-12)
    return float((error / normalizer).item())


def main() -> None:
    args = parse_args()
    started = time.time()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration = balanced_calibration(
        suite, args.samples_per_domain, args.domain_offset
    )
    gates = suite["gates"]

    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    original_weights = {
        name: model.get_submodule(name).weight.detach().cpu().clone()
        for name in source_payload["matrices"]
    }
    matrices = install_checkpoint(model, source_payload, args.device)
    starting = evaluate_domains(model, gates, args.device)
    starting_ratios = ratios(starting, baseline)

    collectors = {
        name: InputMomentCollector(model.get_submodule(name)) for name in matrices
    }
    try:
        with torch.no_grad():
            for index, chunk in enumerate(calibration):
                batch = chunk.unsqueeze(0).to(args.device)
                model(input_ids=batch[:, :-1], use_cache=False)
                if (index + 1) % 12 == 0 or index + 1 == len(calibration):
                    log(f"input moments {index + 1}/{len(calibration)}")
    finally:
        for collector in collectors.values():
            collector.close()

    thresholds = tuple(index / 8 for index in range(0, 25))
    search_stats = {}
    with torch.no_grad():
        for name, matrix in matrices.items():
            original = original_weights[name].to(args.device)
            moment = collectors[name].moment()
            before_codes = matrix.committed_codes.detach().clone()
            before_scale = matrix.group_scale.detach().clone()
            before_error = weighted_error(
                original,
                before_codes,
                before_scale,
                moment,
                matrix.group_size,
            )
            codes, scales, _ = diagonal_ternary_search(
                original,
                moment,
                group_size=matrix.group_size,
                threshold_multipliers=thresholds,
            )
            after_error = weighted_error(
                original, codes, scales, moment, matrix.group_size
            )
            churn = float((codes != before_codes).float().mean().item())
            matrix.committed_codes.copy_(codes)
            matrix.group_scale.copy_(scales)
            matrix.master_weight.copy_(original.float())
            search_stats[name] = {
                "relative_weighted_error_before": before_error,
                "relative_weighted_error_after": after_error,
                "error_ratio": after_error / max(before_error, 1e-12),
                "code_churn": churn,
                "zero_fraction_before": float((before_codes == 0).float().mean()),
                "zero_fraction_after": float((codes == 0).float().mean()),
            }
            log(
                f"{name}: error {before_error:.6f}->{after_error:.6f} "
                f"churn={churn:.3%} zeros={search_stats[name]['zero_fraction_after']:.3%}"
            )

    attempted = evaluate_domains(model, gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    checkpoint_path = WORKSPACE / f"wal2/checkpoints/wal-tat-{args.tag}.pt"
    payload = checkpoint_payload(
        model,
        matrices,
        source_hash,
        attempted,
        attempted_ratios,
        {
            "experiment": args.tag,
            "mode": "activation-weighted-diagonal-ternary-recode-candidate",
            "accepted": False,
            "threshold_multipliers": thresholds,
        },
    )
    torch.save(payload, checkpoint_path)
    result = {
        "schema": "wal-tat-block-diagonal-recode-v1",
        "status": "candidate_not_accepted_until_independent_audit",
        "model": str(model_path),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_sha256": source_hash,
        "suite": str(args.suite.resolve()),
        "suite_sha256": suite_hash,
        "samples_per_domain": args.samples_per_domain,
        "domain_offset": args.domain_offset,
        "threshold_multipliers": thresholds,
        "baseline": baseline,
        "starting": starting,
        "starting_ratios": starting_ratios,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "search": search_stats,
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
    }
    result_path = PROJECT / f"results/{args.tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {result_path}; ratios={attempted_ratios}")
    del model, matrices, collectors, original_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
