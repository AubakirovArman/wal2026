"""Atomically publish an audited strict-ternary plus fallback-QAT overlay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from commit_ternary_recode_artifact import (
    atomic_torch_save,
    grouped,
    load_json,
    require,
)
from compensation_window import PROJECT, sha256_file
from wal_tat.quantization import padded_grouped


def matrix_change_statistics(
    *,
    shape: tuple[int, int],
    group_size: int,
    source_mask: torch.Tensor,
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
    source_master: torch.Tensor,
    artifact_codes: torch.Tensor,
    artifact_scales: torch.Tensor,
    artifact_master: torch.Tensor,
) -> dict:
    active_codes = source_mask.unsqueeze(-1).expand_as(source_codes)
    changed_codes = artifact_codes != source_codes
    changed_scales = artifact_scales != source_scales
    if changed_codes[~active_codes].any():
        raise ValueError("artifact changes code outside committed groups")
    if changed_scales[~source_mask].any():
        raise ValueError("artifact changes scale outside committed groups")
    source_grouped, _, _ = padded_grouped(source_master.float(), group_size)
    artifact_grouped, _, _ = padded_grouped(artifact_master.float(), group_size)
    if tuple(source_master.shape) != shape or tuple(artifact_master.shape) != shape:
        raise ValueError("master shape mismatch")
    if not torch.equal(
        artifact_grouped[source_mask], source_grouped[source_mask]
    ):
        raise ValueError("artifact changes committed master values")
    fallback = ~source_mask
    delta = artifact_grouped[fallback] - source_grouped[fallback]
    source_values = source_grouped[fallback]
    denominator = source_values.abs().mean().clamp_min(1e-12)
    return {
        "committed_code_values": int(active_codes.sum().item()),
        "changed_code_values": int(changed_codes[active_codes].sum().item()),
        "changed_scale_groups": int(changed_scales[source_mask].sum().item()),
        "code_churn": float(changed_codes[active_codes].float().mean().item()),
        "fallback_groups": int(fallback.sum().item()),
        "fallback_weights": int(delta.numel()),
        "changed_fallback_weights": int((delta != 0).sum().item()),
        "fallback_absolute_delta_mean": float(delta.abs().mean().item()),
        "fallback_relative_delta": float(
            delta.abs().mean().div(denominator).item()
        ),
        "committed_master_values_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("development_result", type=Path)
    parser.add_argument("audit_result", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    development_path = args.development_result.resolve()
    audit_path = args.audit_result.resolve()
    policy_path = args.policy.resolve()
    output = args.output.resolve()
    source_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    development_hash = sha256_file(development_path)
    audit_hash = sha256_file(audit_path)
    policy_hash = sha256_file(policy_path)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    development = load_json(development_path)
    audit = load_json(audit_path)
    policy = load_json(policy_path)
    require(payload.get("format") == "wal-tat-multi-g128-v2", "bad checkpoint")
    require(
        artifact.get("format") == "wal-tat-ternary-fallback-compensation-v1",
        "bad artifact",
    )
    require(
        artifact.get("source_checkpoint_sha256") == source_hash,
        "artifact source mismatch",
    )
    require(
        development.get("schema")
        == "wal-tat-mlp-fallback-compensation-recovery-v1",
        "bad development schema",
    )
    require(development.get("passed") is True, "development did not pass")
    require(
        development.get("artifact_sha256") == artifact_hash,
        "development artifact mismatch",
    )
    require(
        audit.get("schema") == "wal-tat-ternary-recode-artifact-audit-v1",
        "bad audit schema",
    )
    require(audit.get("accepted") is True, "frozen audit did not pass")
    require(audit.get("checkpoint_sha256") == source_hash, "audit source mismatch")
    require(audit.get("artifact_sha256") == artifact_hash, "audit artifact mismatch")
    require(
        audit.get("artifact_format")
        == "wal-tat-ternary-fallback-compensation-v1",
        "audit artifact format mismatch",
    )
    require(audit.get("checkpoint_mutated") is False, "audit mutated checkpoint")
    require(audit.get("source_restoration_exact") is True, "audit restoration failed")
    require(
        audit.get("accepted_coverage_changed") is False,
        "audit changed accepted coverage",
    )
    require(
        policy.get("schema") == "wal-tat-fallback-compensation-commit-policy-v1",
        "bad commit policy schema",
    )
    require(policy.get("declared_before_execution") is True, "policy not declared")
    require(policy.get("source_checkpoint_sha256") == source_hash, "policy source mismatch")
    require(policy.get("artifact_sha256") == artifact_hash, "policy artifact mismatch")
    require(
        policy.get("development_result_sha256") == development_hash,
        "policy development mismatch",
    )
    require(policy.get("audit_result_sha256") == audit_hash, "policy audit mismatch")

    target_names = tuple(artifact["target_names"])
    require(
        set(target_names) == set(artifact["matrices"]),
        "artifact target list mismatch",
    )
    require(
        set(target_names) == set(policy["target_names"]),
        "policy target list mismatch",
    )
    coverage_before = sum(
        int(matrix["committed_mask"].sum().item()) * int(matrix["group_size"])
        for matrix in payload["matrices"].values()
    )
    require(
        coverage_before == int(policy["accepted_weights"]),
        "source accepted-weight count differs from policy",
    )
    statistics = {}
    for name in target_names:
        require(name in payload["matrices"], "artifact target absent from checkpoint")
        source_entry = payload["matrices"][name]
        artifact_entry = artifact["matrices"][name]
        shape = tuple(source_entry["shape"])
        group_size = int(source_entry["group_size"])
        source_mask = source_entry["committed_mask"].bool()
        artifact_mask = artifact_entry["committed_mask"].bool()
        source_codes = grouped(
            source_entry["ternary_codes_int8"].to(torch.int8), shape, group_size
        )
        artifact_codes = artifact_entry["ternary_codes_int8"].to(torch.int8)
        source_scales = source_entry["scales_fp16"].half()
        artifact_scales = artifact_entry["scales_fp16"].half()
        source_master = source_entry["fp_master_bf16"].bfloat16()
        artifact_master = artifact_entry["fp_master_bf16"].bfloat16()
        require(torch.equal(source_mask, artifact_mask), "committed mask mismatch")
        require(artifact_codes.shape == source_codes.shape, "code shape mismatch")
        require(artifact_scales.shape == source_scales.shape, "scale shape mismatch")
        require(
            set(artifact_codes.unique().tolist()) <= {-1, 0, 1},
            "artifact contains non-ternary codes",
        )
        require(torch.isfinite(artifact_scales).all(), "artifact scales not finite")
        require((artifact_scales > 0).all(), "artifact scales not positive")
        require(torch.isfinite(artifact_master).all(), "artifact master not finite")
        item = matrix_change_statistics(
            shape=shape,
            group_size=group_size,
            source_mask=source_mask,
            source_codes=source_codes,
            source_scales=source_scales,
            source_master=source_master,
            artifact_codes=artifact_codes,
            artifact_scales=artifact_scales,
            artifact_master=artifact_master,
        )
        expected = policy["expected_changes"][name]
        for field in (
            "changed_code_values",
            "changed_scale_groups",
            "changed_fallback_weights",
        ):
            require(
                int(item[field]) == int(expected[field]),
                f"{name} {field} differs from policy",
            )
        require(
            item["code_churn"] <= float(expected["max_code_churn"]),
            f"{name} code churn exceeds policy",
        )
        require(
            item["fallback_relative_delta"]
            <= float(expected["max_fallback_relative_delta"]),
            f"{name} fallback delta exceeds policy",
        )
        statistics[name] = item
        source_entry["ternary_codes_int8"] = artifact_codes.reshape(shape[0], -1)[
            :, : shape[1]
        ].contiguous()
        source_entry["scales_fp16"] = artifact_scales.contiguous()
        source_entry["fp_master_bf16"] = artifact_master.contiguous()

    coverage_after = sum(
        int(matrix["committed_mask"].sum().item()) * int(matrix["group_size"])
        for matrix in payload["matrices"].values()
    )
    require(coverage_after == coverage_before, "accepted coverage changed")
    require(sha256_file(checkpoint) == source_hash, "source changed before publish")
    previous_metadata = payload.get("metadata", {})
    payload["parent_sha256"] = source_hash
    # Preserve the source checkpoint's native metric payload; the audited
    # ratios are recorded separately and are not interchangeable with NLL/PPL
    # metric dictionaries.
    payload["gate_ratios"] = audit["candidate_ratios_to_bf16"]
    payload["metadata"] = {
        "experiment": args.tag,
        "mode": "coverage-neutral strict ternary MLP recode plus local fallback QAT",
        "source_checkpoint_sha256": source_hash,
        "artifact_sha256": artifact_hash,
        "development_result_sha256": development_hash,
        "frozen_audit_result_sha256": audit_hash,
        "commit_policy_sha256": policy_hash,
        "target_names": list(target_names),
        "recipe": artifact["recipe"],
        "matrix_changes": statistics,
        "aggregate_relative_fallback_delta": artifact["aggregate_relative_delta"],
        "accepted_weights_before": coverage_before,
        "accepted_weights_after": coverage_after,
        "persistent_bf16_residual_on_committed_groups": False,
        "coverage_changed": False,
        "previous_metadata": previous_metadata,
    }
    atomic_torch_save(payload, output)
    require(sha256_file(checkpoint) == source_hash, "source changed during publish")
    result = {
        "schema": "wal-tat-fallback-compensation-commit-v1",
        "status": "published_pending_fresh_verification",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_hash,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "development_result": str(development_path),
        "development_result_sha256": development_hash,
        "audit_result": str(audit_path),
        "audit_result_sha256": audit_hash,
        "policy": str(policy_path),
        "policy_sha256": policy_hash,
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": sha256_file(output),
        "output_checkpoint_bytes": output.stat().st_size,
        "target_names": list(target_names),
        "matrix_changes": statistics,
        "accepted_weights_before": coverage_before,
        "accepted_weights_after": coverage_after,
        "coverage_changed": False,
        "source_checkpoint_mutated": False,
        "old_checkpoint_deleted": False,
        "fresh_verification_required": True,
    }
    result_path = PROJECT / f"results/{args.tag}_commit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
