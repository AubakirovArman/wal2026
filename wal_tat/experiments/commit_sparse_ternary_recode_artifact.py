"""Atomically publish an audited sparse code-and-scale ternary recode."""
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


def sparse_change_statistics(
    source_codes: torch.Tensor,
    source_scales: torch.Tensor,
    artifact_codes: torch.Tensor,
    artifact_scales: torch.Tensor,
    committed_mask: torch.Tensor,
) -> dict:
    if artifact_codes.shape != source_codes.shape:
        raise ValueError("code shape mismatch")
    if artifact_scales.shape != source_scales.shape:
        raise ValueError("scale shape mismatch")
    if committed_mask.shape != source_scales.shape:
        raise ValueError("mask shape mismatch")
    active_values = committed_mask.unsqueeze(-1).expand_as(source_codes)
    changed_codes = artifact_codes != source_codes
    changed_scales = artifact_scales != source_scales
    if changed_codes[~active_values].any():
        raise ValueError("artifact changes code outside committed groups")
    if changed_scales[~committed_mask].any():
        raise ValueError("artifact changes scale outside committed groups")
    changed_code_values = int(changed_codes[active_values].sum().item())
    changed_scale_groups = int(changed_scales[committed_mask].sum().item())
    return {
        "changed_code_values": changed_code_values,
        "changed_scale_groups": changed_scale_groups,
        "code_churn": float(
            changed_codes[active_values].float().mean().item()
        ),
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
    require(artifact.get("format") == "wal-tat-ternary-recode-v1", "bad artifact")
    require(
        artifact.get("source_checkpoint_sha256") == source_hash,
        "artifact source mismatch",
    )
    require(
        development.get("schema") == "wal-tat-boundary-sparse-recode-v1",
        "bad development schema",
    )
    require(development.get("passed") is True, "development result did not pass")
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
    require(audit.get("checkpoint_mutated") is False, "audit mutated checkpoint")
    require(
        audit.get("accepted_coverage_changed") is False,
        "audit changed accepted coverage",
    )
    require(
        policy.get("schema") == "wal-tat-sparse-recode-commit-policy-v1",
        "bad commit policy schema",
    )
    require(policy.get("declared_before_execution") is True, "policy not declared")
    require(policy.get("source_checkpoint_sha256") == source_hash, "policy source mismatch")
    require(policy.get("artifact_sha256") == artifact_hash, "policy artifact mismatch")
    require(policy.get("development_result_sha256") == development_hash, "policy development mismatch")
    require(policy.get("audit_result_sha256") == audit_hash, "policy audit mismatch")

    target_name = artifact["target_name"]
    require(target_name in payload["matrices"], "target missing from checkpoint")
    entry = payload["matrices"][target_name]
    shape = tuple(entry["shape"])
    group_size = int(entry["group_size"])
    source_mask = entry["committed_mask"].bool()
    artifact_mask = artifact["committed_mask"].bool()
    source_codes = grouped(entry["ternary_codes_int8"].to(torch.int8), shape, group_size)
    source_scales = entry["scales_fp16"].half()
    artifact_codes = artifact["ternary_codes_int8"].to(torch.int8)
    artifact_scales = artifact["scales_fp16"].half()
    require(torch.equal(source_mask, artifact_mask), "committed mask mismatch")
    require(
        set(artifact_codes.unique().tolist()) <= {-1, 0, 1},
        "artifact contains non-ternary codes",
    )
    require(torch.isfinite(artifact_scales).all(), "artifact scales are not finite")
    require((artifact_scales > 0).all(), "artifact scales are not positive")
    statistics = sparse_change_statistics(
        source_codes,
        source_scales,
        artifact_codes,
        artifact_scales,
        source_mask,
    )
    expected = policy["expected_changes"]
    require(
        statistics["changed_code_values"] == int(expected["changed_code_values"]),
        "changed code count differs from policy",
    )
    require(
        statistics["changed_scale_groups"] == int(expected["changed_scale_groups"]),
        "changed scale count differs from policy",
    )
    require(
        statistics["code_churn"] <= float(expected["max_code_churn"]),
        "code churn exceeds policy",
    )
    require(statistics["changed_code_values"] > 0, "artifact changes no code values")

    coverage_before = sum(
        int(matrix["committed_mask"].sum().item()) * int(matrix["group_size"])
        for matrix in payload["matrices"].values()
    )
    entry["ternary_codes_int8"] = artifact_codes.reshape(shape[0], -1)[
        :, : shape[1]
    ].contiguous()
    entry["scales_fp16"] = artifact_scales.contiguous()
    coverage_after = sum(
        int(matrix["committed_mask"].sum().item()) * int(matrix["group_size"])
        for matrix in payload["matrices"].values()
    )
    require(coverage_after == coverage_before, "accepted coverage changed")
    previous_metadata = payload.get("metadata", {})
    payload["parent_sha256"] = source_hash
    payload["metadata"] = {
        "experiment": args.tag,
        "mode": "coverage-neutral behavior-gradient sparse adjacent ternary recode",
        "source_checkpoint_sha256": source_hash,
        "target_name": target_name,
        "ternary_recode_artifact_sha256": artifact_hash,
        "development_result_sha256": development_hash,
        "frozen_audit_result_sha256": audit_hash,
        "commit_policy_sha256": policy_hash,
        "recipe": artifact["recipe"],
        **statistics,
        "accepted_weights_before": coverage_before,
        "accepted_weights_after": coverage_after,
        "persistent_bf16_residual": False,
        "previous_metadata": previous_metadata,
    }
    atomic_torch_save(payload, output)
    output_hash = sha256_file(output)
    result = {
        "schema": "wal-tat-sparse-ternary-recode-commit-v1",
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
        "output_checkpoint_sha256": output_hash,
        "target_name": target_name,
        **statistics,
        "accepted_weights_before": coverage_before,
        "accepted_weights_after": coverage_after,
        "coverage_changed": False,
        "old_checkpoint_deleted": False,
        "fresh_verification_required": True,
    }
    result_path = PROJECT / f"results/{args.tag}_commit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
