"""Prospectively validate and commit one frozen partial ternary artifact.

The source checkpoint is immutable.  A new checkpoint is published only when
the policy, recovery result, artifact, and every validation audit agree by hash
and satisfy limits declared before the relevant one-shot audit was opened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from compensation_window import PROJECT, sha256_file
from wal_tat import accepted_weight_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("recovery_result", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("audits", type=Path, nargs="+")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configured_validation_suites(policy: dict) -> list[dict[str, str]]:
    """Normalize recurring and one-shot validation policy schemas.

    Older prospective policies list multiple recurring suites, while newer
    policies seal one audit-only suite in ``one_shot_audit``.  Both schemas
    bind the suite by SHA-256 before the candidate is evaluated.
    """
    validation_policy = policy.get("rotating_validation") or policy.get(
        "recurring_validation"
    )
    if not validation_policy:
        one_shot = policy.get("one_shot_audit")
        if isinstance(one_shot, dict) and one_shot.get("sha256"):
            validation_policy = [
                {"name": "one_shot_audit", "sha256": one_shot["sha256"]}
            ]
    if not validation_policy:
        sealed = policy.get("sealed_audit")
        if isinstance(sealed, dict) and sealed.get("sha256"):
            validation_policy = [
                {"name": "sealed_audit", "sha256": sealed["sha256"]}
            ]
    require(bool(validation_policy), "policy has no validation suites")
    normalized = [
        {"name": str(item["name"]), "sha256": str(item["sha256"])}
        for item in validation_policy
    ]
    require(
        len({item["name"] for item in normalized}) == len(normalized),
        "policy contains duplicate validation suite names",
    )
    require(
        len({item["sha256"] for item in normalized}) == len(normalized),
        "policy contains duplicate validation suite hashes",
    )
    return normalized


def validate_audits(
    audits: list[dict],
    policy: dict,
    *,
    checkpoint_hash: str,
    artifact_hash: str,
) -> dict:
    validation_policy = configured_validation_suites(policy)
    candidate_policy = policy.get("candidate") or policy.get("frozen_candidate")
    require(isinstance(candidate_policy, dict), "policy has no candidate")
    configured = {item["sha256"]: item["name"] for item in validation_policy}
    required_names = set(
        policy["acceptance"].get(
            "validation_suites_required", configured.values()
        )
    )
    observed_names = set()
    cumulative_limit = float(
        policy["acceptance"]["cumulative_point_ratio_max_each_domain"]
    )
    incremental_limit = float(
        policy["acceptance"]["incremental_paired_upper_ratio_max_each_domain"]
    )
    summary = {}
    for audit in audits:
        require(
            audit.get("schema") == "wal-tat-partial-initializer-artifact-audit-v1",
            "unsupported audit schema",
        )
        require(audit["checkpoint_sha256"] == checkpoint_hash, "audit checkpoint mismatch")
        require(audit["artifact_sha256"] == artifact_hash, "audit artifact mismatch")
        suite_hash = audit["suite_sha256"]
        require(suite_hash in configured, "audit suite is absent from policy")
        name = configured[suite_hash]
        require(name not in observed_names, "duplicate recurring-validation audit")
        observed_names.add(name)
        require(
            int(audit["candidate_groups"]) == int(candidate_policy["groups"]),
            "audit candidate group count mismatch",
        )
        cumulative = audit["cumulative_candidate_vs_bf16"]
        incremental = audit["incremental_candidate_vs_frontier"]
        require(cumulative and incremental, "audit domains are empty")
        cumulative_worst = max(float(value["observed_ratio"]) for value in cumulative.values())
        incremental_upper_worst = max(
            float(value["ratio_ci"]["upper"]) for value in incremental.values()
        )
        require(
            all(float(value["observed_ratio"]) <= cumulative_limit for value in cumulative.values()),
            f"{name} fails cumulative point gate",
        )
        require(
            all(float(value["ratio_ci"]["upper"]) <= incremental_limit for value in incremental.values()),
            f"{name} fails incremental upper-confidence gate",
        )
        summary[name] = {
            "suite_sha256": suite_hash,
            "cumulative_worst_observed_ratio": cumulative_worst,
            "incremental_worst_upper_ratio": incremental_upper_worst,
            "cumulative_limit": cumulative_limit,
            "incremental_upper_limit": incremental_limit,
        }
    require(observed_names == required_names, "required recurring-validation suites are missing")
    return summary


def publish_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "output checkpoint already exists")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    recovery_result_path = args.recovery_result.resolve()
    policy_path = args.policy.resolve()
    audit_paths = [path.resolve() for path in args.audits]
    output_checkpoint = args.output_checkpoint.resolve()
    checkpoint_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    policy_hash = sha256_file(policy_path)
    policy = load_json(policy_path)
    recovery = load_json(recovery_result_path)
    audits = [load_json(path) for path in audit_paths]

    policy_schema = policy.get("schema")
    require(
        policy_schema
        in {
            "wal-tat-predeclared-candidate-policy-v1",
            "wal-tat-predeclared-one-shot-audit-policy-v1",
        },
        "unsupported policy schema",
    )
    if policy_schema == "wal-tat-predeclared-candidate-policy-v1":
        require(
            policy.get("created_before_candidate_run") is True,
            "candidate policy is not prospective",
        )
    else:
        require(
            policy.get("declared_before_execution") is True
            or policy.get("created_before_audit_run") is True,
            "audit policy is not prospective",
        )
    candidate_policy = policy.get("candidate") or policy.get("frozen_candidate")
    require(isinstance(candidate_policy, dict), "policy has no candidate")
    require(policy["source_frontier"]["sha256"] == checkpoint_hash, "policy frontier mismatch")
    require(recovery["checkpoint_sha256"] == checkpoint_hash, "recovery checkpoint mismatch")
    require(recovery["output_artifact_sha256"] == artifact_hash, "recovery artifact mismatch")
    require(recovery["checkpoint_mutated"] is False, "recovery mutated source checkpoint")
    require(
        int(recovery["norm_parameters_trained"])
        == int(
            policy["acceptance"].get(
                "norm_parameters_trained_must_equal",
                candidate_policy.get("norm_parameters_trained", 0),
            )
        ),
        "recovery trained forbidden norm parameters",
    )
    require(
        int(recovery["previously_committed_groups_trainable"])
        == int(
            policy["acceptance"].get(
                "previously_committed_groups_trainable_must_equal",
                candidate_policy.get("previously_committed_groups_trainable", 0),
            )
        ),
        "recovery trained previously committed groups",
    )
    validation = validate_audits(
        audits,
        policy,
        checkpoint_hash=checkpoint_hash,
        artifact_hash=artifact_hash,
    )

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    require(
        artifact.get("schema") == "wal-tat-partial-initializer-artifact-v1",
        "unsupported artifact schema",
    )
    require(artifact["source_checkpoint_sha256"] == checkpoint_hash, "artifact source mismatch")
    target_name = candidate_policy["target_name"]
    require(artifact["target_name"] == target_name, "artifact target mismatch")
    require(int(artifact["group_size"]) == int(candidate_policy["group_size"]), "group size mismatch")
    require(
        all(entry["initializer"] == candidate_policy["initializer"] for entry in artifact["entries"]),
        "artifact initializer differs from policy",
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before_counts = accepted_weight_counts(payload)
    entry = payload["matrices"][target_name]
    mask = entry["committed_mask"].bool().clone()
    codes = entry["ternary_codes_int8"].to(torch.int8).clone()
    scales = entry["scales_fp16"].half().clone()
    group_size = int(entry["group_size"])
    grouped_codes = codes.view(mask.shape[0], mask.shape[1], group_size)
    flat_mask = mask.reshape(-1)
    flat_codes = grouped_codes.reshape(-1, group_size)
    flat_scales = scales.reshape(-1)
    all_indices = []
    for artifact_entry in artifact["entries"]:
        indices = artifact_entry["flat_group_indices"].long()
        require(torch.all((indices >= 0) & (indices < flat_mask.numel())).item(), "artifact index out of bounds")
        require(not torch.any(flat_mask.index_select(0, indices)).item(), "artifact overlaps committed groups")
        all_indices.append(indices)
        new_codes = artifact_entry["codes"].to(torch.int8)
        new_scales = artifact_entry["scales"].half()
        require(tuple(new_codes.shape) == (indices.numel(), group_size), "artifact code shape mismatch")
        require(tuple(new_scales.shape) == (indices.numel(),), "artifact scale shape mismatch")
        require(set(new_codes.unique().tolist()) <= {-1, 0, 1}, "artifact contains non-ternary codes")
        require(torch.isfinite(new_scales).all().item(), "artifact contains non-finite scales")
        require(torch.all(new_scales > 0).item(), "artifact contains non-positive scales")
        flat_mask[indices] = True
        flat_codes[indices] = new_codes
        flat_scales[indices] = new_scales
    indices = torch.cat(all_indices)
    require(indices.unique().numel() == indices.numel(), "artifact entries overlap")
    require(indices.numel() == int(candidate_policy["groups"]), "candidate group count mismatch")

    entry["committed_mask"] = mask
    entry["ternary_codes_int8"] = grouped_codes.reshape_as(codes)
    entry["scales_fp16"] = scales
    payload["parent_sha256"] = checkpoint_hash
    payload["metrics"] = recovery["final_full_development"]
    payload["gate_ratios"] = recovery["final_full_ratios_to_bf16"]
    payload["metadata"] = {
        "experiment": args.tag,
        "mode": (
            f"prospective D{candidate_policy.get('sensitivity_decile', 'unknown')} "
            f"{candidate_policy['initializer'].replace('_', '-')} "
            "hard-forward scale recovery"
        ),
        "target_name": target_name,
        "candidate_new_groups": int(indices.numel()),
        "candidate_new_weights": int(indices.numel() * group_size),
        "artifact_sha256": artifact_hash,
        "policy_sha256": policy_hash,
        "recovery_result_sha256": sha256_file(recovery_result_path),
        "audit_result_sha256": {
            name: sha256_file(path) for name, path in zip(validation, audit_paths)
        },
        "norm_parameters_trained": 0,
        "previously_committed_groups_trainable": 0,
    }
    after_counts = accepted_weight_counts(payload)
    added = sum(after_counts.values()) - sum(before_counts.values())
    require(added == int(candidate_policy["weights"]), "accepted weight delta mismatch")
    require(
        sum(after_counts.values()) == int(policy["prospective_if_accepted"]["accepted_weights"]),
        "prospective accepted weight total mismatch",
    )
    require(sha256_file(checkpoint) == checkpoint_hash, "source checkpoint changed before publish")
    publish_checkpoint(output_checkpoint, payload)
    require(sha256_file(checkpoint) == checkpoint_hash, "source checkpoint changed during publish")

    result = {
        "schema": "wal-tat-prospective-artifact-commit-v1",
        "status": "published_pending_fresh_verification",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "output_checkpoint_bytes": output_checkpoint.stat().st_size,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "policy": str(policy_path),
        "policy_sha256": policy_hash,
        "recovery_result": str(recovery_result_path),
        "recovery_result_sha256": sha256_file(recovery_result_path),
        "audit_results": [
            {"path": str(path), "sha256": sha256_file(path)} for path in audit_paths
        ],
        "validation": validation,
        "candidate_groups": int(indices.numel()),
        "candidate_weights": added,
        "accepted_weights_before": sum(before_counts.values()),
        "accepted_weights_after": sum(after_counts.values()),
        "coverage_by_matrix": after_counts,
        "source_checkpoint_mutated": False,
    }
    output = PROJECT / f"results/{args.tag}_commit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
