"""Quality gates that atomically commit or roll back a ternary transaction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from .transaction import TransactionalTernaryMatrix
from .wal import HashChainWAL


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    ratios: Dict[str, float]
    violations: Dict[str, float]


class RatioGate:
    """Accept when every candidate loss is within its baseline ratio limit."""

    def __init__(self, limits: float | Mapping[str, float] = 1.02):
        self.limits = float(limits) if isinstance(limits, (float, int)) else dict(limits)

    def evaluate(
        self, baseline: Mapping[str, float], candidate: Mapping[str, float]
    ) -> GateDecision:
        if baseline.keys() != candidate.keys() or not baseline:
            raise ValueError("baseline and candidate must have identical non-empty domains")
        ratios: Dict[str, float] = {}
        violations: Dict[str, float] = {}
        for domain, before in baseline.items():
            if before <= 0:
                raise ValueError(f"baseline for {domain!r} must be positive")
            ratio = float(candidate[domain]) / float(before)
            limit = self.limits if isinstance(self.limits, float) else self.limits[domain]
            ratios[domain] = ratio
            if ratio > limit:
                violations[domain] = ratio - limit
        return GateDecision(not violations, ratios, violations)


class TransactionController:
    """Connect an in-memory matrix transaction to the durable WAL v2."""

    def __init__(self, wal: HashChainWAL, matrix_name: str, gate: RatioGate | None = None):
        self.wal = wal
        self.matrix_name = matrix_name
        self.gate = gate or RatioGate()

    def begin(
        self,
        matrix: TransactionalTernaryMatrix,
        mask: torch.Tensor,
        *,
        selector: str,
        selected_damage_mean: float | None = None,
    ) -> str:
        transaction_id = matrix.begin(mask)
        mask_bytes = mask.detach().to("cpu", dtype=torch.uint8).numpy().tobytes()
        self.wal.append(
            "begin",
            transaction_id,
            {
                "matrix": self.matrix_name,
                "selector": selector,
                "groups": int(mask.sum().item()),
                "total_groups": mask.numel(),
                "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                "selected_damage_mean": selected_damage_mean,
            },
        )
        return transaction_id

    def progress(
        self,
        matrix: TransactionalTernaryMatrix,
        *,
        step: int,
        pressure: float,
        temperature: float,
        loss: float | None = None,
    ) -> None:
        if not matrix.in_transaction:
            raise RuntimeError("no active transaction")
        matrix.set_candidate_state(pressure, temperature)
        self.wal.append(
            "progress",
            matrix.transaction_id,
            {
                "matrix": self.matrix_name,
                "step": int(step),
                "pressure": float(pressure),
                "temperature": float(temperature),
                "loss": loss,
                "code_churn": matrix.current_code_churn(),
            },
        )

    def decide(
        self,
        matrix: TransactionalTernaryMatrix,
        *,
        baseline: Mapping[str, float],
        candidate: Mapping[str, float],
    ) -> GateDecision:
        if not matrix.in_transaction:
            raise RuntimeError("no active transaction")
        transaction_id = matrix.transaction_id
        decision = self.gate.evaluate(baseline, candidate)
        kind = "commit" if decision.passed else "rollback"
        gate_payload = {
            "matrix": self.matrix_name,
            "baseline": dict(baseline),
            "candidate": dict(candidate),
            "ratios": decision.ratios,
            "violations": decision.violations,
        }
        # This record is durable before the in-memory state transition. If the
        # process stops here, recovery can distinguish an intent from an
        # applied commit/rollback and restart from the last model checkpoint.
        self.wal.append(f"{kind}_intent", transaction_id, gate_payload)
        if decision.passed:
            result = matrix.commit()
        else:
            result = matrix.rollback()
        self.wal.append(
            kind,
            transaction_id,
            {
                **gate_payload,
                **result,
            },
        )
        return decision
