from __future__ import annotations

from dataclasses import dataclass, field

from aigi.core.state import MemoryCandidate, MemoryTier


@dataclass(frozen=True)
class MemoryChangeBudget:
    max_risk_score: int = 5
    max_answer_chars: int = 500
    allowed_tiers: tuple[MemoryTier, ...] = ("wal_recipe", "retrieval", "refusal", "tool")
    allow_overwrite: bool = False
    require_contract_for_overwrite: bool = True


@dataclass(frozen=True)
class BudgetDecision:
    passed: bool
    risk_score: int
    factors: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


class MemoryBudgetEvaluator:
    def __init__(self, budget: MemoryChangeBudget | None = None) -> None:
        self.budget = budget or MemoryChangeBudget()

    def evaluate(
        self,
        candidate: MemoryCandidate,
        tier: MemoryTier,
        *,
        existing_entry: dict[str, str] | None = None,
        contract_present: bool = False,
    ) -> BudgetDecision:
        factors: list[str] = []
        risk_score = 0
        hard_failures: list[str] = []

        if tier not in self.budget.allowed_tiers:
            hard_failures.append(f"tier_not_allowed:{tier}")
            factors.append("tier_not_allowed")
            risk_score += 5

        if len(candidate.answer) > self.budget.max_answer_chars:
            hard_failures.append("answer_too_long")
            factors.append("answer_too_long")
            risk_score += 2

        if candidate.confidence < 0.5:
            factors.append("low_confidence")
            risk_score += 3
        elif candidate.confidence < 0.8:
            factors.append("medium_confidence")
            risk_score += 1

        if tier == "wal_recipe":
            factors.append("weight_tier_candidate")
            risk_score += 1

        if tier == "tool":
            factors.append("tool_policy_candidate")
            risk_score += 1

        if tier == "refusal":
            factors.append("refusal_candidate")
            risk_score += 1

        if existing_entry is not None and existing_entry.get("answer", "").strip() != candidate.answer.strip():
            factors.append("overwrite")
            risk_score += 3
            approved_by_candidate = bool(candidate.metadata.get("allow_overwrite"))
            if not self.budget.allow_overwrite and not approved_by_candidate:
                hard_failures.append("overwrite_not_approved_by_budget")
            if self.budget.require_contract_for_overwrite and not contract_present:
                hard_failures.append("overwrite_requires_contract")

        if risk_score > self.budget.max_risk_score:
            hard_failures.append("risk_budget_exceeded")

        passed = not hard_failures
        return BudgetDecision(
            passed=passed,
            risk_score=risk_score,
            factors=tuple(factors),
            reason="; ".join(hard_failures),
        )

