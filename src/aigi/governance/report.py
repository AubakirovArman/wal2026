from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aigi.core.state import CompileReport, utc_now
from aigi.governance.budget import BudgetDecision


@dataclass(frozen=True)
class CommitDecisionReport:
    candidate_id: str
    tier: str
    decision: str
    committed: bool
    rolled_back: bool
    compile_status: str
    compile_gate_passed: int
    compile_gate_total: int
    budget_status: str
    risk_score: int
    risk_factors: tuple[str, ...]
    regression_status: str
    reason: str = ""
    timestamp: str = field(default_factory=utc_now)


class CommitDecisionReporter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def build(
        self,
        report: CompileReport,
        budget_decision: BudgetDecision,
        *,
        decision: str,
        committed: bool = False,
        rolled_back: bool = False,
        regression_status: str = "NOT_RUN",
        reason: str = "",
    ) -> CommitDecisionReport:
        return CommitDecisionReport(
            candidate_id=report.candidate.candidate_id,
            tier=report.tier,
            decision=decision,
            committed=committed,
            rolled_back=rolled_back,
            compile_status=report.status,
            compile_gate_passed=sum(1 for gate in report.gates if gate.passed),
            compile_gate_total=len(report.gates),
            budget_status=budget_decision.status,
            risk_score=budget_decision.risk_score,
            risk_factors=budget_decision.factors,
            regression_status=regression_status,
            reason=reason,
        )

    def write(self, decision_report: CommitDecisionReport) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision_report), ensure_ascii=False, sort_keys=True) + "\n")

    def load(self) -> list[CommitDecisionReport]:
        if not self.path.exists():
            return []
        reports = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["risk_factors"] = tuple(payload.get("risk_factors", ()))
            reports.append(CommitDecisionReport(**payload))
        return reports

