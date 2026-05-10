from __future__ import annotations

from dataclasses import dataclass

from aigi.core.system import AIGISystem
from aigi.governance.budget import BudgetDecision, MemoryBudgetEvaluator, MemoryChangeBudget
from aigi.governance.report import CommitDecisionReport, CommitDecisionReporter
from aigi.governance.risk import RiskLedger
from aigi.learn.experience import Experience, LessonExtractor
from aigi.verify.contracts import BehavioralContract, BehavioralContractVerifier
from aigi.verify.regression import ContractRegressionSuite


@dataclass(frozen=True)
class LearningLoopResult:
    status: str
    reason: str
    committed: bool = False
    rolled_back: bool = False
    candidate_id: str | None = None
    risk_score: int = 0
    decision_report: CommitDecisionReport | None = None

    @property
    def pass_(self) -> bool:
        return self.status == "PASS"


class VerifiedLearningLoop:
    def __init__(
        self,
        system: AIGISystem,
        *,
        contract: BehavioralContract | None = None,
        budget: MemoryChangeBudget | None = None,
        risk_ledger: RiskLedger | None = None,
        regression_suite: ContractRegressionSuite | None = None,
        decision_reporter: CommitDecisionReporter | None = None,
    ) -> None:
        self.system = system
        self.contract = contract or BehavioralContract()
        self.extractor = LessonExtractor()
        self.contract_verifier = BehavioralContractVerifier()
        self.budget_evaluator = MemoryBudgetEvaluator(budget)
        self.risk_ledger = risk_ledger
        self.regression_suite = regression_suite
        self.decision_reporter = decision_reporter

    def learn_from_experience(self, experience: Experience) -> LearningLoopResult:
        lesson = self.extractor.extract(experience)
        if not lesson.accepted or lesson.candidate is None:
            return LearningLoopResult("FAIL", lesson.reason)

        report = self.system.compile(lesson.candidate)
        if not report.pass_:
            return LearningLoopResult("FAIL", report.reason, candidate_id=lesson.candidate.candidate_id)

        existing_entry = self.system.refusals.lookup(lesson.candidate.question) if report.tier == "refusal" else self.system.retrieval.lookup(lesson.candidate.question)
        budget_decision = self.budget_evaluator.evaluate(
            lesson.candidate,
            report.tier,
            existing_entry=existing_entry,
            contract_present=bool(self.contract.must_answer or self.contract.must_not_answer or self.contract.must_refuse),
        )
        if not budget_decision.passed:
            self._record_risk(report, budget_decision, outcome="budget_rejected")
            decision_report = self._write_decision_report(
                report,
                budget_decision,
                decision="REJECTED",
                reason=budget_decision.reason,
            )
            return LearningLoopResult(
                "FAIL",
                budget_decision.reason,
                candidate_id=lesson.candidate.candidate_id,
                risk_score=budget_decision.risk_score,
                decision_report=decision_report,
            )

        if not self.system.commit(report):
            self._record_risk(report, budget_decision, outcome="commit_failed")
            decision_report = self._write_decision_report(report, budget_decision, decision="REJECTED", reason="commit_failed")
            return LearningLoopResult(
                "FAIL",
                "commit_failed",
                candidate_id=lesson.candidate.candidate_id,
                risk_score=budget_decision.risk_score,
                decision_report=decision_report,
            )

        gates = self.contract_verifier.evaluate_system(self.system.ask, self.contract)
        failures = [gate for gate in gates if not gate.passed]
        regression_result = self.regression_suite.evaluate_system(self.system.ask) if self.regression_suite is not None else None
        regression_failures = list(regression_result.failures) if regression_result is not None else []
        failures.extend(regression_failures)
        if failures:
            self.system.rollback_last()
            self._record_risk(report, budget_decision, outcome="rolled_back", committed=True, rolled_back=True)
            decision_report = self._write_decision_report(
                report,
                budget_decision,
                decision="ROLLED_BACK",
                committed=True,
                rolled_back=True,
                regression_status=regression_result.status if regression_result is not None else "FAIL",
                reason="; ".join(gate.reason for gate in failures if gate.reason) or "contract_failed",
            )
            return LearningLoopResult(
                "FAIL",
                "; ".join(gate.reason for gate in failures if gate.reason) or "contract_failed",
                committed=True,
                rolled_back=True,
                candidate_id=lesson.candidate.candidate_id,
                risk_score=budget_decision.risk_score,
                decision_report=decision_report,
            )

        self._record_risk(report, budget_decision, outcome="accepted", committed=True)
        decision_report = self._write_decision_report(
            report,
            budget_decision,
            decision="ACCEPTED",
            committed=True,
            regression_status=regression_result.status if regression_result is not None else "PASS",
            reason="committed",
        )
        return LearningLoopResult(
            "PASS",
            "committed",
            committed=True,
            candidate_id=lesson.candidate.candidate_id,
            risk_score=budget_decision.risk_score,
            decision_report=decision_report,
        )

    def _record_risk(
        self,
        report,
        budget_decision: BudgetDecision,
        *,
        outcome: str,
        committed: bool = False,
        rolled_back: bool = False,
    ) -> None:
        if self.risk_ledger is None:
            return
        self.risk_ledger.append(
            report.candidate,
            report.tier,
            budget_decision,
            outcome=outcome,
            committed=committed,
            rolled_back=rolled_back,
        )

    def _write_decision_report(
        self,
        report,
        budget_decision: BudgetDecision,
        *,
        decision: str,
        committed: bool = False,
        rolled_back: bool = False,
        regression_status: str = "NOT_RUN",
        reason: str = "",
    ) -> CommitDecisionReport | None:
        if self.decision_reporter is None:
            return None
        decision_report = self.decision_reporter.build(
            report,
            budget_decision,
            decision=decision,
            committed=committed,
            rolled_back=rolled_back,
            regression_status=regression_status,
            reason=reason,
        )
        self.decision_reporter.write(decision_report)
        return decision_report
