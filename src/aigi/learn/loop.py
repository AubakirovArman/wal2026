from __future__ import annotations

from dataclasses import dataclass

from aigi.core.system import AIGISystem
from aigi.learn.experience import Experience, LessonExtractor
from aigi.verify.contracts import BehavioralContract, BehavioralContractVerifier


@dataclass(frozen=True)
class LearningLoopResult:
    status: str
    reason: str
    committed: bool = False
    rolled_back: bool = False
    candidate_id: str | None = None

    @property
    def pass_(self) -> bool:
        return self.status == "PASS"


class VerifiedLearningLoop:
    def __init__(
        self,
        system: AIGISystem,
        *,
        contract: BehavioralContract | None = None,
    ) -> None:
        self.system = system
        self.contract = contract or BehavioralContract()
        self.extractor = LessonExtractor()
        self.contract_verifier = BehavioralContractVerifier()

    def learn_from_experience(self, experience: Experience) -> LearningLoopResult:
        lesson = self.extractor.extract(experience)
        if not lesson.accepted or lesson.candidate is None:
            return LearningLoopResult("FAIL", lesson.reason)

        report = self.system.compile(lesson.candidate)
        if not report.pass_:
            return LearningLoopResult("FAIL", report.reason, candidate_id=lesson.candidate.candidate_id)

        if not self.system.commit(report):
            return LearningLoopResult("FAIL", "commit_failed", candidate_id=lesson.candidate.candidate_id)

        gates = self.contract_verifier.evaluate_system(self.system.ask, self.contract)
        failures = [gate for gate in gates if not gate.passed]
        if failures:
            self.system.rollback_last()
            return LearningLoopResult(
                "FAIL",
                "; ".join(gate.reason for gate in failures if gate.reason) or "contract_failed",
                committed=True,
                rolled_back=True,
                candidate_id=lesson.candidate.candidate_id,
            )

        return LearningLoopResult("PASS", "committed", committed=True, candidate_id=lesson.candidate.candidate_id)

