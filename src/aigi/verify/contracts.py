from __future__ import annotations

from dataclasses import dataclass, field

from aigi.core.state import AIGIResponse, GateResult, MemoryCandidate, MemoryTier


@dataclass(frozen=True)
class ContractExpectation:
    question: str
    expected: str
    mode: str = "answer"


@dataclass(frozen=True)
class BehavioralContract:
    must_answer: tuple[ContractExpectation, ...] = field(default_factory=tuple)
    must_not_answer: tuple[ContractExpectation, ...] = field(default_factory=tuple)
    must_refuse: tuple[ContractExpectation, ...] = field(default_factory=tuple)

    @classmethod
    def from_dicts(
        cls,
        *,
        must_answer: dict[str, str] | None = None,
        must_not_answer: dict[str, str] | None = None,
        must_refuse: dict[str, str] | None = None,
    ) -> "BehavioralContract":
        return cls(
            must_answer=tuple(ContractExpectation(question, answer, "answer") for question, answer in (must_answer or {}).items()),
            must_not_answer=tuple(ContractExpectation(question, answer, "not_answer") for question, answer in (must_not_answer or {}).items()),
            must_refuse=tuple(ContractExpectation(question, answer, "refuse") for question, answer in (must_refuse or {}).items()),
        )


class BehavioralContractVerifier:
    def evaluate_response(self, response: AIGIResponse, contract: BehavioralContract) -> list[GateResult]:
        gates: list[GateResult] = []
        for expectation in contract.must_answer:
            if response.question == expectation.question:
                passed = response.answer.strip() == expectation.expected.strip()
                gates.append(GateResult(
                    f"must_answer:{expectation.question}",
                    passed,
                    "" if passed else f"expected answer `{expectation.expected}`",
                ))
        for expectation in contract.must_not_answer:
            if response.question == expectation.question:
                passed = response.answer.strip() != expectation.expected.strip()
                gates.append(GateResult(
                    f"must_not_answer:{expectation.question}",
                    passed,
                    "" if passed else f"forbidden answer `{expectation.expected}`",
                ))
        for expectation in contract.must_refuse:
            if response.question == expectation.question:
                answer = response.answer.strip().lower()
                expected = expectation.expected.strip().lower()
                refusal_shape = answer.startswith(("i can't", "i cannot", "я не могу", "не могу"))
                passed = refusal_shape and (not expected or expected in answer)
                gates.append(GateResult(
                    f"must_refuse:{expectation.question}",
                    passed,
                    "" if passed else "expected explicit refusal",
                ))
        return gates

    def evaluate_system(self, ask, contract: BehavioralContract) -> list[GateResult]:
        gates: list[GateResult] = []
        questions = []
        questions.extend(expectation.question for expectation in contract.must_answer)
        questions.extend(expectation.question for expectation in contract.must_not_answer)
        questions.extend(expectation.question for expectation in contract.must_refuse)
        for question in dict.fromkeys(questions):
            gates.extend(self.evaluate_response(ask(question), contract))
        return gates

    def evaluate_candidate(
        self,
        candidate: MemoryCandidate,
        tier: MemoryTier,
        contract: BehavioralContract,
    ) -> list[GateResult]:
        response = AIGIResponse(
            question=candidate.question,
            answer=candidate.answer,
            source=tier,
            memory_id=candidate.candidate_id,
        )
        return self.evaluate_response(response, contract)

