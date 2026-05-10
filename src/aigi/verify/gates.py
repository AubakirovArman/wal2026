from __future__ import annotations

import re

from aigi.core.state import GateResult, MemoryCandidate, MemoryTier
from aigi.memory.retrieval_memory import RetrievalMemory


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*[A-Za-z0-9_-]{8,}"),
)


class MemoryVerifier:
    def __init__(self, retrieval: RetrievalMemory, refusals: RetrievalMemory) -> None:
        self.retrieval = retrieval
        self.refusals = refusals

    def evaluate(self, candidate: MemoryCandidate, tier: MemoryTier) -> list[GateResult]:
        return [
            self._has_question_and_answer(candidate),
            self._confidence_in_range(candidate),
            self._no_secret_leak(candidate),
            self._no_unapproved_contradiction(candidate, tier),
            self._refusal_shape(candidate, tier),
        ]

    def _has_question_and_answer(self, candidate: MemoryCandidate) -> GateResult:
        passed = bool(candidate.question.strip()) and bool(candidate.answer.strip())
        return GateResult("has_question_and_answer", passed, "" if passed else "empty question or answer")

    def _confidence_in_range(self, candidate: MemoryCandidate) -> GateResult:
        passed = 0.0 <= candidate.confidence <= 1.0
        return GateResult("confidence_in_range", passed, "" if passed else "confidence outside [0, 1]")

    def _no_secret_leak(self, candidate: MemoryCandidate) -> GateResult:
        text = f"{candidate.question}\n{candidate.answer}"
        passed = not any(pattern.search(text) for pattern in SECRET_PATTERNS)
        return GateResult("no_secret_leak", passed, "" if passed else "secret-like token detected")

    def _no_unapproved_contradiction(self, candidate: MemoryCandidate, tier: MemoryTier) -> GateResult:
        if tier in {"refusal", "reject"}:
            return GateResult("no_unapproved_contradiction", True)
        existing = self.retrieval.lookup(candidate.question)
        if existing is None:
            return GateResult("no_unapproved_contradiction", True)
        allow_overwrite = bool(candidate.metadata.get("allow_overwrite"))
        same_answer = existing["answer"].strip() == candidate.answer.strip()
        passed = same_answer or allow_overwrite
        return GateResult(
            "no_unapproved_contradiction",
            passed,
            "" if passed else "existing memory has different answer and overwrite is not approved",
        )

    def _refusal_shape(self, candidate: MemoryCandidate, tier: MemoryTier) -> GateResult:
        if tier != "refusal":
            return GateResult("refusal_shape", True)
        lowered = candidate.answer.strip().lower()
        passed = lowered.startswith(("i can't", "i cannot", "я не могу", "не могу"))
        return GateResult("refusal_shape", passed, "" if passed else "refusal answer must be explicit")
