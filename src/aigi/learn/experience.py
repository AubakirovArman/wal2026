from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aigi.core.state import MemoryCandidate


@dataclass(frozen=True)
class Experience:
    question: str
    observed_answer: str
    feedback: str
    feedback_type: str = "correction"
    source: str = "feedback"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Lesson:
    accepted: bool
    reason: str
    candidate: MemoryCandidate | None = None


class LessonExtractor:
    def extract(self, experience: Experience) -> Lesson:
        question = experience.question.strip()
        feedback = experience.feedback.strip()
        observed = experience.observed_answer.strip()
        if not question:
            return Lesson(False, "empty_question")
        if not feedback:
            return Lesson(False, "empty_feedback")
        if feedback == observed:
            return Lesson(False, "feedback_matches_observed_answer")
        kind = self._kind_for(experience)
        candidate = MemoryCandidate(
            question=question,
            answer=feedback,
            kind=kind,
            source=experience.source,
            confidence=float(experience.metadata.get("confidence", 1.0)),
            metadata={
                key: value
                for key, value in experience.metadata.items()
                if key != "confidence"
            },
        )
        return Lesson(True, "candidate_extracted", candidate)

    def _kind_for(self, experience: Experience) -> str:
        explicit_kind = experience.metadata.get("kind")
        if isinstance(explicit_kind, str) and explicit_kind:
            return explicit_kind
        feedback_type = experience.feedback_type
        lowered = experience.feedback.strip().lower()
        if feedback_type in {"refusal", "safety"}:
            return "policy_refusal"
        if lowered.startswith(("i can't", "i cannot", "я не могу", "не могу")):
            return "policy_refusal"
        if feedback_type in {"procedure", "tool"}:
            return "procedure"
        if feedback_type == "stable_fact":
            return "stable_fact"
        if feedback_type == "volatile_fact":
            return "volatile_fact"
        return "fact_update"

