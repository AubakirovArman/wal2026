from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


MemoryTier = Literal["wal_recipe", "retrieval", "refusal", "tool", "reject"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemoryPolicy:
    weights_for: tuple[str, ...] = ("stable_fact",)
    retrieval_for: tuple[str, ...] = ("volatile_fact", "hard_fact", "fact_update")
    refusal_for: tuple[str, ...] = ("unsafe_request", "policy_refusal")
    tool_for: tuple[str, ...] = ("procedure", "tool_use")
    default_tier: MemoryTier = "retrieval"
    allow_weight_tier: bool = False


@dataclass(frozen=True)
class MemoryCandidate:
    question: str
    answer: str
    kind: str = "fact_update"
    source: str = "user"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class CompileReport:
    candidate: MemoryCandidate
    tier: MemoryTier
    pass_: bool
    gates: tuple[GateResult, ...]
    artifact_id: str | None = None
    reason: str = ""
    created_at: str = field(default_factory=utc_now)

    @property
    def status(self) -> str:
        return "PASS" if self.pass_ else "FAIL"


@dataclass(frozen=True)
class AIGIResponse:
    question: str
    answer: str
    source: str
    memory_id: str | None = None


@dataclass(frozen=True)
class CommitRecord:
    artifact_id: str
    tier: MemoryTier
    question: str
    previous_entry: dict[str, str] | None = None
    created_at: str = field(default_factory=utc_now)
