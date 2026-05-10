from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aigi.core.state import MemoryCandidate, MemoryTier, utc_now
from aigi.governance.budget import BudgetDecision


@dataclass(frozen=True)
class RiskEntry:
    candidate_id: str
    question: str
    tier: MemoryTier
    risk_score: int
    factors: tuple[str, ...]
    outcome: str
    committed: bool = False
    rolled_back: bool = False
    timestamp: str = field(default_factory=utc_now)


class RiskLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        candidate: MemoryCandidate,
        tier: MemoryTier,
        decision: BudgetDecision,
        *,
        outcome: str,
        committed: bool = False,
        rolled_back: bool = False,
    ) -> RiskEntry:
        entry = RiskEntry(
            candidate_id=candidate.candidate_id,
            question=candidate.question,
            tier=tier,
            risk_score=decision.risk_score,
            factors=decision.factors,
            outcome=outcome,
            committed=committed,
            rolled_back=rolled_back,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[RiskEntry]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["factors"] = tuple(payload.get("factors", ()))
            records.append(RiskEntry(**payload))
        return records

    def summary(self) -> dict[str, int]:
        entries = self.entries()
        active_debt = sum(entry.risk_score for entry in entries if entry.committed and not entry.rolled_back)
        rolled_back_debt = sum(entry.risk_score for entry in entries if entry.rolled_back)
        rejected_debt = sum(entry.risk_score for entry in entries if not entry.committed)
        return {
            "entries": len(entries),
            "active_debt": active_debt,
            "rolled_back_debt": rolled_back_debt,
            "rejected_debt": rejected_debt,
        }

