from __future__ import annotations

from aigi.core.state import MemoryCandidate, MemoryPolicy, MemoryTier


class MemoryCompiler:
    def __init__(self, policy: MemoryPolicy) -> None:
        self.policy = policy

    def select_tier(self, candidate: MemoryCandidate) -> MemoryTier:
        if candidate.confidence <= 0:
            return "reject"
        if candidate.kind in self.policy.refusal_for:
            return "refusal"
        if candidate.kind in self.policy.tool_for:
            return "tool"
        if candidate.kind in self.policy.weights_for:
            return "wal_recipe" if self.policy.allow_weight_tier else "retrieval"
        if candidate.kind in self.policy.retrieval_for:
            return "retrieval"
        return self.policy.default_tier
