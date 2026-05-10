from __future__ import annotations

from pathlib import Path

from aigi.core.state import AIGIResponse, CompileReport, MemoryCandidate, MemoryPolicy
from aigi.event_log import AIGIEventLog
from aigi.memory.compiler import MemoryCompiler
from aigi.memory.retrieval_memory import RetrievalMemory
from aigi.memory.wal_memory import WALMemory
from aigi.verify.gates import MemoryVerifier


class AIGISystem:
    """Verified memory-accumulation system.

    The MVP does not claim autonomous AGI or real weight editing. The
    `wal_recipe` tier writes WAL-compatible recipe artifacts and serves them
    through a retrieval overlay until a real weight-edit backend is attached.
    """

    def __init__(
        self,
        *,
        workdir: str | Path = ".aigi",
        base_model_name: str = "local-symbolic-fallback",
        memory_policy: MemoryPolicy | None = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.base_model_name = base_model_name
        self.policy = memory_policy or MemoryPolicy()
        self.log = AIGIEventLog(self.workdir / "logs" / "events.jsonl")
        self.retrieval = RetrievalMemory(self.workdir / "memory" / "retrieval.json")
        self.refusals = RetrievalMemory(self.workdir / "memory" / "refusals.json")
        self.wal = WALMemory(self.workdir / "wal_recipes")
        self.compiler = MemoryCompiler(self.policy)
        self.verifier = MemoryVerifier(self.retrieval, self.refusals)
        self.last_report: CompileReport | None = None
        self.log.write("system_init", "PASS", {"base_model": self.base_model_name})

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        workdir: str | Path = ".aigi",
        memory_policy: MemoryPolicy | None = None,
    ) -> "AIGISystem":
        return cls(workdir=workdir, base_model_name=model, memory_policy=memory_policy)

    def ask(self, question: str) -> AIGIResponse:
        refusal = self.refusals.lookup(question)
        if refusal is not None:
            self.log.write("ask", "PASS", {"source": "refusal", "question": question})
            return AIGIResponse(question=question, answer=refusal["answer"], source="refusal", memory_id=refusal["id"])

        memory = self.retrieval.lookup(question)
        if memory is not None:
            self.log.write("ask", "PASS", {"source": "retrieval", "question": question})
            return AIGIResponse(question=question, answer=memory["answer"], source=memory["source"], memory_id=memory["id"])

        answer = "I don't know yet."
        self.log.write("ask", "PASS", {"source": "base_model_fallback", "question": question})
        return AIGIResponse(question=question, answer=answer, source="base_model_fallback")

    def propose_memory(
        self,
        *,
        question: str,
        answer: str,
        kind: str = "fact_update",
        source: str = "user",
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryCandidate:
        candidate = MemoryCandidate(
            question=question,
            answer=answer,
            kind=kind,
            source=source,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.log.write("propose_memory", "PASS", {"candidate_id": candidate.candidate_id, "kind": kind})
        return candidate

    def compile(self, candidate: MemoryCandidate) -> CompileReport:
        tier = self.compiler.select_tier(candidate)
        gates = self.verifier.evaluate(candidate, tier)
        passed = all(gate.passed for gate in gates) and tier != "reject"
        reason = "" if passed else "; ".join(gate.reason for gate in gates if not gate.passed)
        artifact_id = self.wal.preview_artifact_id(candidate) if passed and tier == "wal_recipe" else None
        report = CompileReport(
            candidate=candidate,
            tier=tier,
            pass_=passed,
            gates=tuple(gates),
            artifact_id=artifact_id,
            reason=reason,
        )
        self.last_report = report
        self.log.write(
            "compile",
            report.status,
            {"candidate_id": candidate.candidate_id, "tier": tier, "reason": reason},
        )
        return report

    def commit(self, report: CompileReport | None = None) -> bool:
        selected = report or self.last_report
        if selected is None or not selected.pass_:
            self.log.write("commit", "FAIL", {"reason": "missing_or_failed_report"})
            return False

        candidate = selected.candidate
        if selected.tier == "wal_recipe":
            artifact_id = self.wal.write_recipe(candidate)
            self.retrieval.upsert(candidate.question, candidate.answer, source="wal_recipe", memory_id=artifact_id)
        elif selected.tier == "retrieval":
            artifact_id = self.retrieval.upsert(candidate.question, candidate.answer, source="retrieval")
        elif selected.tier == "refusal":
            artifact_id = self.refusals.upsert(candidate.question, candidate.answer, source="refusal")
        elif selected.tier == "tool":
            artifact_id = self.retrieval.upsert(candidate.question, candidate.answer, source="tool_policy")
        else:
            self.log.write("commit", "FAIL", {"reason": f"unsupported_tier:{selected.tier}"})
            return False

        self.log.write("commit", "PASS", {"tier": selected.tier, "artifact_id": artifact_id})
        return True

    def rollback_last(self) -> bool:
        self.log.write("rollback", "PASS", {"mode": "no_mutating_rollback_in_mvp"})
        return True
