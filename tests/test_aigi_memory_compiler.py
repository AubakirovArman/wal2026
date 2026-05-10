from aigi import MemoryCandidate, MemoryPolicy
from aigi.memory.compiler import MemoryCompiler


def test_memory_compiler_routes_stable_fact_to_wal_when_enabled():
    compiler = MemoryCompiler(MemoryPolicy(allow_weight_tier=True))
    candidate = MemoryCandidate(question="q", answer="a", kind="stable_fact")
    assert compiler.select_tier(candidate) == "wal_recipe"


def test_memory_compiler_routes_stable_fact_to_retrieval_by_default():
    compiler = MemoryCompiler(MemoryPolicy())
    candidate = MemoryCandidate(question="q", answer="a", kind="stable_fact")
    assert compiler.select_tier(candidate) == "retrieval"


def test_memory_compiler_rejects_zero_confidence():
    compiler = MemoryCompiler(MemoryPolicy())
    candidate = MemoryCandidate(question="q", answer="a", confidence=0)
    assert compiler.select_tier(candidate) == "reject"
