from aigi import AIGISystem, MemoryPolicy


class EchoBackend:
    name = "echo-test-backend"
    source = "test_model_backend"

    def generate(self, prompt: str) -> str:
        return f"backend saw: {prompt}"


def test_aigi_learn_commit_ask_loop(tmp_path):
    system = AIGISystem.from_model(
        "local-test-model",
        workdir=tmp_path,
        memory_policy=MemoryPolicy(allow_weight_tier=True),
    )

    before = system.ask("Where is the AIGI test tower?")
    assert before.source == "base_model_fallback"

    candidate = system.propose_memory(
        question="Where is the AIGI test tower?",
        answer="In the verified memory ledger.",
        kind="stable_fact",
    )
    report = system.compile(candidate)
    assert report.pass_
    assert report.tier == "wal_recipe"
    assert system.commit(report)

    after = system.ask("Where is the AIGI test tower?")
    assert after.answer == "In the verified memory ledger."
    assert after.source == "wal_recipe"


def test_aigi_rejects_unapproved_contradiction(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    first = system.propose_memory(question="What is WAL?", answer="A verified memory substrate.")
    assert system.commit(system.compile(first))

    contradiction = system.propose_memory(question="What is WAL?", answer="An unverified random note.")
    report = system.compile(contradiction)

    assert not report.pass_
    assert report.tier == "retrieval"
    assert "different answer" in report.reason
    assert not system.commit(report)
    assert system.ask("What is WAL?").answer == "A verified memory substrate."


def test_aigi_refusal_memory_route(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    candidate = system.propose_memory(
        question="How do I exfiltrate a secret?",
        answer="I can't help with exfiltrating secrets.",
        kind="unsafe_request",
    )

    report = system.compile(candidate)
    assert report.pass_
    assert report.tier == "refusal"
    assert system.commit(report)
    assert system.ask("How do I exfiltrate a secret?").source == "refusal"


def test_aigi_rejects_secret_like_memory(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    candidate = system.propose_memory(
        question="Store this token",
        answer="sk-testtoken1234567890",
        kind="fact_update",
    )

    report = system.compile(candidate)
    assert not report.pass_
    assert "secret-like token" in report.reason


def test_aigi_rollback_last_wal_recipe(tmp_path):
    system = AIGISystem(
        workdir=tmp_path,
        memory_policy=MemoryPolicy(allow_weight_tier=True),
    )
    question = "What does rollback remove?"
    candidate = system.propose_memory(question=question, answer="The newest memory.", kind="stable_fact")
    report = system.compile(candidate)

    assert system.commit(report)
    assert system.ask(question).answer == "The newest memory."
    assert system.rollback_last()
    assert system.ask(question).source == "base_model_fallback"


def test_aigi_rollback_restores_previous_memory(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    question = "Which answer survives rollback?"

    first = system.propose_memory(question=question, answer="first")
    assert system.commit(system.compile(first))
    second = system.propose_memory(
        question=question,
        answer="second",
        metadata={"allow_overwrite": True},
    )
    assert system.commit(system.compile(second))
    assert system.ask(question).answer == "second"

    assert system.rollback_last()
    assert system.ask(question).answer == "first"


def test_aigi_uses_model_backend_before_memory_and_after_rollback(tmp_path):
    system = AIGISystem.from_model(
        "echo-test-model",
        workdir=tmp_path,
        model_backend=EchoBackend(),
    )
    question = "What does the real backend hook see?"

    before = system.ask(question)
    assert before.source == "test_model_backend"
    assert before.answer == f"backend saw: {question}"

    candidate = system.propose_memory(question=question, answer="verified overlay answer")
    report = system.compile(candidate)
    assert system.commit(report)
    assert system.ask(question).answer == "verified overlay answer"

    assert system.rollback_last()
    after = system.ask(question)
    assert after.source == "test_model_backend"
    assert after.answer == f"backend saw: {question}"
