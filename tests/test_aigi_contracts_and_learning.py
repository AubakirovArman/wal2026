from aigi import AIGISystem, BehavioralContract, Experience, VerifiedLearningLoop
from aigi.learn.experience import LessonExtractor
from aigi.verify.contracts import BehavioralContractVerifier


def test_behavioral_contract_passes_and_fails(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    question = "What is contract fact?"
    candidate = system.propose_memory(question=question, answer="contract answer")
    assert system.commit(system.compile(candidate))

    contract = BehavioralContract.from_dicts(
        must_answer={question: "contract answer"},
        must_not_answer={question: "old answer"},
    )
    gates = BehavioralContractVerifier().evaluate_system(system.ask, contract)
    assert gates
    assert all(gate.passed for gate in gates)

    failing = BehavioralContract.from_dicts(must_answer={question: "wrong answer"})
    failing_gates = BehavioralContractVerifier().evaluate_system(system.ask, failing)
    assert any(not gate.passed for gate in failing_gates)


def test_lesson_extractor_rejects_empty_and_noop_feedback():
    extractor = LessonExtractor()
    empty = extractor.extract(Experience(question="", observed_answer="old", feedback="new"))
    noop = extractor.extract(Experience(question="q", observed_answer="same", feedback="same"))

    assert not empty.accepted
    assert empty.reason == "empty_question"
    assert not noop.accepted
    assert noop.reason == "feedback_matches_observed_answer"


def test_lesson_extractor_routes_refusal_feedback():
    lesson = LessonExtractor().extract(
        Experience(
            question="Unsafe?",
            observed_answer="Sure.",
            feedback="I can't help with that.",
            feedback_type="refusal",
        )
    )

    assert lesson.accepted
    assert lesson.candidate is not None
    assert lesson.candidate.kind == "policy_refusal"


def test_verified_learning_loop_commits_good_feedback(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    question = "What did feedback teach?"
    contract = BehavioralContract.from_dicts(must_answer={question: "feedback answer"})
    loop = VerifiedLearningLoop(system, contract=contract)

    result = loop.learn_from_experience(
        Experience(question=question, observed_answer="I don't know yet.", feedback="feedback answer")
    )

    assert result.pass_
    assert result.committed
    assert system.ask(question).answer == "feedback answer"


def test_verified_learning_loop_rolls_back_contract_violation(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    question = "What answer is protected?"
    baseline = system.propose_memory(question=question, answer="baseline")
    assert system.commit(system.compile(baseline))
    contract = BehavioralContract.from_dicts(must_answer={question: "baseline"})
    loop = VerifiedLearningLoop(system, contract=contract)

    result = loop.learn_from_experience(
        Experience(
            question=question,
            observed_answer="baseline",
            feedback="bad overwrite",
            metadata={"allow_overwrite": True},
        )
    )

    assert not result.pass_
    assert result.rolled_back
    assert system.ask(question).answer == "baseline"

