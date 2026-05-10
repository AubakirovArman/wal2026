from aigi import (
    AIGISystem,
    BehavioralContract,
    CommitDecisionReporter,
    ContractRegressionSuite,
    Experience,
    MemoryBudgetEvaluator,
    MemoryChangeBudget,
    RiskLedger,
    VerifiedLearningLoop,
)


def test_memory_budget_rejects_uncontracted_overwrite(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    question = "What does budget protect?"
    baseline = system.propose_memory(question=question, answer="baseline")
    assert system.commit(system.compile(baseline))

    overwrite = system.propose_memory(
        question=question,
        answer="new value",
        metadata={"allow_overwrite": True},
    )
    report = system.compile(overwrite)
    decision = MemoryBudgetEvaluator(MemoryChangeBudget()).evaluate(
        overwrite,
        report.tier,
        existing_entry=system.retrieval.lookup(question),
        contract_present=False,
    )

    assert not decision.passed
    assert decision.risk_score == 3
    assert "overwrite_requires_contract" in decision.reason


def test_risk_ledger_records_active_rejected_and_rolled_back_debt(tmp_path):
    system = AIGISystem(workdir=tmp_path / "system")
    ledger = RiskLedger(tmp_path / "risk.jsonl")
    reporter = CommitDecisionReporter(tmp_path / "decisions.jsonl")

    accepted_loop = VerifiedLearningLoop(
        system,
        contract=BehavioralContract.from_dicts(must_answer={"Accepted?": "yes"}),
        risk_ledger=ledger,
        decision_reporter=reporter,
    )
    accepted = accepted_loop.learn_from_experience(Experience("Accepted?", "old", "yes"))
    assert accepted.pass_

    protected_question = "Protected?"
    baseline = system.propose_memory(question=protected_question, answer="baseline")
    assert system.commit(system.compile(baseline))
    rollback_loop = VerifiedLearningLoop(
        system,
        contract=BehavioralContract.from_dicts(must_answer={protected_question: "baseline"}),
        risk_ledger=ledger,
        decision_reporter=reporter,
    )
    rolled_back = rollback_loop.learn_from_experience(
        Experience(
            protected_question,
            "baseline",
            "bad",
            metadata={"allow_overwrite": True},
        )
    )
    assert not rolled_back.pass_
    assert rolled_back.rolled_back

    rejected_loop = VerifiedLearningLoop(
        system,
        risk_ledger=ledger,
        decision_reporter=reporter,
    )
    rejected = rejected_loop.learn_from_experience(
        Experience(
            protected_question,
            "baseline",
            "bad again",
            metadata={"allow_overwrite": True},
        )
    )
    assert not rejected.pass_

    summary = ledger.summary()
    assert summary["entries"] == 3
    assert summary["active_debt"] == accepted.risk_score
    assert summary["rolled_back_debt"] == rolled_back.risk_score
    assert summary["rejected_debt"] == rejected.risk_score
    assert [report.decision for report in reporter.load()] == ["ACCEPTED", "ROLLED_BACK", "REJECTED"]


def test_contract_regression_suite_protects_multiple_facts(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    protected = {
        "Protected fact 1?": "answer 1",
        "Protected fact 2?": "answer 2",
        "Protected fact 3?": "answer 3",
    }
    for question, answer in protected.items():
        assert system.commit(system.compile(system.propose_memory(question=question, answer=answer)))

    suite = ContractRegressionSuite.from_contract(BehavioralContract.from_dicts(must_answer=protected))
    assert suite.evaluate_system(system.ask).pass_

    loop = VerifiedLearningLoop(
        system,
        contract=BehavioralContract.from_dicts(must_answer={"New fact?": "new answer"}),
        regression_suite=suite,
    )
    result = loop.learn_from_experience(Experience("New fact?", "old", "new answer"))
    assert result.pass_
    assert suite.evaluate_system(system.ask).pass_


def test_regression_suite_rolls_back_bad_protected_update(tmp_path):
    system = AIGISystem(workdir=tmp_path)
    question = "Protected regression fact?"
    assert system.commit(system.compile(system.propose_memory(question=question, answer="baseline")))
    suite = ContractRegressionSuite.from_contract(BehavioralContract.from_dicts(must_answer={question: "baseline"}))
    loop = VerifiedLearningLoop(
        system,
        contract=BehavioralContract.from_dicts(must_answer={question: "bad"}),
        regression_suite=suite,
    )

    result = loop.learn_from_experience(
        Experience(question, "baseline", "bad", metadata={"allow_overwrite": True})
    )

    assert not result.pass_
    assert result.rolled_back
    assert system.ask(question).answer == "baseline"

