import json

import pytest
import torch

from wal_tat import (
    HashChainWAL,
    RatioGate,
    TransactionController,
    TransactionalTernaryMatrix,
    WALIntegrityError,
)


def test_hash_chain_round_trip(tmp_path):
    path = tmp_path / "transactions.jsonl"
    wal = HashChainWAL(path, fsync=False)
    first = wal.append("begin", "tx-1", {"groups": 2})
    second = wal.append("commit", "tx-1", {"ratios": {"wiki": 1.01}})
    records = HashChainWAL(path, fsync=False).verify()
    assert [record.kind for record in records] == ["begin", "commit"]
    assert second.previous_hash == first.digest


def test_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "transactions.jsonl"
    wal = HashChainWAL(path, fsync=False)
    wal.append("begin", "tx-1", {"groups": 2})
    raw = json.loads(path.read_text())
    raw["payload"]["groups"] = 999
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(WALIntegrityError, match="digest mismatch"):
        HashChainWAL(path, fsync=False)


def test_controller_commits_passing_candidate(tmp_path):
    matrix = TransactionalTernaryMatrix(torch.randn(2, 4), group_size=2)
    controller = TransactionController(
        HashChainWAL(tmp_path / "pass.jsonl", fsync=False), "toy", RatioGate(1.02)
    )
    mask = torch.tensor([[True, False], [False, False]])
    controller.begin(matrix, mask, selector="causal")
    decision = controller.decide(
        matrix, baseline={"wiki": 2.0, "code": 3.0}, candidate={"wiki": 2.01, "code": 3.05}
    )
    assert decision.passed
    assert matrix.committed_mask.sum() == 1
    assert controller.wal.verify()[-1].kind == "commit"
    assert [record.kind for record in controller.wal.verify()] == [
        "begin", "commit_intent", "commit"
    ]


def test_controller_rolls_back_failing_candidate(tmp_path):
    weight = torch.randn(2, 4)
    matrix = TransactionalTernaryMatrix(weight, group_size=2)
    controller = TransactionController(HashChainWAL(tmp_path / "fail.jsonl", fsync=False), "toy")
    mask = torch.tensor([[True, False], [False, False]])
    controller.begin(matrix, mask, selector="causal")
    with torch.no_grad():
        matrix.master_weight[0, :2] += 5
    decision = controller.decide(
        matrix, baseline={"wiki": 2.0}, candidate={"wiki": 2.2}
    )
    assert not decision.passed
    assert torch.equal(matrix.master_weight, weight)
    assert controller.wal.verify()[-1].kind == "rollback"
