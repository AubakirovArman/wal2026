"""Tiny executable WAL-TAT transaction with a synthetic quality gate."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from wal_tat import (
    HashChainWAL,
    RatioGate,
    TransactionController,
    TransactionalTernaryMatrix,
    group_weight_mse,
    select_group_mask,
)


def main() -> None:
    torch.manual_seed(7)
    matrix = TransactionalTernaryMatrix(torch.randn(4, 8), group_size=4)
    scores = group_weight_mse(matrix.master_weight, group_size=4)
    mask = select_group_mask(scores, count=2, strategy="lowest")

    with TemporaryDirectory() as directory:
        wal = HashChainWAL(Path(directory) / "transactions.jsonl", fsync=False)
        controller = TransactionController(wal, "toy.linear", RatioGate(1.02))
        transaction_id = controller.begin(matrix, mask, selector="weight_mse_demo")
        controller.progress(matrix, step=1, pressure=1.0, temperature=0.0)
        decision = controller.decide(
            matrix,
            baseline={"general": 2.0, "code": 3.0},
            candidate={"general": 2.01, "code": 3.04},
        )
        print(
            {
                "transaction_id": transaction_id,
                "passed": decision.passed,
                "committed_groups": int(matrix.committed_mask.sum()),
                "wal_records": [record.kind for record in wal.verify()],
            }
        )


if __name__ == "__main__":
    main()
