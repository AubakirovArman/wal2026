import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from build_matched_validation_recovery_suite import interleave, windows  # noqa: E402


def test_windows_are_contiguous_and_include_next_token():
    ids = torch.arange(30)
    result = windows(ids, count=2, length=4, offset=3)
    assert [value.tolist() for value in result] == [
        [3, 4, 5, 6, 7],
        [7, 8, 9, 10, 11],
    ]


def test_interleave_preserves_weighted_domain_order():
    values = {
        "c4": [torch.tensor([0]), torch.tensor([1])],
        "squad": [
            torch.tensor([10]),
            torch.tensor([11]),
            torch.tensor([12]),
            torch.tensor([13]),
        ],
    }
    result = interleave(values, {"c4": 1, "squad": 2}, base_count=2)
    assert [int(value.item()) for value in result] == [0, 10, 12, 1, 11, 13]
