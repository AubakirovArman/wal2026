import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from verify_mixed_q2_q4_artifact import valid_weight_count  # noqa: E402


def test_valid_weight_count_handles_partial_last_group():
    mask = torch.tensor([[True, False], [True, True]])
    assert valid_weight_count(mask, columns=6, group_size=4) == 10
