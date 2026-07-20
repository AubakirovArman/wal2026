import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from wal_tat import valid_group_weight_count


def test_valid_weight_count_handles_partial_last_group():
    mask = torch.tensor([[True, False], [True, True]])
    assert valid_group_weight_count(mask, columns=6, group_size=4) == 10
