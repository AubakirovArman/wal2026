from pathlib import Path
import sys

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from build_audit_holdout import c4_arrow_pattern, text_windows  # noqa: E402


def test_text_windows_uses_declared_offset_and_extra_label_token() -> None:
    ids = torch.arange(20)
    windows = text_windows(ids, count=2, length=4, offset=3)
    assert [value.tolist() for value in windows] == [
        [3, 4, 5, 6, 7],
        [7, 8, 9, 10, 11],
    ]


def test_text_windows_rejects_short_stream() -> None:
    with pytest.raises(RuntimeError):
        text_windows(torch.arange(5), count=2, length=4, offset=0)


def test_c4_train_shard_pattern_is_explicit_and_zero_padded() -> None:
    assert "c4-train-00001-of-" in c4_arrow_pattern("train", 1)
    assert c4_arrow_pattern("validation", 999).endswith("c4-validation.arrow")
    with pytest.raises(ValueError, match="non-negative"):
        c4_arrow_pattern("train", -1)
