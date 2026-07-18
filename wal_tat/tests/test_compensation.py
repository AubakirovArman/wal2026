import pytest
import torch

from wal_tat import linked_down_group_mask, structured_channel_candidate_mask


def test_structured_selection_bounds_linked_blocks():
    scores = torch.arange(32, dtype=torch.float32).view(8, 4)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    mask, blocks = structured_channel_candidate_mask(
        scores,
        eligible,
        count=6,
        channel_block_size=2,
        block_count=2,
    )
    assert blocks == (0, 1)
    assert mask.sum() == 6
    selected_rows = torch.where(mask.any(dim=1))[0]
    assert int(selected_rows.max()) < 4


def test_structured_selection_respects_eligibility():
    scores = torch.arange(16, dtype=torch.float32).view(4, 4)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    eligible[0] = False
    mask, _ = structured_channel_candidate_mask(
        scores, eligible, count=3, channel_block_size=2, block_count=1
    )
    assert not mask[0].any()
    assert mask.sum() == 3


def test_linked_down_mask_selects_whole_input_groups():
    state = torch.ones((3, 5), dtype=torch.bool)
    mask = linked_down_group_mask(state, (1, 4))
    assert mask.sum() == 6
    assert mask[:, 1].all() and mask[:, 4].all()
    assert not mask[:, 0].any()


def test_linked_down_mask_rejects_bad_block():
    with pytest.raises(ValueError, match="outside"):
        linked_down_group_mask(torch.ones((2, 3), dtype=torch.bool), (3,))
