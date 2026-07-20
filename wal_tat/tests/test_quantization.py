import math

import pytest
import torch

from wal_tat import (
    activation_fisher_group_damage,
    diagonal_ternary_search,
    hard_codes_scales,
    q2_g128_physical_bpw,
    select_group_mask,
    sensitivity_decile_mask,
    transaction_schedule,
)


def test_hard_quantizer_is_strictly_ternary():
    weight = torch.tensor([[0.1, -0.8, 1.7, -2.2]])
    codes, scales = hard_codes_scales(weight, group_size=2)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scales.shape == (1, 2)


def test_schedule_finishes_hard():
    assert transaction_schedule(0, 100) == pytest.approx((0.0, 0.3508855606815209))
    assert transaction_schedule(100, 100) == (1.0, 0.0)


def test_prism_compatible_layout_is_2_125_physical_bpw():
    assert q2_g128_physical_bpw() == 2.125
    assert math.log2(3) == pytest.approx(1.5849625)


def test_fisher_score_uses_causal_moments():
    weight = torch.tensor([[0.1, 0.3, 0.1, 0.3]])
    output = torch.ones(1)
    uniform = activation_fisher_group_damage(
        weight, torch.ones(4), output, group_size=2, relative=False
    )
    shifted = activation_fisher_group_damage(
        weight, torch.tensor([100.0, 100.0, 1.0, 1.0]), output, group_size=2, relative=False
    )
    assert uniform[0, 0] == pytest.approx(uniform[0, 1])
    assert shifted[0, 0] > shifted[0, 1]


def test_select_mask_respects_eligibility():
    scores = torch.tensor([[4.0, 1.0], [3.0, 2.0]])
    eligible = torch.tensor([[True, False], [True, True]])
    mask = select_group_mask(scores, 2, eligible=eligible, strategy="lowest")
    assert torch.equal(mask, torch.tensor([[False, False], [True, True]]))


def test_diagonal_search_never_worse_than_its_tested_zero_threshold():
    weight = torch.tensor([[0.1, -0.4, 1.2, -2.0]])
    moment = torch.tensor([1.0, 5.0, 2.0, 0.5])
    codes, scales, error = diagonal_ternary_search(weight, moment, group_size=4)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    baseline_scale = (moment * weight.abs()).sum() / moment.sum()
    baseline_error = (moment * (weight - weight.sign() * baseline_scale).square()).sum()
    assert error.item() <= baseline_error.item() + 1e-6


def test_sensitivity_decile_mask_covers_requested_rank_bucket():
    scores = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    mask = sensitivity_decile_mask(scores, eligible, decile=9, count=5)
    selected = scores[mask]
    assert mask.sum().item() == 5
    assert selected.min().item() >= 80
    assert selected.max().item() <= 89
    assert selected.tolist() == [80.0, 82.0, 84.0, 87.0, 89.0]


def test_sensitivity_decile_mask_respects_sparse_eligibility():
    scores = torch.arange(120, dtype=torch.float32).reshape(12, 10)
    eligible = scores.remainder(2).eq(0)
    mask = sensitivity_decile_mask(scores, eligible, decile=10, count=3)
    assert torch.all(eligible[mask])
    assert scores[mask].tolist() == [108.0, 112.0, 118.0]
