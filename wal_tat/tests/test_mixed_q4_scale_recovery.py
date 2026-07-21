import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from mixed_q4_scale_recovery import (  # noqa: E402
    FixedCodeMixedScaleLinear,
    lowest_damage_progressive_masks,
    parse_progressive_levels,
)


def test_fixed_code_mixed_scale_layer_changes_only_q4_groups():
    entry = {
        "shape": (1, 4),
        "q2_codes_int8": torch.tensor([[[1, -1], [0, 0]]], dtype=torch.int8),
        "q2_scales_fp16": torch.tensor([[0.5, 1.0]], dtype=torch.float16),
        "q4_codes_int8": torch.tensor([[[0, 0], [2, -2]]], dtype=torch.int8),
        "q4_scales_fp16": torch.tensor([[1.0, 0.25]], dtype=torch.float16),
        "q4_mask": torch.tensor([[False, True]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry, max_abs_log_scale_delta=0.5, bias=None
    )
    baseline = layer.effective_weight().detach().clone()
    with torch.no_grad():
        layer.log_scale_delta.fill_(0.2)
        layer.constrain_()
    changed = layer.effective_weight().detach()
    assert torch.equal(changed[:, :2], baseline[:, :2])
    assert not torch.equal(changed[:, 2:], baseline[:, 2:])
    assert layer.log_scale_delta[0, 0].item() == 0.0


def test_q4_proxy_is_hard_forward_and_reports_boundary_mobility():
    entry = {
        "shape": (1, 4),
        "q2_codes_int8": torch.tensor([[[1, -1], [0, 0]]], dtype=torch.int8),
        "q2_scales_fp16": torch.tensor([[0.5, 1.0]], dtype=torch.float16),
        "q4_codes_int8": torch.tensor([[[0, 0], [2, -2]]], dtype=torch.int8),
        "q4_scales_fp16": torch.tensor([[1.0, 0.25]], dtype=torch.float16),
        "q4_mask": torch.tensor([[False, True]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry, max_abs_log_scale_delta=0.5, bias=None, train_codes=True
    )
    with torch.no_grad():
        layer.proxy_code[0, 1] = torch.tensor([2.49, -1.51])
    assert torch.allclose(
        layer.effective_weight().detach()[:, 2:], torch.tensor([[0.5, -0.5]])
    )
    statistics = layer.proxy_statistics()
    assert statistics["mean_abs_displacement"] == pytest.approx(0.49)
    assert statistics["near_boundary_fraction"] == 1.0
    assert layer.code_churn() == 0.0

    with torch.no_grad():
        layer.proxy_code[0, 1, 0] = 2.51
    assert layer.code_churn() == 0.5


def test_layer_can_reset_to_three_level_hard_codebook():
    entry = {
        "shape": (1, 4),
        "q2_codes_int8": torch.tensor([[[1, -1], [0, 0]]], dtype=torch.int8),
        "q2_scales_fp16": torch.tensor([[0.5, 1.0]], dtype=torch.float16),
        "q4_codes_int8": torch.tensor([[[0, 0], [2, -2]]], dtype=torch.int8),
        "q4_scales_fp16": torch.tensor([[1.0, 0.25]], dtype=torch.float16),
        "q4_mask": torch.tensor([[False, True]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry, max_abs_log_scale_delta=0.5, bias=None, train_codes=True
    )
    codes = torch.tensor([[[0, 0], [1, -1]]], dtype=torch.int8)
    scales = torch.tensor([[1.0, 0.75]])
    layer.reset_codebook_(codes, scales, levels=3)
    assert layer.code_lower == -1
    assert layer.code_upper == 1
    assert set(layer.deploy_codes().unique().tolist()) <= {-1, 0, 1}
    assert torch.allclose(
        layer.effective_weight().detach(), torch.tensor([[0.5, -0.5, 0.75, -0.75]])
    )


def test_progressive_level_schedule_must_strictly_decrease():
    assert parse_progressive_levels("7,5,3") == (7, 5, 3)
    assert parse_progressive_levels("") == ()
    with pytest.raises(ValueError, match="strictly decreasing"):
        parse_progressive_levels("5,7")
    with pytest.raises(ValueError, match="odd integers"):
        parse_progressive_levels("7,4,3")


def test_q8_groups_remain_fixed_while_q4_proxy_trains():
    entry = {
        "shape": (1, 6),
        "q2_codes_int8": torch.tensor(
            [[[1, -1], [0, 0], [0, 0]]], dtype=torch.int8
        ),
        "q2_scales_fp16": torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.float16),
        "q4_codes_int8": torch.tensor(
            [[[0, 0], [2, -2], [0, 0]]], dtype=torch.int8
        ),
        "q4_scales_fp16": torch.tensor([[1.0, 0.25, 1.0]], dtype=torch.float16),
        "q4_mask": torch.tensor([[False, True, False]]),
        "q8_codes_int8": torch.tensor(
            [[[0, 0], [0, 0], [100, -100]]], dtype=torch.int8
        ),
        "q8_scales_fp16": torch.tensor([[1.0, 1.0, 0.01]], dtype=torch.float16),
        "q8_mask": torch.tensor([[False, False, True]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry, max_abs_log_scale_delta=0.5, bias=None, train_codes=True
    )
    baseline = layer.effective_weight().detach().clone()
    with torch.no_grad():
        layer.proxy_code[0, 1] = torch.tensor([3.0, -3.0])
        layer.log_scale_delta[0, 1] = 0.2
        layer.constrain_()
    changed = layer.effective_weight().detach()
    assert not torch.equal(changed[:, 2:4], baseline[:, 2:4])
    assert torch.equal(changed[:, 4:], baseline[:, 4:])


def test_optional_q8_scale_recovery_changes_only_q8_groups():
    entry = {
        "shape": (1, 6),
        "q2_codes_int8": torch.tensor(
            [[[1, -1], [0, 0], [0, 0]]], dtype=torch.int8
        ),
        "q2_scales_fp16": torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.float16),
        "q4_codes_int8": torch.tensor(
            [[[0, 0], [2, -2], [0, 0]]], dtype=torch.int8
        ),
        "q4_scales_fp16": torch.tensor([[1.0, 0.25, 1.0]], dtype=torch.float16),
        "q4_mask": torch.tensor([[False, True, False]]),
        "q8_codes_int8": torch.tensor(
            [[[0, 0], [0, 0], [100, -100]]], dtype=torch.int8
        ),
        "q8_scales_fp16": torch.tensor([[1.0, 1.0, 0.01]], dtype=torch.float16),
        "q8_mask": torch.tensor([[False, False, True]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry,
        max_abs_log_scale_delta=0.5,
        bias=None,
        train_q8_scales=True,
    )
    baseline = layer.effective_weight().detach().clone()
    with torch.no_grad():
        layer.q8_log_scale_delta.fill_(0.2)
        layer.constrain_()
    changed = layer.effective_weight().detach()

    assert torch.equal(changed[:, :4], baseline[:, :4])
    assert not torch.equal(changed[:, 4:], baseline[:, 4:])
    assert layer.q8_log_scale_delta[0, 0].item() == 0.0
    assert layer.q8_log_scale_delta[0, 1].item() == 0.0
    assert layer.deploy_q8_scales()[0, 2] != entry["q8_scales_fp16"][0, 2]


def test_empty_q4_mask_has_finite_zero_diagnostics():
    entry = {
        "shape": (1, 2),
        "q2_codes_int8": torch.tensor([[[1, -1]]], dtype=torch.int8),
        "q2_scales_fp16": torch.tensor([[0.5]], dtype=torch.float16),
        "q4_codes_int8": torch.zeros((1, 1, 2), dtype=torch.int8),
        "q4_scales_fp16": torch.ones((1, 1), dtype=torch.float16),
        "q4_mask": torch.tensor([[False]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry, max_abs_log_scale_delta=0.5, bias=None, train_codes=True
    )
    assert layer.code_churn() == 0.0
    assert layer.code_anchor_loss().item() == 0.0
    assert layer.proxy_statistics() == {
        "mean_abs_displacement": 0.0,
        "max_abs_displacement": 0.0,
        "near_boundary_fraction": 0.0,
    }


def test_masked_progressive_collapse_preserves_q4_fallback_q2_and_q8():
    entry = {
        "shape": (1, 8),
        "q2_codes_int8": torch.tensor(
            [[[1, -1], [0, 0], [0, 0], [0, 0]]], dtype=torch.int8
        ),
        "q2_scales_fp16": torch.tensor(
            [[0.5, 1.0, 1.0, 1.0]], dtype=torch.float16
        ),
        "q4_codes_int8": torch.tensor(
            [[[0, 0], [3, -3], [7, -8], [0, 0]]], dtype=torch.int8
        ),
        "q4_scales_fp16": torch.tensor(
            [[1.0, 0.25, 0.125, 1.0]], dtype=torch.float16
        ),
        "q4_mask": torch.tensor([[False, True, True, False]]),
        "q8_codes_int8": torch.tensor(
            [[[0, 0], [0, 0], [0, 0], [100, -100]]], dtype=torch.int8
        ),
        "q8_scales_fp16": torch.tensor(
            [[1.0, 1.0, 1.0, 0.01]], dtype=torch.float16
        ),
        "q8_mask": torch.tensor([[False, False, False, True]]),
    }
    layer = FixedCodeMixedScaleLinear(
        entry,
        max_abs_log_scale_delta=0.5,
        bias=None,
        train_codes=True,
        progressive_mask=torch.tensor([[False, True, False, False]]),
    )
    baseline = layer.effective_weight().detach().clone()
    codes = torch.tensor(
        [[[0, 0], [1, -1], [0, 0], [0, 0]]], dtype=torch.int8
    )
    scales = torch.tensor([[1.0, 0.5, 1.0, 1.0]])
    layer.reset_codebook_(codes, scales, levels=3)
    collapsed = layer.effective_weight().detach()

    assert torch.equal(collapsed[:, :2], baseline[:, :2])
    assert not torch.equal(collapsed[:, 2:4], baseline[:, 2:4])
    assert torch.equal(collapsed[:, 4:6], baseline[:, 4:6])
    assert torch.equal(collapsed[:, 6:], baseline[:, 6:])
    deployed = layer.deploy_codes()
    assert set(deployed[0, 1].tolist()) <= {-1, 0, 1}
    assert deployed[0, 2].tolist() == [7, -8]

    with torch.no_grad():
        layer.proxy_code[0, 2] = torch.tensor([0.0, 0.0])
        layer.log_scale_delta[0, 2] = 0.4
        layer.constrain_()
    assert layer.deploy_codes()[0, 2].tolist() == [7, -8]
    assert layer.log_scale_delta[0, 2].item() == 0.0


def test_progressive_mask_must_be_q4_subset():
    entry = {
        "shape": (1, 2),
        "q2_codes_int8": torch.tensor([[[1, -1]]], dtype=torch.int8),
        "q2_scales_fp16": torch.tensor([[0.5]], dtype=torch.float16),
        "q4_codes_int8": torch.zeros((1, 1, 2), dtype=torch.int8),
        "q4_scales_fp16": torch.ones((1, 1), dtype=torch.float16),
        "q4_mask": torch.tensor([[False]]),
    }
    with pytest.raises(ValueError, match="subset"):
        FixedCodeMixedScaleLinear(
            entry,
            max_abs_log_scale_delta=0.5,
            bias=None,
            progressive_mask=torch.tensor([[True]]),
        )


def test_lowest_damage_progressive_masks_select_globally():
    damage = {
        "a": torch.tensor([[4.0, 1.0, 9.0]]),
        "b": torch.tensor([[0.5, 2.0]]),
    }
    eligible = {
        "a": torch.tensor([[True, True, False]]),
        "b": torch.tensor([[True, True]]),
    }
    masks, selected, total = lowest_damage_progressive_masks(
        damage, eligible, 0.5
    )
    assert (selected, total) == (2, 4)
    assert masks["a"].tolist() == [[False, True, False]]
    assert masks["b"].tolist() == [[True, False]]
