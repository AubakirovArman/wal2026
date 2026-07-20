import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from mixed_q4_scale_recovery import FixedCodeMixedScaleLinear  # noqa: E402


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
