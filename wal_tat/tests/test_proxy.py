import torch

from wal_tat import ProxyTernaryMatrix, soft_ternary_proxy


def test_soft_proxy_is_ternary_like_at_low_temperature():
    values = soft_ternary_proxy(torch.tensor([-1.0, 0.0, 1.0]), 0.05)
    assert torch.allclose(values, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-4)


def test_proxy_matrix_has_exact_hard_forward_and_soft_gradient():
    codes = torch.tensor([[[-1, 0, 1, 1]]], dtype=torch.int8)
    scales = torch.tensor([[2.0]])
    matrix = ProxyTernaryMatrix(codes, scales, compute_dtype=torch.float32)
    weight = matrix.effective_weight()
    assert torch.equal(weight.detach(), torch.tensor([[-2.0, 0.0, 2.0, 2.0]]))
    weight.sum().backward()
    assert matrix.proxy_code.grad is not None
    assert torch.count_nonzero(matrix.proxy_code.grad) > 0
    assert set(matrix.hard_codes().flatten().tolist()) <= {-1, 0, 1}


def test_proxy_churn_and_constraint():
    matrix = ProxyTernaryMatrix(
        torch.zeros((1, 1, 4), dtype=torch.int8),
        torch.ones((1, 1)),
        compute_dtype=torch.float32,
    )
    with torch.no_grad():
        matrix.proxy_code[0, 0, 0] = 0.6
        matrix.proxy_code[0, 0, 1] = 7.0
        matrix.group_scale.fill_(-3.0)
    assert matrix.code_churn() == 0.5
    matrix.constrain_()
    assert float(matrix.proxy_code.detach().max()) == 1.5
    assert torch.isclose(
        matrix.group_scale.detach().min(), torch.tensor(1e-5), rtol=1e-6
    )


def test_partial_proxy_preserves_bf16_groups_and_masks_gradients():
    codes = torch.tensor(
        [[[-1, 0, 1, 1], [1, 1, 1, 1]]], dtype=torch.int8
    )
    scales = torch.tensor([[2.0, 7.0]])
    committed = torch.tensor([[True, False]])
    master = torch.tensor([[-9.0, -8.0, -7.0, -6.0, 3.0, 4.0, 5.0]])
    matrix = ProxyTernaryMatrix(
        codes,
        scales,
        compute_dtype=torch.float32,
        committed_mask=committed,
        master_weight=master,
    )
    weight = matrix.effective_weight()
    assert torch.equal(
        weight.detach(), torch.tensor([[-2.0, 0.0, 2.0, 2.0, 3.0, 4.0, 5.0]])
    )
    weight.sum().backward()
    assert torch.count_nonzero(matrix.proxy_code.grad[0, 0]) > 0
    assert torch.count_nonzero(matrix.proxy_code.grad[0, 1]) == 0
    assert matrix.group_scale.grad[0, 0] != 0
    assert matrix.group_scale.grad[0, 1] == 0


def test_partial_proxy_churn_ignores_uncommitted_codes():
    matrix = ProxyTernaryMatrix(
        torch.zeros((1, 2, 4), dtype=torch.int8),
        torch.ones((1, 2)),
        compute_dtype=torch.float32,
        committed_mask=torch.tensor([[True, False]]),
        master_weight=torch.ones((1, 8)),
    )
    with torch.no_grad():
        matrix.proxy_code[0, 0, 0] = 0.6
        matrix.proxy_code[0, 1] = 1.0
    assert matrix.code_churn() == 0.25
    assert matrix.proxy_anchor_loss() > 0


def test_deployment_statistics_report_boundary_and_ignore_uncommitted_groups():
    matrix = ProxyTernaryMatrix(
        torch.tensor([[[-1, 0, 1, 0], [1, 1, 1, 1]]], dtype=torch.int8),
        torch.tensor([[2.0, 9.0]]),
        compute_dtype=torch.float32,
        committed_mask=torch.tensor([[True, False]]),
        master_weight=torch.ones((1, 8)),
    )
    with torch.no_grad():
        matrix.proxy_code[0, 0, 1] = 0.51
        matrix.proxy_code[0, 1] = -0.5
    statistics = matrix.deployment_statistics(boundary_epsilon=0.02)
    assert statistics["code_counts"] == {"-1": 1, "0": 1, "1": 2}
    assert statistics["zero_fraction"] == 0.25
    assert statistics["code_churn"] == 0.25
    assert statistics["near_boundary_fraction"] == 0.25
    assert statistics["boundary_distance_min"] < 0.011
    assert statistics["scale_min"] == 2.0
    assert statistics["scale_max"] == 2.0


def test_deployment_statistics_reject_negative_boundary_epsilon():
    matrix = ProxyTernaryMatrix(
        torch.zeros((1, 1, 4), dtype=torch.int8),
        torch.ones((1, 1)),
        compute_dtype=torch.float32,
    )
    try:
        matrix.deployment_statistics(boundary_epsilon=-0.1)
    except ValueError as error:
        assert "boundary_epsilon" in str(error)
    else:
        raise AssertionError("negative boundary epsilon must be rejected")
