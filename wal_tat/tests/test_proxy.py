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
