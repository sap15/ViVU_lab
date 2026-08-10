from __future__ import annotations

import pytest
import torch

from gnn_siamese.models import ModelAProjectionHead


def _head() -> ModelAProjectionHead:
    return ModelAProjectionHead(
        z_delta_pair_dim=7, hidden_dim=9, projection_dim=3
    )


def test_public_api_shape_cpu_and_distinct_dimensions() -> None:
    head = _head().cpu()
    output = head(torch.randn(4, 7))
    assert output.shape == (4, 3)
    assert output.device.type == "cpu"
    assert sum(parameter.numel() for parameter in head.parameters()) > 0


@pytest.mark.parametrize("shape", [(2, 6), (2, 7, 1)])
def test_dimension_validation_is_informative(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="z_delta_pair"):
        _head()(torch.randn(*shape))


def test_dtype_device_and_eval_determinism() -> None:
    head = _head().double().eval()
    value = torch.randn(3, 7, dtype=torch.float64)
    first = head(value)
    second = head(value.clone())
    assert first.dtype == value.dtype
    assert first.device == value.device
    assert torch.equal(first, second)


def test_backward_and_optimizer_step_change_projection_weights() -> None:
    torch.manual_seed(10)
    head = _head()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in head.parameters()]
    optimizer.zero_grad()
    head(torch.randn(5, 7)).square().mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    optimizer.step()
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, head.parameters(), strict=True)
    )


def test_one_instance_is_reused_for_both_views() -> None:
    head = _head()
    calls: list[int] = []
    hook = head.register_forward_hook(lambda module, _args, _out: calls.append(id(module)))
    head(torch.randn(2, 7))
    head(torch.randn(2, 7))
    hook.remove()
    assert calls == [id(head), id(head)]
