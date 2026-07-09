from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from gnn_siamese.losses import DeltaLoss, DeltaLossOutput
from gnn_siamese.models.relational import MLPDelta


def test_delta_none_returns_zero_loss_and_is_inactive() -> None:
    criterion = DeltaLoss(mode="none")
    z_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    output = criterion(z_delta)
    output.loss.backward()

    assert isinstance(output, DeltaLossOutput)
    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.is_active is False
    assert output.mode == "none"
    assert output.batch_size == 2
    assert output.embedding_dim == 2
    assert z_delta.grad is not None
    assert z_delta.grad.abs().sum().item() == pytest.approx(0.0, abs=1.0e-8)


def test_delta_consistency_requires_second_view() -> None:
    criterion = DeltaLoss(mode="consistency")
    z_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="requires z_delta_2"):
        criterion(z_delta)


def test_delta_consistency_is_finite_and_smaller_for_identical_views() -> None:
    criterion = DeltaLoss(mode="consistency")
    z_delta_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    z_delta_2 = torch.tensor([[0.5, 0.0], [0.0, 0.5]], requires_grad=True)
    identical = z_delta_1.detach().clone().requires_grad_(True)

    output_different = criterion(z_delta_1, z_delta_2=z_delta_2)
    output_identical = criterion(z_delta_1.detach(), z_delta_2=identical)
    output_different.loss.backward()

    assert torch.isfinite(output_different.loss)
    assert output_identical.loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert output_identical.loss.item() < output_different.loss.item()
    assert z_delta_1.grad is not None
    assert z_delta_2.grad is not None
    assert z_delta_1.grad.abs().sum().item() > 0.0
    assert z_delta_2.grad.abs().sum().item() > 0.0


def test_delta_variance_penalizes_collapse_and_returns_metric() -> None:
    criterion = DeltaLoss(mode="variance", gamma=1.0)
    collapsed = torch.ones((4, 3), requires_grad=True)
    varied = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.5],
            [0.5, -1.0, 0.0],
        ],
        requires_grad=True,
    )

    collapsed_output = criterion(collapsed)
    varied_output = criterion(varied)

    assert collapsed_output.loss.item() > 0.0
    assert varied_output.loss.item() < collapsed_output.loss.item()
    assert collapsed_output.variance_metric is not None
    assert collapsed_output.variance_metric.item() < criterion.gamma


def test_delta_variance_requires_batch_at_least_two() -> None:
    criterion = DeltaLoss(mode="variance")
    z_delta = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="batch size >= 2"):
        criterion(z_delta)


def test_delta_covariance_penalizes_correlated_dimensions_and_returns_metric() -> None:
    criterion = DeltaLoss(mode="covariance")
    correlated = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
        ],
        requires_grad=True,
    )
    weakly_correlated = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.5],
            [0.5, -1.0],
        ],
        requires_grad=True,
    )

    correlated_output = criterion(correlated)
    weak_output = criterion(weakly_correlated)

    assert torch.isfinite(correlated_output.loss)
    assert correlated_output.loss.item() > weak_output.loss.item()
    assert correlated_output.covariance_metric is not None
    assert correlated_output.covariance_metric.item() > 0.0


def test_delta_covariance_requires_batch_at_least_two() -> None:
    criterion = DeltaLoss(mode="covariance")
    z_delta = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="batch size >= 2"):
        criterion(z_delta)


def test_delta_descriptor_requires_target() -> None:
    criterion = DeltaLoss(mode="descriptor")
    z_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="requires target_name"):
        criterion(z_delta)


def test_delta_descriptor_requires_target_tensor_after_target_name() -> None:
    criterion = DeltaLoss(mode="descriptor")
    z_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="requires an explicit target"):
        criterion(z_delta, target_name="delta_descriptor")


def test_delta_descriptor_blocks_energy_target_by_default() -> None:
    criterion = DeltaLoss(mode="descriptor")
    z_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.5, 0.0], [0.0, 0.5]])

    with pytest.raises(ValueError, match="allow_energy_target=True"):
        criterion(z_delta, target=target, target_name="custom_structure_energy")


def test_delta_descriptor_allows_energy_target_when_enabled() -> None:
    criterion = DeltaLoss(mode="descriptor", allow_energy_target=True)
    z_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.5, 0.0], [0.0, 0.5]])

    output = criterion(z_delta, target=target, target_name="custom_structure_energy")

    assert torch.isfinite(output.loss)
    assert output.target_name == "custom_structure_energy"
    assert output.prediction is not None


def test_delta_descriptor_supports_allowed_target_and_gradients_flow() -> None:
    predictor = nn.Linear(3, 3, bias=False)
    criterion = DeltaLoss(mode="descriptor", predictor=predictor)
    z_delta = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, -0.5]])

    output = criterion(z_delta, target=target, target_name="mechanism_direction")
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.target_name == "mechanism_direction"
    assert output.prediction is not None
    assert output.prediction.shape == target.shape
    assert z_delta.grad is not None
    assert z_delta.grad.abs().sum().item() > 0.0
    assert predictor.weight.grad is not None
    assert predictor.weight.grad.abs().sum().item() > 0.0


def test_delta_loss_public_api_exports_symbols() -> None:
    from gnn_siamese.losses import __all__ as losses_public_api

    assert "DeltaLoss" in losses_public_api
    assert "DeltaLossOutput" in losses_public_api


def test_delta_loss_trains_mlp_delta_route_with_gradient_and_weight_change() -> None:
    mlp_delta = MLPDelta(
        input_dim=10,
        hidden_dim=6,
        output_dim=3,
        num_layers=2,
        dropout=0.0,
    )
    criterion = DeltaLoss(mode="variance", gamma=1.0)
    optimizer = torch.optim.SGD(mlp_delta.parameters(), lr=0.1)
    r_delta = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0, 0.2, -0.1, 0.3, 0.4, -0.2, 0.0],
            [0.0, 1.0, -0.5, 0.2, -0.3, 0.1, -0.4, 0.0, 0.5, -0.1],
            [1.0, 1.0, 0.0, -0.2, 0.4, -0.3, 0.2, -0.5, 0.1, 0.3],
        ],
        dtype=torch.float32,
    )

    before = mlp_delta.network[0].weight.detach().clone()
    z_delta = mlp_delta(r_delta)
    output = criterion(z_delta)
    output.loss.backward()

    grads = [parameter.grad for parameter in mlp_delta.parameters()]
    assert any(grad is not None and grad.abs().sum().item() > 0.0 for grad in grads)

    optimizer.step()
    after = mlp_delta.network[0].weight.detach()

    assert output.is_active is True
    assert torch.isfinite(output.loss)
    assert not torch.allclose(before, after)
