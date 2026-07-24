from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.losses import RelativeWTLoss, RelativeWTLossOutput


def test_relative_wt_none_returns_zero_loss_and_is_inactive() -> None:
    criterion = RelativeWTLoss(mode="none")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    h_wt = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)

    output = criterion(h_mut, h_wt)
    output.loss.backward()

    assert isinstance(output, RelativeWTLossOutput)
    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.is_active is False
    assert output.mode == "none"
    assert output.mean_distance.item() == pytest.approx(1.0, abs=1.0e-6)
    assert h_mut.grad is not None
    assert h_wt.grad is not None
    assert h_mut.grad.abs().sum().item() == pytest.approx(0.0, abs=1.0e-8)
    assert h_wt.grad.abs().sum().item() == pytest.approx(0.0, abs=1.0e-8)


def test_relative_wt_none_does_not_alter_baseline_scalar() -> None:
    baseline = torch.tensor(1.75)
    criterion = RelativeWTLoss(mode="none")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    h_wt = torch.zeros_like(h_mut)

    output = criterion(h_mut, h_wt)

    assert (baseline + output.loss).item() == pytest.approx(baseline.item(), abs=1.0e-8)


def test_relative_wt_margin_is_positive_when_margin_is_violated() -> None:
    criterion = RelativeWTLoss(mode="margin", margin=0.5)
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    h_wt = torch.zeros_like(h_mut, requires_grad=True)

    output = criterion(h_mut, h_wt)

    expected = 0.5 * (((1.0 - 0.5) ** 2) + ((2.0 - 0.5) ** 2)) / 2.0
    assert output.loss.item() == pytest.approx(expected, abs=1.0e-6)
    assert output.mean_distance.item() == pytest.approx(1.5, abs=1.0e-6)
    assert output.is_active is True
    assert output.margin == pytest.approx(0.5)


def test_relative_wt_margin_is_zero_when_min_direction_is_not_violated() -> None:
    criterion = RelativeWTLoss(mode="margin", margin=1.0, direction="min")
    h_mut = torch.tensor([[0.2, 0.0], [0.0, 0.8]])
    h_wt = torch.zeros_like(h_mut)

    output = criterion(h_mut, h_wt)

    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-8)


def test_relative_wt_margin_stop_gradient_wt_blocks_wt_gradients() -> None:
    criterion = RelativeWTLoss(mode="margin", margin=0.5, stop_gradient_wt=True)
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    h_wt = torch.zeros_like(h_mut, requires_grad=True)

    output = criterion(h_mut, h_wt)
    output.loss.backward()

    assert h_mut.grad is not None
    assert h_mut.grad.abs().sum().item() > 0.0
    assert h_wt.grad is None or h_wt.grad.abs().sum().item() == pytest.approx(0.0, abs=1.0e-8)


def test_relative_wt_margin_does_not_treat_mutant_wt_as_strong_positive() -> None:
    criterion = RelativeWTLoss(mode="margin", margin=0.0, direction="max")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    h_wt = torch.zeros_like(h_mut)

    output = criterion(h_mut, h_wt)

    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.mean_distance.item() > 0.0


def test_relative_wt_ranking_requires_target() -> None:
    criterion = RelativeWTLoss(mode="ranking")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    h_wt = torch.zeros_like(h_mut)

    with pytest.raises(ValueError, match="requires an explicit target"):
        criterion(h_mut, h_wt)


def test_relative_wt_ranking_requires_explicit_target_name() -> None:
    criterion = RelativeWTLoss(mode="ranking", margin=0.1)
    h_mut = torch.tensor([[0.2, 0.0], [2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    h_wt = torch.zeros_like(h_mut)
    ranking_target = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="requires an explicit target_name"):
        criterion(h_mut, h_wt, ranking_target=ranking_target)


def test_relative_wt_ranking_returns_finite_loss_for_ordered_targets() -> None:
    criterion = RelativeWTLoss(mode="ranking", margin=0.1)
    h_mut = torch.tensor([[0.2, 0.0], [2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    h_wt = torch.zeros_like(h_mut)
    ranking_target = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    output = criterion(h_mut, h_wt, ranking_target=ranking_target, target_name="external_rank")
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.is_active is True
    assert output.num_pairs == 3
    assert output.target_name == "external_rank"
    assert h_mut.grad is not None
    assert h_mut.grad.abs().sum().item() > 0.0


def test_relative_wt_ranking_ignores_equal_targets() -> None:
    criterion = RelativeWTLoss(mode="ranking")
    h_mut = torch.tensor([[0.5, 0.0], [0.0, 0.5], [2.0, 0.0]])
    h_wt = torch.zeros_like(h_mut)
    ranking_target = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)

    output = criterion(h_mut, h_wt, ranking_target=ranking_target, target_name="severity_rank")

    assert output.num_pairs == 2
    assert output.target_name == "severity_rank"
    assert torch.isfinite(output.loss)


def test_relative_wt_predictive_requires_target() -> None:
    criterion = RelativeWTLoss(mode="predictive")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    h_wt = torch.zeros_like(h_mut)

    with pytest.raises(ValueError, match="requires an explicit target"):
        criterion(h_mut, h_wt, target_name="delta_descriptor")


def test_relative_wt_predictive_requires_explicit_target_name() -> None:
    criterion = RelativeWTLoss(mode="predictive")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    h_wt = torch.zeros_like(h_mut)
    auxiliary_target = torch.tensor([1.5, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="requires an explicit target_name"):
        criterion(h_mut, h_wt, auxiliary_target=auxiliary_target)


def test_relative_wt_predictive_returns_prediction_and_finite_loss() -> None:
    criterion = RelativeWTLoss(mode="predictive")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    h_wt = torch.zeros_like(h_mut)
    auxiliary_target = torch.tensor([1.5, 1.0], dtype=torch.float32)

    output = criterion(
        h_mut,
        h_wt,
        auxiliary_target=auxiliary_target,
        target_name="delta_descriptor",
    )
    output.loss.backward()

    assert output.prediction is not None
    assert output.prediction.shape == auxiliary_target.shape
    assert torch.isfinite(output.loss)
    assert output.target_name == "delta_descriptor"
    assert h_mut.grad is not None
    assert h_mut.grad.abs().sum().item() > 0.0


def test_relative_wt_predictive_rejects_energy_target_by_default() -> None:
    criterion = RelativeWTLoss(mode="predictive")
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    h_wt = torch.zeros_like(h_mut)
    auxiliary_target = torch.tensor([0.5, 0.5], dtype=torch.float32)

    with pytest.raises(ValueError, match="allow_energy_target=True"):
        criterion(
            h_mut,
            h_wt,
            auxiliary_target=auxiliary_target,
            target_name="custom_structure_energy",
        )


def test_relative_wt_predictive_allows_energy_target_when_enabled() -> None:
    criterion = RelativeWTLoss(mode="predictive", allow_energy_target=True)
    h_mut = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    h_wt = torch.zeros_like(h_mut)
    auxiliary_target = torch.tensor([0.5, 0.5], dtype=torch.float32)

    output = criterion(
        h_mut,
        h_wt,
        auxiliary_target=auxiliary_target,
        target_name="custom_structure_energy",
    )

    assert output.target_name == "custom_structure_energy"
    assert torch.isfinite(output.loss)


def test_relative_wt_public_api_exports_relative_wt_symbols() -> None:
    from gnn_siamese.losses import __all__ as losses_public_api

    assert "RelativeWTLoss" in losses_public_api
    assert "RelativeWTLossOutput" in losses_public_api
