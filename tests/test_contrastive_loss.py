from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.losses import NTXentLoss


def test_nt_xent_returns_finite_scalar_loss_and_metrics() -> None:
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    z2 = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 1.2]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = criterion(z1, z2)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.mean_positive_similarity)
    assert torch.isfinite(output.mean_negative_similarity)
    assert output.batch_size == 3
    assert output.temperature == pytest.approx(0.2)


def test_nt_xent_loss_is_low_when_positive_views_match() -> None:
    criterion = NTXentLoss(temperature=0.1)
    z1 = torch.eye(4, dtype=torch.float32)
    z2 = z1.clone()

    output = criterion(z1, z2)

    assert output.loss.item() < 0.05
    assert output.mean_positive_similarity.item() == pytest.approx(1.0, abs=1.0e-6)


def test_nt_xent_loss_changes_when_positives_are_permuted() -> None:
    criterion = NTXentLoss(temperature=0.1)
    z1 = torch.eye(4, dtype=torch.float32)
    z2 = z1.clone()
    permuted_z2 = z2[torch.tensor([1, 2, 3, 0])]

    aligned_loss = criterion(z1, z2).loss
    permuted_loss = criterion(z1, permuted_z2).loss

    assert permuted_loss.item() > aligned_loss.item() + 0.5


def test_nt_xent_backward_produces_gradients_in_both_views() -> None:
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.randn(5, 6, dtype=torch.float32, requires_grad=True)
    z2 = torch.randn(5, 6, dtype=torch.float32, requires_grad=True)

    output = criterion(z1, z2)
    output.loss.backward()

    assert z1.grad is not None
    assert z2.grad is not None
    assert torch.isfinite(z1.grad).all()
    assert torch.isfinite(z2.grad).all()
    assert z1.grad.abs().sum().item() > 0.0
    assert z2.grad.abs().sum().item() > 0.0


def test_nt_xent_rejects_invalid_batch_size() -> None:
    criterion = NTXentLoss()
    z1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    z2 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="batch size >= 2"):
        criterion(z1, z2)


def test_nt_xent_does_not_define_mutant_wt_as_default_positive() -> None:
    criterion = NTXentLoss(temperature=0.1)
    z1 = torch.eye(4, dtype=torch.float32)
    z2 = z1[torch.tensor([1, 0, 2, 3])]

    output = criterion(z1, z2)

    assert output.mean_positive_similarity.item() < 1.0
    assert output.loss.item() > 0.1


def test_nt_xent_validate_pre_normalized_embeddings_when_requested() -> None:
    criterion = NTXentLoss(normalize=False)
    z1 = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    z2 = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="unit-normalized embeddings"):
        criterion(z1, z2)
