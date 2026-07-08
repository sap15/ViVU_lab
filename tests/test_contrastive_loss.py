from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.losses import NTXentLoss, build_false_negative_mask


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


def test_nt_xent_with_none_mask_matches_baseline_loss() -> None:
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    z2 = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 1.2]], dtype=torch.float32)
    mask_output = build_false_negative_mask(batch_size=3, mode="none")

    baseline = criterion(z1, z2)
    masked = criterion(z1, z2, mask_output=mask_output)

    assert masked.loss.item() == pytest.approx(baseline.loss.item(), abs=1.0e-6)
    assert masked.mean_negative_similarity.item() == pytest.approx(
        baseline.mean_negative_similarity.item(),
        abs=1.0e-6,
    )
    assert masked.mask_stats is not None
    assert masked.mask_stats.mode == "none"


def test_nt_xent_same_position_mask_changes_denominator_as_expected() -> None:
    criterion = NTXentLoss(temperature=1.0, normalize=False)
    z1 = torch.eye(3, dtype=torch.float32)
    z2 = z1.clone()
    mask_output = build_false_negative_mask(
        batch_size=3,
        mode="same_position",
        positions=[100, 100, 220],
        min_valid_negatives=1,
        min_valid_fraction=0.1,
    )

    baseline_loss = criterion(z1, z2).loss.item()
    masked_output = criterion(z1, z2, mask_output=mask_output)
    # NT-Xent averages over 2 * batch_size anchors: here 6 anchors total.
    # Four anchors belong to the two variants at position 100 and keep only
    # two full negatives, while the two anchors for position 220 remain
    # unmasked and keep four full negatives.
    loss_anchor_same_position = -math.log(math.e / (math.e + 2.0))
    loss_anchor_unmasked = -math.log(math.e / (math.e + 4.0))
    expected_loss = (
        4.0 * loss_anchor_same_position + 2.0 * loss_anchor_unmasked
    ) / 6.0

    assert masked_output.loss.item() < baseline_loss
    assert masked_output.loss.item() == pytest.approx(expected_loss, abs=1.0e-6)


def test_nt_xent_structural_soft_applies_alpha_weights() -> None:
    criterion = NTXentLoss(temperature=1.0, normalize=False)
    z1 = torch.eye(3, dtype=torch.float32)
    z2 = z1.clone()
    mask_output = build_false_negative_mask(
        batch_size=3,
        mode="structural_soft",
        positions=[100, 150, 220],
        structural_neighbors=[
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ],
        alpha=0.5,
        min_valid_negatives=1,
        min_valid_fraction=0.1,
    )

    output = criterion(z1, z2, mask_output=mask_output)
    # NT-Xent averages over 2 * batch_size anchors: here 6 anchors total.
    # Four anchors belong to variants affected by the soft structural weight
    # alpha=0.5, so their denominator is e + 3; the two anchors for the
    # unaffected variant keep the baseline denominator e + 4.
    loss_anchor_soft = -math.log(math.e / (math.e + 3.0))
    loss_anchor_unmasked = -math.log(math.e / (math.e + 4.0))
    expected_loss = (4.0 * loss_anchor_soft + 2.0 * loss_anchor_unmasked) / 6.0

    assert output.loss.item() == pytest.approx(expected_loss, abs=1.0e-6)


def test_nt_xent_returns_mask_metrics_when_mask_is_provided() -> None:
    criterion = NTXentLoss()
    z1 = torch.randn(3, 4, dtype=torch.float32)
    z2 = torch.randn(3, 4, dtype=torch.float32)
    mask_output = build_false_negative_mask(batch_size=3, mode="none")

    output = criterion(z1, z2, mask_output=mask_output)

    assert output.mask_stats is not None
    assert output.mask_stats.mode == "none"


def test_nt_xent_accepts_direct_negative_weights() -> None:
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    z2 = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 1.2]], dtype=torch.float32)
    mask_output = build_false_negative_mask(batch_size=3, mode="none")

    from_output = criterion(z1, z2, mask_output=mask_output)
    from_weights = criterion(z1, z2, negative_weights=mask_output.negative_weights)

    assert from_weights.loss.item() == pytest.approx(from_output.loss.item(), abs=1.0e-6)
    assert from_weights.mask_stats is None
