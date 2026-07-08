from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.losses.false_negative_mask import (
    FalseNegativeMaskDegenerateError,
    build_false_negative_mask,
)


def test_mode_none_produces_expected_negative_weight_matrix() -> None:
    output = build_false_negative_mask(batch_size=3, mode="none")

    expected = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(output.negative_weights, expected)


def test_same_position_masks_pairs_with_same_position() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="same_position",
        positions=[100, 100, 220],
    )

    assert output.negative_weights[0, 1].item() == 0.0
    assert output.negative_weights[0, 4].item() == 0.0
    assert output.negative_weights[0, 2].item() == 1.0
    assert output.negative_weights[0, 5].item() == 1.0


def test_structural_hard_masks_structural_neighbors() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="structural_hard",
        positions=[100, 150, 300],
        structural_neighbors=[
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ],
    )

    assert output.negative_weights[0, 1].item() == 0.0
    assert output.negative_weights[0, 4].item() == 0.0
    assert output.negative_weights[0, 2].item() == 1.0


def test_structural_soft_assigns_alpha_to_structural_neighbors() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="structural_soft",
        positions=[100, 150, 300],
        structural_neighbors=[
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ],
        alpha=0.25,
    )

    assert output.negative_weights[0, 1].item() == pytest.approx(0.25)
    assert output.negative_weights[0, 4].item() == pytest.approx(0.25)
    assert output.negative_weights[0, 2].item() == pytest.approx(1.0)


def test_alpha_out_of_range_fails() -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        build_false_negative_mask(
            batch_size=3,
            mode="structural_soft",
            positions=[1, 2, 3],
            structural_neighbors=torch.zeros((3, 3), dtype=torch.bool),
            alpha=1.5,
        )


def test_positive_pairs_and_self_pairs_are_not_counted_as_negatives() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="same_position",
        positions=[100, 100, 220],
    )

    assert output.negative_weights[0, 0].item() == 0.0
    assert output.negative_weights[0, 3].item() == 0.0
    assert output.per_anchor_stats[0].potential_negatives == 4


def test_per_anchor_stats_are_computed() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="same_position",
        positions=[100, 100, 220],
        min_valid_negatives=1,
        min_valid_fraction=0.1,
    )

    anchor0 = output.per_anchor_stats[0]
    assert anchor0.potential_negatives == 4
    assert anchor0.valid_negatives == pytest.approx(2.0)
    assert anchor0.valid_negative_fraction == pytest.approx(0.5)
    assert anchor0.masked_negatives == pytest.approx(2.0)
    assert anchor0.is_degenerate_anchor is False


def test_degenerate_anchors_are_detected() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="same_position",
        positions=[100, 100, 220],
        min_valid_negatives=3,
        min_valid_fraction=0.8,
    )

    assert output.batch_stats.number_degenerate_anchors == 4
    assert output.batch_stats.has_degenerate_anchors is True
    assert output.batch_stats.min_valid_negatives == pytest.approx(2.0)
    assert output.batch_stats.min_valid_negative_fraction == pytest.approx(0.5)


def test_strict_true_fails_on_degenerate_anchors() -> None:
    with pytest.raises(FalseNegativeMaskDegenerateError, match="degenerate anchors"):
        build_false_negative_mask(
            batch_size=3,
            mode="same_position",
            positions=[100, 100, 220],
            min_valid_negatives=3,
            min_valid_fraction=0.8,
            strict=True,
        )


def test_strict_false_does_not_fail_on_degenerate_anchors() -> None:
    output = build_false_negative_mask(
        batch_size=3,
        mode="same_position",
        positions=[100, 100, 220],
        min_valid_negatives=3,
        min_valid_fraction=0.8,
        strict=False,
    )

    assert output.batch_stats.has_degenerate_anchors is True
