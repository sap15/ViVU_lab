"""Intra-batch false-negative masking utilities for NT-Xent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class FalseNegativeAnchorStats:
    """Effective negative statistics for one NT-Xent anchor."""

    anchor_index: int
    potential_negatives: int
    valid_negatives: float
    valid_negative_fraction: float
    masked_negatives: float
    is_degenerate_anchor: bool


@dataclass(frozen=True)
class FalseNegativeBatchStats:
    """Batch-level summary of effective negatives after masking."""

    min_valid_negatives: float
    mean_valid_negatives: float
    min_valid_negative_fraction: float
    number_degenerate_anchors: int
    mode: str
    alpha: float | None
    has_degenerate_anchors: bool


@dataclass(frozen=True)
class FalseNegativeMaskOutput:
    """Expanded negative weights and diagnostics for a two-view NT-Xent batch."""

    negative_weights: Tensor
    per_anchor_stats: tuple[FalseNegativeAnchorStats, ...]
    batch_stats: FalseNegativeBatchStats


class FalseNegativeMaskDegenerateError(ValueError):
    """Raised when effective negatives fall below configured thresholds in strict mode."""


def _validate_mode(mode: str) -> str:
    supported = {"none", "same_position", "structural_hard", "structural_soft"}
    normalized = str(mode)
    if normalized not in supported:
        raise ValueError(f"Unsupported false-negative masking mode {mode!r}.")
    return normalized


def _validate_alpha(mode: str, alpha: float | None) -> float | None:
    if mode != "structural_soft":
        return None
    if alpha is None:
        raise ValueError("structural_soft mode requires an alpha value.")
    alpha_value = float(alpha)
    if not 0.0 <= alpha_value <= 1.0:
        raise ValueError("False-negative masking alpha must be within [0, 1].")
    return alpha_value


def _as_positions(positions: Sequence[int] | Tensor | None, *, batch_size: int) -> tuple[int, ...]:
    if positions is None:
        raise ValueError("positions are required for same_position and structural masking modes.")
    if isinstance(positions, Tensor):
        raw = positions.detach().cpu().tolist()
    else:
        raw = list(positions)
    if len(raw) != batch_size:
        raise ValueError(
            f"positions must contain exactly {batch_size} entries, received {len(raw)}."
        )
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be integer-like values.") from exc


def _as_structural_neighbors(
    structural_neighbors: Tensor | Sequence[Sequence[bool]] | None,
    *,
    batch_size: int,
) -> Tensor:
    if structural_neighbors is None:
        raise ValueError("structural_neighbors are required for structural masking modes.")
    matrix = torch.as_tensor(structural_neighbors, dtype=torch.bool)
    if matrix.shape != (batch_size, batch_size):
        raise ValueError(
            "structural_neighbors must be a square [batch_size, batch_size] matrix."
        )
    if not torch.equal(matrix, matrix.transpose(0, 1)):
        raise ValueError("structural_neighbors must be symmetric.")
    matrix = matrix.clone()
    matrix.fill_diagonal_(False)
    return matrix


def _build_variant_weights(
    batch_size: int,
    *,
    mode: str,
    positions: tuple[int, ...] | None,
    structural_neighbors: Tensor | None,
    alpha: float | None,
    combine_same_position: bool,
) -> Tensor:
    weights = torch.ones((batch_size, batch_size), dtype=torch.float32)
    weights.fill_diagonal_(0.0)

    if mode == "none":
        return weights

    if mode == "same_position":
        assert positions is not None
        for left in range(batch_size):
            for right in range(batch_size):
                if left != right and positions[left] == positions[right]:
                    weights[left, right] = 0.0
        return weights

    assert structural_neighbors is not None
    target_weight = 0.0 if mode == "structural_hard" else float(alpha)
    weights[structural_neighbors] = target_weight

    if combine_same_position:
        assert positions is not None
        for left in range(batch_size):
            for right in range(batch_size):
                if left != right and positions[left] == positions[right]:
                    weights[left, right] = 0.0
    return weights


def _expand_to_two_views(variant_weights: Tensor) -> Tensor:
    return torch.cat(
        [
            torch.cat([variant_weights, variant_weights], dim=1),
            torch.cat([variant_weights, variant_weights], dim=1),
        ],
        dim=0,
    )


def _build_stats(
    negative_weights: Tensor,
    *,
    mode: str,
    alpha: float | None,
    min_valid_negatives: float,
    min_valid_fraction: float,
) -> tuple[tuple[FalseNegativeAnchorStats, ...], FalseNegativeBatchStats]:
    total_samples = int(negative_weights.shape[0])
    potential_negatives = total_samples - 2
    per_anchor: list[FalseNegativeAnchorStats] = []

    for anchor_index in range(total_samples):
        valid_negatives = float(negative_weights[anchor_index].sum().item())
        valid_fraction = (
            valid_negatives / float(potential_negatives) if potential_negatives > 0 else 0.0
        )
        masked_negatives = float(potential_negatives) - valid_negatives
        is_degenerate = (
            valid_negatives < float(min_valid_negatives)
            or valid_fraction < float(min_valid_fraction)
        )
        per_anchor.append(
            FalseNegativeAnchorStats(
                anchor_index=anchor_index,
                potential_negatives=potential_negatives,
                valid_negatives=valid_negatives,
                valid_negative_fraction=valid_fraction,
                masked_negatives=masked_negatives,
                is_degenerate_anchor=is_degenerate,
            )
        )

    valid_counts = [entry.valid_negatives for entry in per_anchor]
    valid_fractions = [entry.valid_negative_fraction for entry in per_anchor]
    number_degenerate = sum(int(entry.is_degenerate_anchor) for entry in per_anchor)
    batch_stats = FalseNegativeBatchStats(
        min_valid_negatives=min(valid_counts, default=0.0),
        mean_valid_negatives=sum(valid_counts) / len(valid_counts) if valid_counts else 0.0,
        min_valid_negative_fraction=min(valid_fractions, default=0.0),
        number_degenerate_anchors=number_degenerate,
        mode=mode,
        alpha=alpha,
        has_degenerate_anchors=number_degenerate > 0,
    )
    return tuple(per_anchor), batch_stats


def build_false_negative_mask(
    batch_size: int,
    *,
    mode: str = "none",
    positions: Sequence[int] | Tensor | None = None,
    structural_neighbors: Tensor | Sequence[Sequence[bool]] | None = None,
    alpha: float | None = None,
    combine_same_position: bool = False,
    min_valid_negatives: float = 8,
    min_valid_fraction: float = 0.25,
    strict: bool = False,
) -> FalseNegativeMaskOutput:
    """Build intra-batch NT-Xent negative weights for two augmented views.

    The output matrix is shaped ``[2 * batch_size, 2 * batch_size]`` and assigns
    weights only to negative pairs. Self/self pairs and cross-view positives for
    the same mutant always receive weight ``0``.
    """

    if batch_size < 2:
        raise ValueError("False-negative masking requires batch_size >= 2.")
    if min_valid_negatives < 0.0:
        raise ValueError("min_valid_negatives must be non-negative.")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be within [0, 1].")

    normalized_mode = _validate_mode(mode)
    alpha_value = _validate_alpha(normalized_mode, alpha)
    positions_value: tuple[int, ...] | None = None
    if normalized_mode in {"same_position", "structural_hard", "structural_soft"} or combine_same_position:
        positions_value = _as_positions(positions, batch_size=batch_size)

    structural_matrix: Tensor | None = None
    if normalized_mode in {"structural_hard", "structural_soft"}:
        structural_matrix = _as_structural_neighbors(
            structural_neighbors,
            batch_size=batch_size,
        )

    variant_weights = _build_variant_weights(
        batch_size,
        mode=normalized_mode,
        positions=positions_value,
        structural_neighbors=structural_matrix,
        alpha=alpha_value,
        combine_same_position=combine_same_position,
    )
    negative_weights = _expand_to_two_views(variant_weights)
    per_anchor_stats, batch_stats = _build_stats(
        negative_weights,
        mode=normalized_mode,
        alpha=alpha_value,
        min_valid_negatives=min_valid_negatives,
        min_valid_fraction=min_valid_fraction,
    )

    if strict and batch_stats.has_degenerate_anchors:
        raise FalseNegativeMaskDegenerateError(
            "False-negative masking produced degenerate anchors under the configured thresholds."
        )

    return FalseNegativeMaskOutput(
        negative_weights=negative_weights,
        per_anchor_stats=per_anchor_stats,
        batch_stats=batch_stats,
    )
