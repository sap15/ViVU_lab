"""Loss functions for the PKP2 siamese GNN project."""

from gnn_siamese.losses.contrastive import NTXentLoss, NTXentLossOutput
from gnn_siamese.losses.false_negative_mask import (
    FalseNegativeAnchorStats,
    FalseNegativeBatchStats,
    FalseNegativeMaskDegenerateError,
    FalseNegativeMaskOutput,
    build_false_negative_mask,
)

__all__ = [
    "FalseNegativeAnchorStats",
    "FalseNegativeBatchStats",
    "FalseNegativeMaskDegenerateError",
    "FalseNegativeMaskOutput",
    "NTXentLoss",
    "NTXentLossOutput",
    "build_false_negative_mask",
]
