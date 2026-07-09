"""Loss functions for the PKP2 siamese GNN project."""

from gnn_siamese.losses.contrastive import NTXentLoss, NTXentLossOutput
from gnn_siamese.losses.false_negative_mask import (
    FalseNegativeAnchorStats,
    FalseNegativeBatchStats,
    FalseNegativeMaskDegenerateError,
    FalseNegativeMaskOutput,
    build_false_negative_mask,
)
from gnn_siamese.losses.relative_wt import RelativeWTLoss, RelativeWTLossOutput

__all__ = [
    "FalseNegativeAnchorStats",
    "FalseNegativeBatchStats",
    "FalseNegativeMaskDegenerateError",
    "FalseNegativeMaskOutput",
    "NTXentLoss",
    "NTXentLossOutput",
    "RelativeWTLoss",
    "RelativeWTLossOutput",
    "build_false_negative_mask",
]
