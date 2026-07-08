"""Loss functions for the PKP2 siamese GNN project."""

from gnn_siamese.losses.contrastive import NTXentLoss, NTXentLossOutput

__all__ = ["NTXentLoss", "NTXentLossOutput"]
