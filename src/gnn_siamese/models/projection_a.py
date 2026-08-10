"""Projection head for Model A pair-level contrastive learning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModelAProjectionHead(nn.Module):
    """Project ``z_delta_pair`` into the technical ``z_instance_pair`` space.

    Outputs are intentionally not L2-normalized: the shared ``NTXentLoss``
    performs that normalization exactly once.
    """

    def __init__(
        self,
        *,
        z_delta_pair_dim: int,
        hidden_dim: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("z_delta_pair_dim", z_delta_pair_dim),
            ("hidden_dim", hidden_dim),
            ("projection_dim", projection_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.z_delta_pair_dim = z_delta_pair_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.network = nn.Sequential(
            nn.Linear(z_delta_pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, z_delta_pair: Tensor) -> Tensor:
        """Return unnormalized ``z_instance_pair`` with shape ``[B, P]``."""

        if not isinstance(z_delta_pair, Tensor):
            raise TypeError("z_delta_pair must be a torch.Tensor.")
        if z_delta_pair.ndim != 2:
            raise ValueError(
                "z_delta_pair must have shape [batch_size, z_delta_pair_dim], "
                f"got {tuple(z_delta_pair.shape)}."
            )
        if z_delta_pair.shape[-1] != self.z_delta_pair_dim:
            raise ValueError(
                "z_delta_pair last dimension must equal "
                f"z_delta_pair_dim={self.z_delta_pair_dim}, got "
                f"{z_delta_pair.shape[-1]}."
            )
        if not z_delta_pair.is_floating_point():
            raise TypeError("z_delta_pair must use a floating-point dtype.")
        if not torch.isfinite(z_delta_pair).all():
            raise ValueError("z_delta_pair contains NaN or Inf.")
        return self.network(z_delta_pair)
