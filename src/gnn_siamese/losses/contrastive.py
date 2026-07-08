"""Contrastive losses for two augmented views of the same mutant."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NTXentLossOutput:
    """Scalar NT-Xent loss with basic diagnostic metrics."""

    loss: Tensor
    mean_positive_similarity: Tensor
    mean_negative_similarity: Tensor
    temperature: float
    batch_size: int


class NTXentLoss(nn.Module):
    """Baseline NT-Xent loss for two augmented views of the same mutant.

    Positives are defined only between matching rows of ``z1`` and ``z2``.
    Mutant-WT relations, false-negative masking, ``L_relative_WT`` and
    relational losses are intentionally excluded from this baseline.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        *,
        normalize: bool = True,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("NT-Xent temperature must be strictly positive.")
        if eps <= 0.0:
            raise ValueError("NT-Xent epsilon must be strictly positive.")

        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def _validate_inputs(self, z1: Tensor, z2: Tensor) -> None:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("NT-Xent expects z1 and z2 shaped [batch, dim].")
        if z1.shape != z2.shape:
            raise ValueError(
                f"NT-Xent expects z1 and z2 with identical shape, got {tuple(z1.shape)} "
                f"and {tuple(z2.shape)}."
            )
        if z1.shape[0] < 2:
            raise ValueError("NT-Xent requires batch size >= 2 to provide valid negatives.")
        if z1.shape[1] < 1:
            raise ValueError("NT-Xent requires embedding dimension >= 1.")
        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            raise ValueError("NT-Xent received non-finite embeddings.")

    def _prepare_embeddings(self, z: Tensor) -> Tensor:
        if self.normalize:
            return F.normalize(z, p=2, dim=-1, eps=self.eps)

        norms = torch.linalg.vector_norm(z, ord=2, dim=-1)
        if not torch.isfinite(norms).all():
            raise ValueError("NT-Xent received embeddings with non-finite norms.")
        if torch.any(torch.abs(norms - 1.0) > 1.0e-4):
            raise ValueError(
                "NT-Xent expects unit-normalized embeddings when normalize=False."
            )
        return z

    def forward(self, z1: Tensor, z2: Tensor) -> NTXentLossOutput:
        self._validate_inputs(z1, z2)

        z1_normalized = self._prepare_embeddings(z1)
        z2_normalized = self._prepare_embeddings(z2)
        representations = torch.cat([z1_normalized, z2_normalized], dim=0)

        similarity_matrix = representations @ representations.transpose(0, 1)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        if not torch.isfinite(similarity_matrix).all():
            raise RuntimeError("NT-Xent produced a non-finite similarity matrix.")

        total_samples = similarity_matrix.shape[0]
        diagonal_mask = torch.eye(total_samples, device=similarity_matrix.device, dtype=torch.bool)
        logits = similarity_matrix / self.temperature
        logits = logits.masked_fill(diagonal_mask, float("-inf"))

        batch_size = z1.shape[0]
        positive_indices = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=logits.device),
                torch.arange(0, batch_size, device=logits.device),
            ]
        )

        loss = F.cross_entropy(logits, positive_indices)
        if not torch.isfinite(loss):
            raise RuntimeError("NT-Xent produced a non-finite loss.")

        positive_similarities = torch.sum(z1_normalized * z2_normalized, dim=-1)
        negative_mask = ~diagonal_mask
        row_indices = torch.arange(total_samples, device=logits.device)
        negative_mask[row_indices, positive_indices] = False
        negative_values = similarity_matrix[negative_mask]
        mean_negative_similarity = (
            negative_values.mean()
            if negative_values.numel() > 0
            else similarity_matrix.new_tensor(0.0)
        )

        return NTXentLossOutput(
            loss=loss,
            mean_positive_similarity=positive_similarities.mean(),
            mean_negative_similarity=mean_negative_similarity,
            temperature=self.temperature,
            batch_size=batch_size,
        )
