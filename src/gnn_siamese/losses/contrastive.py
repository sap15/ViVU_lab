"""Contrastive losses for two augmented views of the same mutant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gnn_siamese.losses.false_negative_mask import FalseNegativeMaskOutput


@dataclass(frozen=True)
class NTXentLossOutput:
    """Scalar NT-Xent loss with basic diagnostic metrics."""

    loss: Tensor
    mean_positive_similarity: Tensor
    mean_negative_similarity: Tensor
    temperature: float
    batch_size: int
    mask_stats: Any | None = None


class NTXentLoss(nn.Module):
    """NT-Xent loss for two augmented views of the same mutant.

    Positives are defined only between matching rows of ``z1`` and ``z2``.
    Mutant-WT relations, ``L_relative_WT`` and relational losses remain
    intentionally excluded here. False-negative masking is optional, operates
    only within the batch and does not alter the positive-pair definition.
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

    def _validate_negative_weights(
        self,
        negative_weights: Tensor,
        *,
        total_samples: int,
        positive_indices: Tensor,
    ) -> Tensor:
        if negative_weights.shape != (total_samples, total_samples):
            raise ValueError(
                "negative_weights must match the expanded similarity matrix shape "
                f"({total_samples}, {total_samples})."
            )
        if not torch.isfinite(negative_weights).all():
            raise ValueError("negative_weights must be finite.")
        if (negative_weights < 0.0).any() or (negative_weights > 1.0).any():
            raise ValueError("negative_weights entries must be within [0, 1].")

        weights = negative_weights.to(dtype=torch.float32)
        row_indices = torch.arange(total_samples, device=weights.device)
        weights = weights.clone()
        weights[row_indices, row_indices] = 0.0
        weights[row_indices, positive_indices] = 0.0
        return weights

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        *,
        negative_weights: Tensor | None = None,
        mask_output: FalseNegativeMaskOutput | None = None,
    ) -> NTXentLossOutput:
        self._validate_inputs(z1, z2)
        if negative_weights is not None and mask_output is not None:
            raise ValueError("Provide either negative_weights or mask_output, not both.")

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

        total_samples = logits.shape[0]
        mask_stats = None
        if mask_output is not None:
            negative_weights = mask_output.negative_weights
            mask_stats = mask_output.batch_stats

        if negative_weights is None:
            loss = F.cross_entropy(logits, positive_indices)
        else:
            weights = self._validate_negative_weights(
                negative_weights.to(device=logits.device),
                total_samples=total_samples,
                positive_indices=positive_indices,
            )
            positive_logits = logits[
                torch.arange(total_samples, device=logits.device),
                positive_indices,
            ]
            negative_exp = torch.exp(logits) * weights
            denominator = torch.exp(positive_logits) + negative_exp.sum(dim=1)
            loss = -(positive_logits - torch.log(denominator)).mean()

        if not torch.isfinite(loss):
            raise RuntimeError("NT-Xent produced a non-finite loss.")

        positive_similarities = torch.sum(z1_normalized * z2_normalized, dim=-1)
        negative_mask = ~diagonal_mask
        row_indices = torch.arange(total_samples, device=logits.device)
        negative_mask[row_indices, positive_indices] = False
        if negative_weights is None:
            negative_values = similarity_matrix[negative_mask]
            mean_negative_similarity = (
                negative_values.mean()
                if negative_values.numel() > 0
                else similarity_matrix.new_tensor(0.0)
            )
        else:
            weights = self._validate_negative_weights(
                negative_weights.to(device=logits.device),
                total_samples=total_samples,
                positive_indices=positive_indices,
            )
            weighted_sum = (similarity_matrix * weights).sum()
            total_weight = weights.sum()
            mean_negative_similarity = (
                weighted_sum / total_weight
                if total_weight.item() > 0.0
                else similarity_matrix.new_tensor(0.0)
            )

        return NTXentLossOutput(
            loss=loss,
            mean_positive_similarity=positive_similarities.mean(),
            mean_negative_similarity=mean_negative_similarity,
            temperature=self.temperature,
            batch_size=batch_size,
            mask_stats=mask_stats,
        )
