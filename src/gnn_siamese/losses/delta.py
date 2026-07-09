"""Auxiliary losses for learned Mutant-WT relational representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DeltaLossOutput:
    """Delta loss output with diagnostics for relational embeddings."""

    loss: Tensor
    mode: str
    is_active: bool
    batch_size: int
    embedding_dim: int
    mean_norm: Tensor
    variance_metric: Tensor | None = None
    covariance_metric: Tensor | None = None
    target_name: str | None = None
    prediction: Tensor | None = None


class DeltaLoss(nn.Module):
    """Auxiliary loss modes for relational embeddings derived from Mutant-WT pairs."""

    def __init__(
        self,
        mode: str = "none",
        *,
        consistency_loss: str = "mse",
        gamma: float = 1.0,
        descriptor_loss: str = "mse",
        allow_energy_target: bool = False,
        predictor: nn.Module | None = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"none", "consistency", "variance", "covariance", "descriptor"}:
            raise ValueError(f"Unsupported DeltaLoss mode {mode!r}.")
        normalized_consistency_loss = consistency_loss.strip().lower()
        if normalized_consistency_loss not in {"mse", "smooth_l1"}:
            raise ValueError(f"Unsupported consistency loss {consistency_loss!r}.")
        normalized_descriptor_loss = descriptor_loss.strip().lower()
        if normalized_descriptor_loss not in {"mse", "smooth_l1", "cosine"}:
            raise ValueError(f"Unsupported descriptor loss {descriptor_loss!r}.")
        if gamma < 0.0:
            raise ValueError("DeltaLoss gamma must be non-negative.")
        if eps <= 0.0:
            raise ValueError("DeltaLoss epsilon must be strictly positive.")

        self.mode = normalized_mode
        self.consistency_loss = normalized_consistency_loss
        self.gamma = float(gamma)
        self.descriptor_loss = normalized_descriptor_loss
        self.allow_energy_target = bool(allow_energy_target)
        self.predictor = predictor
        self.eps = float(eps)

    def _validate_embeddings(self, z_delta: Tensor, *, name: str) -> None:
        if z_delta.ndim != 2:
            raise ValueError(f"DeltaLoss expects {name} shaped [batch, dim].")
        if z_delta.shape[0] < 1 or z_delta.shape[1] < 1:
            raise ValueError("DeltaLoss expects non-empty 2D relational embeddings.")
        if not torch.isfinite(z_delta).all():
            raise ValueError(f"DeltaLoss received non-finite values in {name}.")

    def _validate_pair(self, z_delta_1: Tensor, z_delta_2: Tensor) -> None:
        if z_delta_1.shape != z_delta_2.shape:
            raise ValueError(
                "DeltaLoss expects both relational views to share the same shape, got "
                f"{tuple(z_delta_1.shape)} and {tuple(z_delta_2.shape)}."
            )

    def _require_batch_at_least_two(self, z_delta: Tensor, *, mode_name: str) -> None:
        if z_delta.shape[0] < 2:
            raise ValueError(f"DeltaLoss mode={mode_name!r} requires batch size >= 2.")

    def _base_output(
        self,
        z_delta: Tensor,
        *,
        loss: Tensor,
        is_active: bool,
        variance_metric: Tensor | None = None,
        covariance_metric: Tensor | None = None,
        target_name: str | None = None,
        prediction: Tensor | None = None,
    ) -> DeltaLossOutput:
        return DeltaLossOutput(
            loss=loss,
            mode=self.mode,
            is_active=is_active,
            batch_size=int(z_delta.shape[0]),
            embedding_dim=int(z_delta.shape[1]),
            mean_norm=torch.linalg.vector_norm(z_delta, ord=2, dim=-1).mean(),
            variance_metric=variance_metric,
            covariance_metric=covariance_metric,
            target_name=target_name,
            prediction=prediction,
        )

    def _zero_output(self, z_delta: Tensor) -> DeltaLossOutput:
        zero_loss = z_delta.sum() * 0.0
        return self._base_output(z_delta, loss=zero_loss, is_active=False)

    def _compute_consistency_loss(self, z_delta_1: Tensor, z_delta_2: Tensor) -> Tensor:
        if self.consistency_loss == "mse":
            return F.mse_loss(z_delta_1, z_delta_2)
        return F.smooth_l1_loss(z_delta_1, z_delta_2)

    def _compute_variance_loss(self, z_delta: Tensor) -> tuple[Tensor, Tensor]:
        centered = z_delta - z_delta.mean(dim=0, keepdim=True)
        per_dim_std = torch.sqrt(centered.var(dim=0, unbiased=False) + self.eps)
        loss = torch.relu(self.gamma - per_dim_std).mean()
        return loss, per_dim_std.mean()

    @staticmethod
    def _off_diagonal(matrix: Tensor) -> Tensor:
        dim = matrix.shape[0]
        return matrix[~torch.eye(dim, device=matrix.device, dtype=torch.bool)]

    def _compute_covariance_loss(self, z_delta: Tensor) -> tuple[Tensor, Tensor]:
        centered = z_delta - z_delta.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / (z_delta.shape[0] - 1)
        off_diagonal = self._off_diagonal(covariance)
        loss = off_diagonal.square().mean()
        metric = off_diagonal.abs().mean()
        return loss, metric

    def _require_target(self, target: Tensor | None) -> Tensor:
        if target is None:
            raise ValueError("DeltaLoss descriptor mode requires an explicit target.")
        if target.ndim not in {1, 2}:
            raise ValueError("DeltaLoss target must be shaped [batch] or [batch, dim].")
        if not torch.isfinite(target).all():
            raise ValueError("DeltaLoss target must be finite.")
        return target.to(dtype=torch.float32)

    def _predict_descriptor(self, z_delta: Tensor) -> Tensor:
        if self.predictor is None:
            return z_delta

        prediction = self.predictor(z_delta)
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(-1)
        return prediction

    def _compute_descriptor_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "DeltaLoss descriptor mode expects prediction and target with identical "
                f"shape, got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if self.descriptor_loss == "mse":
            return F.mse_loss(prediction, target)
        if self.descriptor_loss == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        if prediction.ndim != 2:
            raise ValueError("DeltaLoss cosine descriptor loss requires rank-2 tensors.")
        return (1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=self.eps)).mean()

    def forward(
        self,
        z_delta_1: Tensor,
        *,
        z_delta_2: Tensor | None = None,
        target: Tensor | None = None,
        target_name: str | None = None,
    ) -> DeltaLossOutput:
        self._validate_embeddings(z_delta_1, name="z_delta_1")

        if self.mode == "none":
            return self._zero_output(z_delta_1)

        if self.mode == "consistency":
            if z_delta_2 is None:
                raise ValueError("DeltaLoss consistency mode requires z_delta_2.")
            self._validate_embeddings(z_delta_2, name="z_delta_2")
            self._validate_pair(z_delta_1, z_delta_2)
            loss = self._compute_consistency_loss(z_delta_1, z_delta_2)
            return self._base_output(z_delta_1, loss=loss, is_active=True)

        self._require_batch_at_least_two(z_delta_1, mode_name=self.mode)

        if self.mode == "variance":
            loss, variance_metric = self._compute_variance_loss(z_delta_1)
            return self._base_output(
                z_delta_1,
                loss=loss,
                is_active=True,
                variance_metric=variance_metric,
            )

        if self.mode == "covariance":
            loss, covariance_metric = self._compute_covariance_loss(z_delta_1)
            return self._base_output(
                z_delta_1,
                loss=loss,
                is_active=True,
                covariance_metric=covariance_metric,
            )

        if not target_name:
            raise ValueError("DeltaLoss descriptor mode requires target_name.")
        if target_name == "custom_structure_energy" and not self.allow_energy_target:
            raise ValueError(
                "DeltaLoss descriptor mode does not allow custom_structure_energy "
                "unless allow_energy_target=True."
            )

        target_tensor = self._require_target(target).to(device=z_delta_1.device)
        if target_tensor.shape[0] != z_delta_1.shape[0]:
            raise ValueError("DeltaLoss descriptor target length must match batch size.")
        prediction = self._predict_descriptor(z_delta_1)
        if target_tensor.ndim == 1 and prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(-1)
        loss = self._compute_descriptor_loss(prediction, target_tensor)
        return self._base_output(
            z_delta_1,
            loss=loss,
            is_active=True,
            target_name=target_name,
            prediction=prediction,
        )
