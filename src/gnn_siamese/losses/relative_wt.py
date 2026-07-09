"""Auxiliary Mutant-WT relational loss modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RelativeWTLossOutput:
    """Relative WT loss output with diagnostics."""

    loss: Tensor
    mode: str
    is_active: bool
    mean_distance: Tensor
    margin: float | None = None
    num_pairs: int | None = None
    target_name: str | None = None
    prediction: Tensor | None = None


class RelativeWTLoss(nn.Module):
    """Auxiliary loss that keeps WT as reference instead of strong positive."""

    def __init__(
        self,
        mode: str = "none",
        *,
        distance: str = "euclidean",
        margin: float = 0.0,
        direction: str = "min",
        stop_gradient_wt: bool = False,
        predictive_loss: str = "mse",
        allow_energy_target: bool = False,
        predictor: nn.Module | None = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"none", "margin", "ranking", "predictive"}:
            raise ValueError(f"Unsupported RelativeWTLoss mode {mode!r}.")
        normalized_distance = distance.strip().lower()
        if normalized_distance not in {"euclidean", "l2", "cosine", "l1"}:
            raise ValueError(f"Unsupported distance metric {distance!r}.")
        normalized_direction = direction.strip().lower()
        if normalized_direction not in {"min", "max"}:
            raise ValueError(f"Unsupported direction {direction!r}.")
        normalized_predictive_loss = predictive_loss.strip().lower()
        if normalized_predictive_loss not in {"mse", "smooth_l1"}:
            raise ValueError(f"Unsupported predictive loss {predictive_loss!r}.")
        if margin < 0.0:
            raise ValueError("RelativeWTLoss margin must be non-negative.")
        if eps <= 0.0:
            raise ValueError("RelativeWTLoss epsilon must be strictly positive.")

        self.mode = normalized_mode
        self.distance = normalized_distance
        self.margin = float(margin)
        self.direction = normalized_direction
        self.stop_gradient_wt = bool(stop_gradient_wt)
        self.predictive_loss = normalized_predictive_loss
        self.allow_energy_target = bool(allow_energy_target)
        self.predictor = predictor
        self.eps = float(eps)

    def _validate_embeddings(self, h_mut: Tensor, h_wt: Tensor) -> None:
        if h_mut.ndim != 2 or h_wt.ndim != 2:
            raise ValueError("RelativeWTLoss expects h_mut and h_wt shaped [batch, dim].")
        if h_mut.shape != h_wt.shape:
            raise ValueError(
                "RelativeWTLoss expects h_mut and h_wt with identical shape, got "
                f"{tuple(h_mut.shape)} and {tuple(h_wt.shape)}."
            )
        if h_mut.shape[0] < 1 or h_mut.shape[1] < 1:
            raise ValueError("RelativeWTLoss expects non-empty 2D embeddings.")
        if not torch.isfinite(h_mut).all() or not torch.isfinite(h_wt).all():
            raise ValueError("RelativeWTLoss received non-finite embeddings.")

    def _compute_distances(self, h_mut: Tensor, h_wt: Tensor) -> Tensor:
        reference = h_wt.detach() if self.stop_gradient_wt else h_wt
        if self.distance in {"euclidean", "l2"}:
            distances = torch.linalg.vector_norm(h_mut - reference, ord=2, dim=-1)
        elif self.distance == "l1":
            distances = torch.linalg.vector_norm(h_mut - reference, ord=1, dim=-1)
        else:
            similarity = F.cosine_similarity(h_mut, reference, dim=-1, eps=self.eps)
            distances = 1.0 - similarity
        if not torch.isfinite(distances).all():
            raise RuntimeError("RelativeWTLoss produced non-finite distances.")
        return distances

    def _zero_output(self, h_mut: Tensor, h_wt: Tensor, *, mean_distance: Tensor) -> RelativeWTLossOutput:
        zero_loss = (h_mut.sum() * 0.0) + (h_wt.sum() * 0.0)
        return RelativeWTLossOutput(
            loss=zero_loss,
            mode=self.mode,
            is_active=False,
            mean_distance=mean_distance,
            margin=self.margin if self.mode == "margin" else None,
        )

    def _compute_margin_loss(self, distances: Tensor) -> Tensor:
        if self.direction == "min":
            violation = distances - self.margin
        else:
            violation = self.margin - distances
        return 0.5 * torch.square(torch.relu(violation)).mean()

    def _require_target(self, target: Tensor | None, *, mode_name: str) -> Tensor:
        if target is None:
            raise ValueError(f"RelativeWTLoss mode={mode_name!r} requires an explicit target.")
        if target.ndim != 1:
            raise ValueError("RelativeWTLoss target must be shaped [batch].")
        if not torch.isfinite(target).all():
            raise ValueError("RelativeWTLoss target must be finite.")
        return target.to(dtype=torch.float32)

    def _compute_ranking_loss(self, distances: Tensor, target: Tensor) -> tuple[Tensor, int]:
        target_deltas = target.unsqueeze(1) - target.unsqueeze(0)
        valid_mask = torch.triu(target_deltas != 0, diagonal=1)
        if not torch.any(valid_mask):
            return distances.sum() * 0.0, 0

        expected_sign = torch.sign(target_deltas[valid_mask])
        distance_deltas = distances.unsqueeze(1) - distances.unsqueeze(0)
        pairwise_margin = self.margin - (expected_sign * distance_deltas[valid_mask])
        loss = 0.5 * torch.square(torch.relu(pairwise_margin)).mean()
        return loss, int(valid_mask.sum().item())

    def _predict_from_inputs(self, h_mut: Tensor, h_wt: Tensor, distances: Tensor) -> Tensor:
        if self.predictor is None:
            return distances

        predictor_input = torch.cat([h_mut, h_wt, h_mut - h_wt], dim=-1)
        prediction = self.predictor(predictor_input)
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(-1)
        return prediction

    def _compute_predictive_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.ndim != 1 or prediction.shape != target.shape:
            raise ValueError(
                "RelativeWTLoss predictive mode expects 1D predictions matching the target "
                f"shape, got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if self.predictive_loss == "mse":
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target)

    def forward(
        self,
        h_mut: Tensor,
        h_wt: Tensor,
        *,
        severity_target: Tensor | None = None,
        auxiliary_target: Tensor | None = None,
        ranking_target: Tensor | None = None,
        target_name: str | None = None,
    ) -> RelativeWTLossOutput:
        self._validate_embeddings(h_mut, h_wt)
        distances = self._compute_distances(h_mut, h_wt)
        mean_distance = distances.mean()

        if self.mode == "none":
            return self._zero_output(h_mut, h_wt, mean_distance=mean_distance)

        if self.mode == "margin":
            return RelativeWTLossOutput(
                loss=self._compute_margin_loss(distances),
                mode=self.mode,
                is_active=True,
                mean_distance=mean_distance,
                margin=self.margin,
            )

        if self.mode == "ranking":
            target = ranking_target if ranking_target is not None else severity_target
            ranking_values = self._require_target(target, mode_name=self.mode).to(device=distances.device)
            if ranking_values.shape[0] != distances.shape[0]:
                raise ValueError("RelativeWTLoss ranking target length must match batch size.")
            loss, num_pairs = self._compute_ranking_loss(distances, ranking_values)
            return RelativeWTLossOutput(
                loss=loss,
                mode=self.mode,
                is_active=num_pairs > 0,
                mean_distance=mean_distance,
                margin=self.margin,
                num_pairs=num_pairs,
                target_name=target_name or "severity_target",
            )

        target = self._require_target(auxiliary_target, mode_name=self.mode).to(device=distances.device)
        if target.shape[0] != distances.shape[0]:
            raise ValueError("RelativeWTLoss predictive target length must match batch size.")
        normalized_target_name = target_name or "auxiliary_target"
        if normalized_target_name == "custom_structure_energy" and not self.allow_energy_target:
            raise ValueError(
                "RelativeWTLoss predictive mode does not allow custom_structure_energy "
                "unless allow_energy_target=True."
            )
        prediction = self._predict_from_inputs(h_mut, h_wt, distances)
        loss = self._compute_predictive_loss(prediction, target)
        return RelativeWTLossOutput(
            loss=loss,
            mode=self.mode,
            is_active=True,
            mean_distance=mean_distance,
            target_name=normalized_target_name,
            prediction=prediction,
        )
