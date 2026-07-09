"""Minimal explicit assembly of the current training losses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from gnn_siamese.losses import DeltaLoss, NTXentLoss, RelativeWTLoss
from gnn_siamese.losses.false_negative_mask import FalseNegativeMaskOutput


@dataclass(frozen=True)
class TotalLossConfig:
    """Minimal config container for explicit `L_total` assembly."""

    nt_xent_weight: float = 1.0
    relative_wt_weight: float = 0.0
    delta_weight: float = 0.0
    nt_xent_kwargs: dict[str, Any] = field(default_factory=dict)
    relative_wt_kwargs: dict[str, Any] = field(default_factory=lambda: {"mode": "none"})
    delta_kwargs: dict[str, Any] = field(default_factory=lambda: {"mode": "none"})


@dataclass(frozen=True)
class TotalLossOutput:
    """Structured output for explicit weighted loss composition."""

    loss: Tensor
    components: dict[str, Tensor]
    weights: dict[str, float]
    active_components: list[str]
    inactive_components: list[str]
    skipped_components: list[str]
    metrics: dict[str, Any]
    audit_flags: dict[str, Any]


class TotalLossAssembler(nn.Module):
    """Combine NT-Xent, Relative-WT and Delta losses into an explicit `L_total`."""

    COMPONENT_NAMES = ("nt_xent", "relative_wt", "delta")

    def __init__(
        self,
        *,
        nt_xent: NTXentLoss | None = None,
        relative_wt: RelativeWTLoss | None = None,
        delta: DeltaLoss | None = None,
        nt_xent_weight: float = 1.0,
        relative_wt_weight: float = 0.0,
        delta_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.nt_xent = nt_xent if nt_xent is not None else NTXentLoss()
        self.relative_wt = relative_wt if relative_wt is not None else RelativeWTLoss(mode="none")
        self.delta = delta if delta is not None else DeltaLoss(mode="none")
        self.weights = {
            "nt_xent": self._validate_weight(nt_xent_weight, name="nt_xent_weight"),
            "relative_wt": self._validate_weight(relative_wt_weight, name="relative_wt_weight"),
            "delta": self._validate_weight(delta_weight, name="delta_weight"),
        }

    @classmethod
    def from_config(cls, config: TotalLossConfig | Mapping[str, Any]) -> "TotalLossAssembler":
        """Build the assembler from a minimal dataclass or mapping config."""

        if isinstance(config, Mapping):
            config = TotalLossConfig(
                nt_xent_weight=float(config.get("nt_xent_weight", 1.0)),
                relative_wt_weight=float(config.get("relative_wt_weight", 0.0)),
                delta_weight=float(config.get("delta_weight", 0.0)),
                nt_xent_kwargs=dict(config.get("nt_xent_kwargs", {})),
                relative_wt_kwargs=dict(config.get("relative_wt_kwargs", {"mode": "none"})),
                delta_kwargs=dict(config.get("delta_kwargs", {"mode": "none"})),
            )

        return cls(
            nt_xent=NTXentLoss(**config.nt_xent_kwargs),
            relative_wt=RelativeWTLoss(**config.relative_wt_kwargs),
            delta=DeltaLoss(**config.delta_kwargs),
            nt_xent_weight=config.nt_xent_weight,
            relative_wt_weight=config.relative_wt_weight,
            delta_weight=config.delta_weight,
        )

    @staticmethod
    def _validate_weight(weight: float, *, name: str) -> float:
        value = float(weight)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return value

    @staticmethod
    def _infer_reference_tensor(tensors: list[Tensor | None]) -> Tensor | None:
        for tensor in tensors:
            if tensor is not None:
                return tensor
        return None

    @staticmethod
    def _zero_loss(reference: Tensor | None) -> Tensor:
        if reference is not None:
            return reference.sum() * 0.0
        return torch.tensor(0.0, dtype=torch.float32)

    @staticmethod
    def _as_float(value: Tensor | float | int | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _is_mode_none(self, component_name: str) -> bool:
        if component_name == "relative_wt":
            return self.relative_wt.mode == "none"
        if component_name == "delta":
            return self.delta.mode == "none"
        return False

    def forward(
        self,
        *,
        z1: Tensor | None = None,
        z2: Tensor | None = None,
        negative_weights: Tensor | None = None,
        mask_output: FalseNegativeMaskOutput | None = None,
        h_mut: Tensor | None = None,
        h_wt: Tensor | None = None,
        severity_target: Tensor | None = None,
        auxiliary_target: Tensor | None = None,
        ranking_target: Tensor | None = None,
        relative_wt_target_name: str | None = None,
        z_delta: Tensor | None = None,
        z_delta_2: Tensor | None = None,
        delta_target: Tensor | None = None,
        delta_target_name: str | None = None,
    ) -> TotalLossOutput:
        reference = self._infer_reference_tensor([z1, z2, h_mut, h_wt, z_delta, z_delta_2])
        zero = self._zero_loss(reference)
        components = {name: zero for name in self.COMPONENT_NAMES}
        metrics: dict[str, Any] = {}
        audit_flags: dict[str, Any] = {
            "component_status": {},
            "weight_zero_components": [],
            "all_components_inactive": False,
            "z_delta_not_trained": False,
        }
        active_components: list[str] = []
        inactive_components: list[str] = []
        skipped_components: list[str] = []
        weighted_losses: list[Tensor] = []

        for name in self.COMPONENT_NAMES:
            weight = self.weights[name]
            if weight == 0.0:
                skipped_components.append(name)
                audit_flags["component_status"][name] = "skipped_weight_zero"
                audit_flags["weight_zero_components"].append(name)
                continue
            if self._is_mode_none(name):
                inactive_components.append(name)
                audit_flags["component_status"][name] = "inactive_mode_none"
                if name == "delta":
                    audit_flags["z_delta_not_trained"] = True
                continue

            if name == "nt_xent":
                if z1 is None or z2 is None:
                    raise ValueError("nt_xent_weight > 0 requires both z1 and z2.")
                output = self.nt_xent(
                    z1,
                    z2,
                    negative_weights=negative_weights,
                    mask_output=mask_output,
                )
                components[name] = output.loss
                weighted_losses.append(weight * output.loss)
                active_components.append(name)
                audit_flags["component_status"][name] = "active"
                metrics["nt_xent_mean_positive_similarity"] = output.mean_positive_similarity
                metrics["nt_xent_mean_negative_similarity"] = output.mean_negative_similarity
                metrics["nt_xent_temperature"] = output.temperature
                metrics["nt_xent_batch_size"] = output.batch_size
                metrics["nt_xent_is_active"] = True
                continue

            if name == "relative_wt":
                if h_mut is None or h_wt is None:
                    raise ValueError("relative_wt_weight > 0 requires both h_mut and h_wt.")
                output = self.relative_wt(
                    h_mut,
                    h_wt,
                    severity_target=severity_target,
                    auxiliary_target=auxiliary_target,
                    ranking_target=ranking_target,
                    target_name=relative_wt_target_name,
                )
                components[name] = output.loss
                weighted_losses.append(weight * output.loss)
                active_components.append(name)
                audit_flags["component_status"][name] = "active"
                metrics["relative_wt_mode"] = output.mode
                metrics["relative_wt_is_active"] = output.is_active
                metrics["relative_wt_mean_distance"] = output.mean_distance
                metrics["relative_wt_margin"] = output.margin
                metrics["relative_wt_num_pairs"] = output.num_pairs
                metrics["relative_wt_target_name"] = output.target_name
                continue

            if z_delta is None:
                raise ValueError("delta_weight > 0 requires z_delta.")
            output = self.delta(
                z_delta,
                z_delta_2=z_delta_2,
                target=delta_target,
                target_name=delta_target_name,
            )
            components[name] = output.loss
            weighted_losses.append(weight * output.loss)
            active_components.append(name)
            audit_flags["component_status"][name] = "active"
            metrics["delta_mode"] = output.mode
            metrics["delta_is_active"] = output.is_active
            metrics["delta_batch_size"] = output.batch_size
            metrics["delta_embedding_dim"] = output.embedding_dim
            metrics["delta_mean_norm"] = output.mean_norm
            metrics["delta_variance_metric"] = output.variance_metric
            metrics["delta_covariance_metric"] = output.covariance_metric
            metrics["delta_target_name"] = output.target_name

        if weighted_losses:
            total_loss = torch.stack(weighted_losses).sum()
        else:
            total_loss = zero
            audit_flags["all_components_inactive"] = True

        if "delta" not in active_components:
            audit_flags["z_delta_not_trained"] = True

        metrics["loss_total"] = total_loss
        metrics["active_component_count"] = len(active_components)
        metrics["inactive_component_count"] = len(inactive_components)
        metrics["skipped_component_count"] = len(skipped_components)
        metrics["weighted_components"] = {
            name: self.weights[name] * self._as_float(components[name])
            for name in self.COMPONENT_NAMES
        }

        return TotalLossOutput(
            loss=total_loss,
            components=components,
            weights=dict(self.weights),
            active_components=active_components,
            inactive_components=inactive_components,
            skipped_components=skipped_components,
            metrics=metrics,
            audit_flags=audit_flags,
        )
