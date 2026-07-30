"""Directed relational composition of Model A multiscale pooling outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn

from gnn_siamese.models.multiscale_pooling_a import (
    ModelAMultiscalePoolingOutput,
    ScalePoolResult,
)
from gnn_siamese.models.pair_fusion_a import MLPPairFusion


SCALE_ORDER_A: Final[tuple[str, ...]] = ("mutation", "local", "domain", "global")


@dataclass(frozen=True)
class ModelAMultiscaleRelationalOutput:
    z_delta_mutation: Tensor | None
    z_delta_local: Tensor | None
    z_delta_domain: Tensor | None
    z_delta_global: Tensor | None
    h_pair_delta: Tensor
    z_delta_pair: Tensor
    active_scales: tuple[str, ...]
    scale_order: tuple[str, ...]
    scale_dimensions: dict[str, int]
    pair_dimension: int
    scale_valid_masks: dict[str, Tensor]
    scale_counts: dict[str, dict[str, Tensor]]
    pair_valid_mask: Tensor
    pair_fusion_mode: str


def build_scale_relational(
    h_k_MUT: Tensor,
    h_k_WT: Tensor,
    h_k_delta: Tensor,
    *,
    scale: str,
) -> Tensor:
    """Build ``[MUT, WT, MUT-WT, abs(MUT-WT), delta]`` exactly."""

    if scale not in SCALE_ORDER_A:
        raise ValueError(f"unknown scale {scale!r}; expected one of {SCALE_ORDER_A}.")
    tensors = {
        f"h_{scale}_MUT": h_k_MUT,
        f"h_{scale}_WT": h_k_WT,
        f"h_{scale}_delta": h_k_delta,
    }
    for name, value in tensors.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.ndim != 2:
            raise ValueError(
                f"{name} must be two-dimensional [B, D], got shape {tuple(value.shape)}."
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf.")
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    if len({value.shape[0] for value in tensors.values()}) != 1:
        raise ValueError(f"{scale} pooled batch sizes must match, got {shapes}.")
    if len({value.shape[1] for value in tensors.values()}) != 1:
        raise ValueError(f"{scale} pooled embedding dimensions must match, got {shapes}.")
    devices = {value.device for value in tensors.values()}
    if len(devices) != 1:
        raise ValueError(f"{scale} pooled tensors must share one device, got {devices}.")
    dtypes = {value.dtype for value in tensors.values()}
    if len(dtypes) != 1:
        raise TypeError(f"{scale} pooled tensors must share one dtype, got {dtypes}.")

    directed_delta = h_k_MUT - h_k_WT
    output = torch.cat(
        (h_k_MUT, h_k_WT, directed_delta, directed_delta.abs(), h_k_delta),
        dim=-1,
    )
    expected = 5 * h_k_MUT.shape[1]
    if output.shape[1] != expected:
        raise RuntimeError(
            f"z_delta_{scale} must have dimension 5 * D = {expected}, "
            f"got {output.shape[1]}."
        )
    return output


class ModelAMultiscaleRelational(nn.Module):
    """Compose active A3 scales canonically and apply final pair fusion."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        active_scales: tuple[str, ...] = ("mutation", "local", "global"),
        pair_fusion_enabled: bool = True,
        pair_fusion_hidden_dim: int = 256,
        pair_fusion_output_dim: int | None = 128,
        pair_fusion_activation: str = "relu",
        pair_fusion_dropout: float = 0.1,
        pair_fusion_disabled_policy: str = "identity",
        pair_fusion_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        if not isinstance(active_scales, tuple):
            raise TypeError("active_scales must be a tuple of scale names.")
        if not active_scales:
            raise ValueError("active_scales must contain at least one scale.")
        if len(active_scales) != len(set(active_scales)):
            raise ValueError("active_scales must not contain duplicates.")
        unknown = set(active_scales) - set(SCALE_ORDER_A)
        if unknown:
            raise ValueError(f"active_scales contains unknown scales: {sorted(unknown)}.")

        self.embedding_dim = embedding_dim
        self.active_scales = tuple(
            scale for scale in SCALE_ORDER_A if scale in active_scales
        )
        derived_input_dim = len(self.active_scales) * 5 * embedding_dim
        if pair_fusion_input_dim is not None and pair_fusion_input_dim != derived_input_dim:
            raise ValueError(
                "pair_fusion input_dim is incompatible with active scales: "
                f"derived {derived_input_dim}, got {pair_fusion_input_dim}."
            )
        self.pair_fusion = MLPPairFusion(
            input_dim=derived_input_dim,
            hidden_dim=pair_fusion_hidden_dim,
            output_dim=pair_fusion_output_dim,
            activation=pair_fusion_activation,
            dropout=pair_fusion_dropout,
            enabled=pair_fusion_enabled,
            disabled_policy=pair_fusion_disabled_policy,
        )

    @staticmethod
    def _result(
        pooling: ModelAMultiscalePoolingOutput, branch: str, scale: str
    ) -> ScalePoolResult | None:
        branch_output = getattr(pooling, branch)
        return getattr(branch_output, "global_" if scale == "global" else scale)

    @staticmethod
    def _validate_metadata(
        results: dict[str, ScalePoolResult],
        *,
        scale: str,
        batch_size: int,
        device: torch.device,
    ) -> None:
        for branch, result in results.items():
            if result.valid_mask.shape != (batch_size,):
                raise ValueError(
                    f"{scale} {branch} valid_mask must have shape [{batch_size}], "
                    f"got {tuple(result.valid_mask.shape)}."
                )
            if result.valid_mask.dtype != torch.bool:
                raise TypeError(f"{scale} {branch} valid_mask must use torch.bool.")
            if result.counts.shape != (batch_size,):
                raise ValueError(
                    f"{scale} {branch} counts must have shape [{batch_size}], "
                    f"got {tuple(result.counts.shape)}."
                )
            if result.counts.dtype != torch.long:
                raise TypeError(f"{scale} {branch} counts must use torch.long.")
            if torch.any(result.counts < 0):
                raise ValueError(f"{scale} {branch} counts must be non-negative.")
            if result.valid_mask.device != device or result.counts.device != device:
                raise ValueError(
                    f"{scale} {branch} masks, counts, and values must share device {device}."
                )
            if not torch.equal(result.valid_mask, result.counts > 0):
                raise ValueError(
                    f"{scale} {branch} valid_mask must equal counts > 0."
                )

    def forward(
        self, pooling: ModelAMultiscalePoolingOutput
    ) -> ModelAMultiscaleRelationalOutput:
        if not isinstance(pooling, ModelAMultiscalePoolingOutput):
            raise TypeError("pooling must be a ModelAMultiscalePoolingOutput.")

        z_by_scale: dict[str, Tensor] = {}
        valid_by_scale: dict[str, Tensor] = {}
        counts_by_scale: dict[str, dict[str, Tensor]] = {}
        dimensions: dict[str, int] = {}
        for scale in self.active_scales:
            raw = {
                branch: self._result(pooling, branch, scale)
                for branch in ("MUT", "WT", "delta")
            }
            missing = [branch for branch, result in raw.items() if result is None]
            if missing:
                raise ValueError(
                    f"active scale {scale!r} is unavailable for branches {missing}; "
                    "A3 must mark it active and provide real pooling outputs."
                )
            results = {branch: result for branch, result in raw.items() if result is not None}
            values = [results[branch].values for branch in ("MUT", "WT", "delta")]
            z_scale = build_scale_relational(*values, scale=scale)
            if values[0].shape[1] != self.embedding_dim:
                raise ValueError(
                    f"{scale} embedding dimension must equal configured "
                    f"embedding_dim={self.embedding_dim}, got {values[0].shape[1]}."
                )
            self._validate_metadata(
                results,
                scale=scale,
                batch_size=z_scale.shape[0],
                device=z_scale.device,
            )
            z_by_scale[scale] = z_scale
            dimensions[scale] = z_scale.shape[1]
            counts_by_scale[scale] = {
                branch: results[branch].counts for branch in ("MUT", "WT", "delta")
            }
            valid_by_scale[scale] = torch.stack(
                [results[branch].valid_mask for branch in ("MUT", "WT", "delta")]
            ).all(dim=0)

        h_pair_delta = torch.cat(
            [z_by_scale[scale] for scale in self.active_scales], dim=-1
        )
        pair_valid_mask = torch.stack(
            [valid_by_scale[scale] for scale in self.active_scales]
        ).all(dim=0)
        z_delta_pair = self.pair_fusion(h_pair_delta)
        return ModelAMultiscaleRelationalOutput(
            z_delta_mutation=z_by_scale.get("mutation"),
            z_delta_local=z_by_scale.get("local"),
            z_delta_domain=z_by_scale.get("domain"),
            z_delta_global=z_by_scale.get("global"),
            h_pair_delta=h_pair_delta,
            z_delta_pair=z_delta_pair,
            active_scales=self.active_scales,
            scale_order=SCALE_ORDER_A,
            scale_dimensions=dimensions,
            pair_dimension=h_pair_delta.shape[1],
            scale_valid_masks=valid_by_scale,
            scale_counts=counts_by_scale,
            pair_valid_mask=pair_valid_mask,
            pair_fusion_mode=self.pair_fusion.mode,
        )
