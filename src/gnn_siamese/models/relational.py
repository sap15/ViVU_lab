"""Relational Mutant-WT representations built from shared encoder outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class RDelta(nn.Module):
    """Deterministic relational baseline without trainable parameters."""

    def forward(self, h_mut: Tensor, h_wt: Tensor) -> Tensor:
        if h_mut.shape != h_wt.shape:
            raise ValueError("h_mut and h_wt must have the same shape.")
        delta = h_mut - h_wt
        return torch.cat((h_mut, h_wt, delta, delta.abs(), h_mut * h_wt), dim=-1)


class MLPDelta(nn.Module):
    """Optional learnable transformation applied on top of r_delta."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("MLPDelta dimensions must be positive.")
        if num_layers <= 0:
            raise ValueError("MLPDelta num_layers must be positive.")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, r_delta: Tensor) -> Tensor:
        return self.network(r_delta)


@dataclass(frozen=True)
class RelationalOutput:
    """Structured relational outputs derived from h_mut and h_wt."""

    r_delta: Tensor
    severity: Tensor
    mechanism_direction: Tensor
    z_delta: Tensor | None
    z_delta_status: str

    @property
    def z_delta_is_validated(self) -> bool:
        return self.z_delta is not None and self.z_delta_status == "validated"

    def to_dict(self) -> dict[str, Tensor | str]:
        output: dict[str, Tensor | str] = {
            "r_delta": self.r_delta,
            "severity": self.severity,
            "mechanism_direction": self.mechanism_direction,
        }
        if self.z_delta is not None:
            output["z_delta"] = self.z_delta
            output["z_delta_status"] = self.z_delta_status
        return output


class RelationalRepresentation(nn.Module):
    """Compute deterministic and optional learned Mutant-WT representations."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        severity_eps: float = 1.0e-8,
        mlp_delta_enabled: bool = False,
        mlp_delta_hidden_dim: int | None = None,
        mlp_delta_output_dim: int | None = None,
        mlp_delta_num_layers: int = 2,
        mlp_delta_dropout: float = 0.0,
        z_delta_validated: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if severity_eps <= 0:
            raise ValueError("severity_eps must be positive.")

        self.embedding_dim = int(embedding_dim)
        self.severity_eps = float(severity_eps)
        self.r_delta = RDelta()
        self.z_delta_validated = bool(z_delta_validated)

        if mlp_delta_enabled:
            hidden_dim = mlp_delta_hidden_dim or 2 * self.embedding_dim
            output_dim = mlp_delta_output_dim or self.embedding_dim
            self.mlp_delta: MLPDelta | None = MLPDelta(
                input_dim=5 * self.embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=mlp_delta_num_layers,
                dropout=mlp_delta_dropout,
            )
        else:
            self.mlp_delta = None

    @property
    def mlp_delta_enabled(self) -> bool:
        return self.mlp_delta is not None

    def forward(self, h_mut: Tensor, h_wt: Tensor) -> RelationalOutput:
        if h_mut.ndim != 2 or h_wt.ndim != 2:
            raise ValueError("h_mut and h_wt must be rank-2 tensors shaped [batch, dim].")
        if h_mut.shape != h_wt.shape:
            raise ValueError("h_mut and h_wt must have the same shape.")
        if h_mut.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got last dimension {h_mut.shape[-1]}."
            )

        delta = h_mut - h_wt
        r_delta = self.r_delta(h_mut, h_wt)
        severity = torch.linalg.vector_norm(delta, ord=2, dim=-1)
        safe_denominator = severity.clamp_min(self.severity_eps).unsqueeze(-1)
        mechanism_direction = delta / safe_denominator

        z_delta = self.mlp_delta(r_delta) if self.mlp_delta is not None else None
        if z_delta is None:
            z_delta_status = "inactive"
        elif self.z_delta_validated:
            z_delta_status = "validated"
        else:
            z_delta_status = "unvalidated"

        return RelationalOutput(
            r_delta=r_delta,
            severity=severity,
            mechanism_direction=mechanism_direction,
            z_delta=z_delta,
            z_delta_status=z_delta_status,
        )
