"""Projection heads for individual and relational contrastive spaces."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ProjectionHeadConfig:
    """Configuration shared by projection heads."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float = 0.0
    use_layer_norm: bool = True
    normalize_output: bool = False


class _BaseProjectionHead(nn.Module):
    """Shared implementation with semantic validation per projection role."""

    def __init__(
        self,
        *,
        config: ProjectionHeadConfig,
        semantic_role: str,
        allowed_input_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        if config.input_dim <= 0 or config.hidden_dim <= 0 or config.output_dim <= 0:
            raise ValueError("Projection head dimensions must be positive.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("Projection head dropout must be in the range [0, 1).")
        if not allowed_input_names:
            raise ValueError("Projection head must declare at least one allowed input name.")

        self.input_dim = int(config.input_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.output_dim = int(config.output_dim)
        self.dropout = float(config.dropout)
        self.use_layer_norm = bool(config.use_layer_norm)
        self.normalize_output = bool(config.normalize_output)
        self.semantic_role = semantic_role
        self.allowed_input_names = tuple(allowed_input_names)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_norm = nn.LayerNorm(self.hidden_dim) if self.use_layer_norm else nn.Identity()
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def _validate_input(self, x: Tensor, input_name: str | None) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"{self.semantic_role} projection expects a rank-2 tensor shaped [batch, dim]."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"{self.semantic_role} projection expects last dimension {self.input_dim}, "
                f"got {x.shape[-1]}."
            )
        if input_name is not None and input_name not in self.allowed_input_names:
            allowed = ", ".join(self.allowed_input_names)
            raise ValueError(
                f"{self.semantic_role} projection received semantic input '{input_name}', "
                f"but only [{allowed}] are allowed."
            )

    def forward(self, x: Tensor, *, input_name: str | None = None) -> Tensor:
        self._validate_input(x, input_name=input_name)

        hidden = self.input_layer(x)
        hidden = self.hidden_norm(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout_layer(hidden)
        output = self.output_layer(hidden)
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=-1)
        return output


class InstanceProjectionHead(_BaseProjectionHead):
    """Projection head for individual mutant embeddings."""

    def __init__(self, *, config: ProjectionHeadConfig) -> None:
        super().__init__(
            config=config,
            semantic_role="instance",
            allowed_input_names=("h_encoder_mut", "h_mut", "h_encoder_view"),
        )


class PairProjectionHead(_BaseProjectionHead):
    """Projection head for paired Mutant-WT relational embeddings."""

    def __init__(self, *, config: ProjectionHeadConfig) -> None:
        super().__init__(
            config=config,
            semantic_role="pair",
            allowed_input_names=("r_delta", "z_delta"),
        )
