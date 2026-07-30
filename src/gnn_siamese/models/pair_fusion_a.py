"""Final multiscale pair fusion for one-view Model A."""

from __future__ import annotations

import torch
from torch import Tensor, nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class MLPPairFusion(nn.Module):
    """Map ``h_pair_delta`` to ``z_delta_pair`` or apply explicit identity."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        enabled: bool = True,
        disabled_policy: str = "identity",
    ) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if output_dim is not None and (
            isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0
        ):
            raise ValueError("output_dim must be None or a positive integer.")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}."
            )
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a real number in [0, 1).")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if disabled_policy != "identity":
            raise ValueError(
                "disabled_policy must be 'identity'; "
                f"got {disabled_policy!r}."
            )
        if not enabled and output_dim is not None and output_dim != input_dim:
            raise ValueError(
                "disabled identity fusion requires output_dim to be None or equal "
                f"to input_dim={input_dim}, got {output_dim}."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim if not enabled else int(output_dim or input_dim)
        self.activation_name = activation
        self.dropout_probability = float(dropout)
        self.enabled = enabled
        self.disabled_policy = disabled_policy
        self.mode = "mlp" if enabled else "identity"
        self.network: nn.Module
        if enabled:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                _ACTIVATIONS[activation](),
                nn.Dropout(self.dropout_probability),
                nn.Linear(hidden_dim, self.output_dim),
            )
        else:
            self.network = nn.Identity()

    def forward(self, h_pair_delta: Tensor) -> Tensor:
        if not isinstance(h_pair_delta, Tensor):
            raise TypeError("h_pair_delta must be a torch.Tensor.")
        if h_pair_delta.ndim != 2:
            raise ValueError(
                "h_pair_delta must be two-dimensional [B, pair_dimension], got "
                f"shape {tuple(h_pair_delta.shape)}."
            )
        if h_pair_delta.shape[1] != self.input_dim:
            raise ValueError(
                f"h_pair_delta dimension must equal input_dim={self.input_dim}, "
                f"got {h_pair_delta.shape[1]}."
            )
        if not h_pair_delta.is_floating_point():
            raise TypeError("h_pair_delta must use a floating-point dtype.")
        if not torch.isfinite(h_pair_delta).all():
            raise ValueError("h_pair_delta contains NaN or Inf.")
        return self.network(h_pair_delta)
