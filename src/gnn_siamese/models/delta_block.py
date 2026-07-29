"""Learnable directed node deltas for globally aligned Mutant--WT nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


@dataclass(frozen=True)
class NodeDeltaOutput:
    """Node deltas and unchanged A1 batch segmentation metadata."""

    H_delta: Tensor
    aligned_pair_batch: Tensor | None
    alignment_ptr: Tensor | None


def build_node_delta_features(H_MUT_aligned: Tensor, H_WT_aligned: Tensor) -> Tensor:
    """Build ``[MUT, WT, MUT-WT, abs(MUT-WT)]`` features in that exact order."""

    if not isinstance(H_MUT_aligned, Tensor):
        raise TypeError("H_MUT_aligned must be a torch.Tensor.")
    if not isinstance(H_WT_aligned, Tensor):
        raise TypeError("H_WT_aligned must be a torch.Tensor.")
    if H_MUT_aligned.ndim != 2:
        raise ValueError(
            f"H_MUT_aligned must be two-dimensional [A, D], got shape "
            f"{tuple(H_MUT_aligned.shape)}."
        )
    if H_WT_aligned.ndim != 2:
        raise ValueError(
            f"H_WT_aligned must be two-dimensional [A, D], got shape "
            f"{tuple(H_WT_aligned.shape)}."
        )
    if H_MUT_aligned.shape != H_WT_aligned.shape:
        raise ValueError(
            "H_MUT_aligned and H_WT_aligned must have identical shapes, got "
            f"{tuple(H_MUT_aligned.shape)} and {tuple(H_WT_aligned.shape)}."
        )
    if H_MUT_aligned.device != H_WT_aligned.device:
        raise ValueError(
            "H_MUT_aligned and H_WT_aligned must be on the same device, got "
            f"{H_MUT_aligned.device} and {H_WT_aligned.device}."
        )
    if H_MUT_aligned.dtype != H_WT_aligned.dtype:
        raise TypeError(
            "H_MUT_aligned and H_WT_aligned must have the same dtype, got "
            f"{H_MUT_aligned.dtype} and {H_WT_aligned.dtype}."
        )
    if not (H_MUT_aligned.is_floating_point() and H_WT_aligned.is_floating_point()):
        raise TypeError("H_MUT_aligned and H_WT_aligned must use a floating-point dtype.")
    if not torch.isfinite(H_MUT_aligned).all():
        raise ValueError("H_MUT_aligned contains NaN or Inf.")
    if not torch.isfinite(H_WT_aligned).all():
        raise ValueError("H_WT_aligned contains NaN or Inf.")

    directed_delta = H_MUT_aligned - H_WT_aligned
    return torch.cat(
        (
            H_MUT_aligned,
            H_WT_aligned,
            directed_delta,
            directed_delta.abs(),
        ),
        dim=-1,
    )


class NodeDeltaBlock(nn.Module):
    """Transform aligned node embeddings into directed learned node deltas."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if activation not in _ACTIVATIONS:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"activation must be one of {{{choices}}}, got {activation!r}.")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a real number in [0, 1).")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_probability = float(dropout)
        self.network = nn.Sequential(
            nn.Linear(4 * input_dim, hidden_dim),
            _ACTIVATIONS[activation](),
            nn.Dropout(self.dropout_probability),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _validate_index(index: Tensor, *, name: str, node_count: int, device: torch.device) -> None:
        if not isinstance(index, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if index.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {tuple(index.shape)}.")
        if index.dtype != torch.long:
            raise TypeError(f"{name} must use torch.long, got {index.dtype}.")
        if index.device != device:
            raise ValueError(
                f"{name} must be on the embeddings device {device}, got {index.device}."
            )
        if torch.any(index < 0):
            raise IndexError(f"{name} contains a negative index.")
        if index.numel() and torch.any(index >= node_count):
            maximum = int(index.max().item())
            raise IndexError(
                f"{name} contains index {maximum} outside [0, {node_count})."
            )

    @staticmethod
    def _validate_aligned_pair_batch(
        aligned_pair_batch: Tensor | None,
        *,
        alignment_count: int,
        device: torch.device,
    ) -> None:
        if aligned_pair_batch is None:
            return
        if not isinstance(aligned_pair_batch, Tensor):
            raise TypeError("aligned_pair_batch must be a torch.Tensor when provided.")
        if aligned_pair_batch.ndim != 1:
            raise ValueError(
                "aligned_pair_batch must be one-dimensional, got shape "
                f"{tuple(aligned_pair_batch.shape)}."
            )
        if aligned_pair_batch.dtype != torch.long:
            raise TypeError(
                f"aligned_pair_batch must use torch.long, got {aligned_pair_batch.dtype}."
            )
        if aligned_pair_batch.device != device:
            raise ValueError(
                "aligned_pair_batch must be on the embeddings device "
                f"{device}, got {aligned_pair_batch.device}."
            )
        if aligned_pair_batch.numel() != alignment_count:
            raise ValueError(
                f"aligned_pair_batch must have length A={alignment_count}, got "
                f"{aligned_pair_batch.numel()}."
            )
        if torch.any(aligned_pair_batch < 0):
            raise ValueError("aligned_pair_batch contains a negative pair index.")

    @staticmethod
    def _validate_alignment_ptr(
        alignment_ptr: Tensor | None,
        *,
        aligned_pair_batch: Tensor | None,
        alignment_count: int,
        device: torch.device,
    ) -> None:
        if alignment_ptr is None:
            return
        if not isinstance(alignment_ptr, Tensor):
            raise TypeError("alignment_ptr must be a torch.Tensor when provided.")
        if alignment_ptr.ndim != 1:
            raise ValueError(
                f"alignment_ptr must be one-dimensional, got shape {tuple(alignment_ptr.shape)}."
            )
        if alignment_ptr.dtype != torch.long:
            raise TypeError(f"alignment_ptr must use torch.long, got {alignment_ptr.dtype}.")
        if alignment_ptr.device != device:
            raise ValueError(
                f"alignment_ptr must be on the embeddings device {device}, got "
                f"{alignment_ptr.device}."
            )
        if alignment_ptr.numel() == 0:
            raise ValueError("alignment_ptr must contain at least its initial zero.")
        if int(alignment_ptr[0].item()) != 0:
            raise ValueError("alignment_ptr must start at zero.")
        if torch.any(alignment_ptr[1:] < alignment_ptr[:-1]):
            raise ValueError("alignment_ptr must be monotonically non-decreasing.")
        if int(alignment_ptr[-1].item()) != alignment_count:
            raise ValueError(
                f"alignment_ptr must end at A={alignment_count}, got "
                f"{int(alignment_ptr[-1].item())}."
            )

        if aligned_pair_batch is not None:
            pair_count = alignment_ptr.numel() - 1
            if aligned_pair_batch.numel() and int(aligned_pair_batch.max().item()) >= pair_count:
                raise ValueError(
                    "aligned_pair_batch contains a pair index incompatible with alignment_ptr."
                )
            for pair_index in range(pair_count):
                start = int(alignment_ptr[pair_index].item())
                end = int(alignment_ptr[pair_index + 1].item())
                if start != end and not torch.all(aligned_pair_batch[start:end] == pair_index):
                    raise ValueError(
                        "aligned_pair_batch is incompatible with alignment_ptr in segment "
                        f"{pair_index}."
                    )

    def forward(
        self,
        H_MUT: Tensor,
        H_WT: Tensor,
        mut_aligned_index: Tensor,
        wt_aligned_index: Tensor,
        aligned_pair_batch: Tensor | None = None,
        alignment_ptr: Tensor | None = None,
    ) -> NodeDeltaOutput:
        """Gather A1 correspondences and return one learned row per aligned node."""

        if not isinstance(H_MUT, Tensor):
            raise TypeError("H_MUT must be a torch.Tensor.")
        if not isinstance(H_WT, Tensor):
            raise TypeError("H_WT must be a torch.Tensor.")
        if H_MUT.ndim != 2:
            raise ValueError(f"H_MUT must be two-dimensional [N_MUT, D], got {tuple(H_MUT.shape)}.")
        if H_WT.ndim != 2:
            raise ValueError(f"H_WT must be two-dimensional [N_WT, D], got {tuple(H_WT.shape)}.")
        if H_MUT.shape[1] != H_WT.shape[1]:
            raise ValueError(
                "H_MUT and H_WT latent dimensions must match, got "
                f"{H_MUT.shape[1]} and {H_WT.shape[1]}."
            )
        if H_MUT.shape[1] != self.input_dim:
            raise ValueError(
                f"H_MUT and H_WT latent dimension must equal input_dim={self.input_dim}, "
                f"got {H_MUT.shape[1]}."
            )
        if H_MUT.device != H_WT.device:
            raise ValueError(
                f"H_MUT and H_WT must be on the same device, got {H_MUT.device} and {H_WT.device}."
            )
        if H_MUT.dtype != H_WT.dtype:
            raise TypeError(
                f"H_MUT and H_WT must have the same dtype, got {H_MUT.dtype} and {H_WT.dtype}."
            )
        if not (H_MUT.is_floating_point() and H_WT.is_floating_point()):
            raise TypeError("H_MUT and H_WT must use a floating-point dtype.")
        if not torch.isfinite(H_MUT).all():
            raise ValueError("H_MUT contains NaN or Inf.")
        if not torch.isfinite(H_WT).all():
            raise ValueError("H_WT contains NaN or Inf.")

        self._validate_index(
            mut_aligned_index,
            name="mut_aligned_index",
            node_count=H_MUT.shape[0],
            device=H_MUT.device,
        )
        self._validate_index(
            wt_aligned_index,
            name="wt_aligned_index",
            node_count=H_WT.shape[0],
            device=H_MUT.device,
        )
        if mut_aligned_index.numel() != wt_aligned_index.numel():
            raise ValueError(
                "mut_aligned_index and wt_aligned_index must have the same length, got "
                f"{mut_aligned_index.numel()} and {wt_aligned_index.numel()}."
            )

        alignment_count = mut_aligned_index.numel()
        self._validate_aligned_pair_batch(
            aligned_pair_batch,
            alignment_count=alignment_count,
            device=H_MUT.device,
        )
        self._validate_alignment_ptr(
            alignment_ptr,
            aligned_pair_batch=aligned_pair_batch,
            alignment_count=alignment_count,
            device=H_MUT.device,
        )

        features = build_node_delta_features(
            H_MUT.index_select(0, mut_aligned_index),
            H_WT.index_select(0, wt_aligned_index),
        )
        if alignment_count == 0:
            H_delta = H_MUT.new_empty((0, self.output_dim))
        else:
            H_delta = self.network(features)

        return NodeDeltaOutput(
            H_delta=H_delta,
            aligned_pair_batch=aligned_pair_batch,
            alignment_ptr=alignment_ptr,
        )
