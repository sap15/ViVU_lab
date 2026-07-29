"""Parameter-free multiscale pooling for Model A node representations.

The module only aggregates already-computed node representations.  It does not
perform node alignment, infer spatial neighborhoods, or fuse scale/branch
outputs.  Empty segments contain a neutral zero row and are unambiguously
identified by ``valid_mask=False`` and ``counts=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn


_SCALE_ORDER: Final[tuple[str, ...]] = ("mutation", "local", "domain", "global")
_POOLING_MODES: Final[frozenset[str]] = frozenset({"mean", "sum"})


@dataclass(frozen=True)
class ScalePoolResult:
    """One scale pooled to ``values[B, D]`` with explicit availability."""

    values: Tensor
    valid_mask: Tensor
    counts: Tensor


@dataclass(frozen=True)
class BranchMultiscalePooling:
    """Independent scale summaries for one of MUT, WT, or delta."""

    mutation: ScalePoolResult | None
    local: ScalePoolResult | None
    domain: ScalePoolResult | None
    global_: ScalePoolResult | None


@dataclass(frozen=True)
class ModelAMultiscalePoolingOutput:
    """Unfused, semantically separated Model A multiscale summaries."""

    MUT: BranchMultiscalePooling
    WT: BranchMultiscalePooling
    delta: BranchMultiscalePooling


def _validate_num_pairs(num_pairs: int) -> None:
    if isinstance(num_pairs, bool) or not isinstance(num_pairs, int):
        raise TypeError("num_pairs must be an integer.")
    if num_pairs < 0:
        raise ValueError("num_pairs must be non-negative.")


def _validate_embeddings(embeddings: Tensor, *, name: str) -> None:
    if not isinstance(embeddings, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if embeddings.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional [N, D], got shape {tuple(embeddings.shape)}."
        )
    if embeddings.shape[1] <= 0:
        raise ValueError(f"{name} must have a positive feature dimension D.")
    if not embeddings.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _validate_batch(
    batch: Tensor,
    *,
    name: str,
    row_count: int,
    num_pairs: int,
    device: torch.device,
) -> None:
    if not isinstance(batch, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if batch.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(batch.shape)}.")
    if batch.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long, got {batch.dtype}.")
    if batch.numel() != row_count:
        raise ValueError(f"{name} must have length {row_count}, got {batch.numel()}.")
    if batch.device != device:
        raise ValueError(f"{name} must be on device {device}, got {batch.device}.")
    if torch.any(batch < 0):
        raise ValueError(f"{name} contains a negative pair index.")
    if batch.numel() and int(batch.max().item()) >= num_pairs:
        raise ValueError(
            f"{name} contains pair index {int(batch.max().item())}, "
            f"which is not smaller than num_pairs={num_pairs}."
        )


def _validate_mask(
    mask: Tensor | None,
    *,
    name: str,
    row_count: int,
    device: torch.device,
    required: bool,
) -> Tensor:
    if mask is None:
        if required:
            raise ValueError(f"{name} is required when its scale is enabled.")
        raise AssertionError("Internal error: optional absent masks must not be validated.")
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(mask.shape)}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool, got {mask.dtype}.")
    if mask.numel() != row_count:
        raise ValueError(f"{name} must have length {row_count}, got {mask.numel()}.")
    if mask.device != device:
        raise ValueError(f"{name} must be on device {device}, got {mask.device}.")
    return mask


def _validate_index(
    index: Tensor,
    *,
    name: str,
    row_count: int,
    device: torch.device,
    unique: bool = True,
) -> None:
    if not isinstance(index, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if index.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(index.shape)}.")
    if index.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long, got {index.dtype}.")
    if index.device != device:
        raise ValueError(f"{name} must be on device {device}, got {index.device}.")
    if torch.any(index < 0):
        raise IndexError(f"{name} contains a negative index.")
    if index.numel() and torch.any(index >= row_count):
        raise IndexError(
            f"{name} contains index {int(index.max().item())} outside [0, {row_count})."
        )
    if unique and index.numel() != torch.unique(index).numel():
        raise ValueError(f"{name} contains duplicate indices.")


def _validate_ptr(
    ptr: Tensor,
    *,
    name: str,
    total: int,
    num_pairs: int,
    device: torch.device,
) -> None:
    if not isinstance(ptr, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if ptr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(ptr.shape)}.")
    if ptr.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long, got {ptr.dtype}.")
    if ptr.device != device:
        raise ValueError(f"{name} must be on device {device}, got {ptr.device}.")
    if ptr.numel() != num_pairs + 1:
        raise ValueError(f"{name} must have length num_pairs + 1 = {num_pairs + 1}.")
    if int(ptr[0].item()) != 0:
        raise ValueError(f"{name} must start at zero.")
    if torch.any(ptr[1:] < ptr[:-1]):
        raise ValueError(f"{name} must be monotonically non-decreasing.")
    if int(ptr[-1].item()) != total:
        raise ValueError(f"{name} must end at {total}, got {int(ptr[-1].item())}.")


def validate_delta_segmentation(
    aligned_pair_batch: Tensor,
    alignment_ptr: Tensor | None,
    *,
    alignment_count: int,
    num_pairs: int,
    device: torch.device,
) -> None:
    """Validate A1/A2 row-to-pair metadata, including optional pointer segments."""

    _validate_batch(
        aligned_pair_batch,
        name="aligned_pair_batch",
        row_count=alignment_count,
        num_pairs=num_pairs,
        device=device,
    )
    if alignment_ptr is None:
        return
    _validate_ptr(
        alignment_ptr,
        name="alignment_ptr",
        total=alignment_count,
        num_pairs=num_pairs,
        device=device,
    )
    for pair_index in range(num_pairs):
        start = int(alignment_ptr[pair_index].item())
        end = int(alignment_ptr[pair_index + 1].item())
        if start != end and not torch.all(aligned_pair_batch[start:end] == pair_index):
            raise ValueError(
                "aligned_pair_batch is incompatible with alignment_ptr in segment "
                f"{pair_index}."
            )


def indices_to_mask(
    index: Tensor,
    *,
    row_count: int,
    name: str = "index",
) -> Tensor:
    """Convert explicit unique row indices to a boolean selection mask."""

    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer.")
    if not isinstance(index, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    _validate_index(
        index,
        name=name,
        row_count=row_count,
        device=index.device,
        unique=True,
    )
    mask = torch.zeros(row_count, dtype=torch.bool, device=index.device)
    mask[index] = True
    return mask


def aligned_selection_mask(
    aligned_node_index: Tensor,
    alignment_ptr: Tensor,
    selected_node_index: Tensor,
    selection_ptr: Tensor,
    *,
    num_pairs: int,
) -> Tensor:
    """Map per-pair node selections exactly onto aligned rows.

    Every selected node must occur exactly once in the corresponding global
    alignment segment.  This is the exact bridge from A1 local indices (or a
    mutation index) to rows of ``H_delta``; first-match/row-order heuristics are
    deliberately rejected.
    """

    _validate_num_pairs(num_pairs)
    if not isinstance(aligned_node_index, Tensor):
        raise TypeError("aligned_node_index must be a torch.Tensor.")
    device = aligned_node_index.device
    _validate_index(
        aligned_node_index,
        name="aligned_node_index",
        row_count=(int(aligned_node_index.max().item()) + 1) if aligned_node_index.numel() else 0,
        device=device,
        unique=False,
    )
    _validate_index(
        selected_node_index,
        name="selected_node_index",
        row_count=(int(selected_node_index.max().item()) + 1) if selected_node_index.numel() else 0,
        device=device,
        unique=True,
    )
    _validate_ptr(
        alignment_ptr,
        name="alignment_ptr",
        total=aligned_node_index.numel(),
        num_pairs=num_pairs,
        device=device,
    )
    _validate_ptr(
        selection_ptr,
        name="selection_ptr",
        total=selected_node_index.numel(),
        num_pairs=num_pairs,
        device=device,
    )

    result = torch.zeros(aligned_node_index.numel(), dtype=torch.bool, device=device)
    for pair_index in range(num_pairs):
        aligned_start = int(alignment_ptr[pair_index].item())
        aligned_end = int(alignment_ptr[pair_index + 1].item())
        selection_start = int(selection_ptr[pair_index].item())
        selection_end = int(selection_ptr[pair_index + 1].item())
        aligned_segment = aligned_node_index[aligned_start:aligned_end]
        selected_segment = selected_node_index[selection_start:selection_end]
        if not selected_segment.numel():
            continue

        sorted_aligned, permutation = torch.sort(aligned_segment)
        positions = torch.searchsorted(sorted_aligned, selected_segment, side="left")
        right_positions = torch.searchsorted(
            sorted_aligned, selected_segment, side="right"
        )
        match_counts = right_positions - positions
        valid = match_counts == 1
        if not torch.all(valid):
            invalid_offset = int(torch.nonzero(~valid, as_tuple=False)[0].item())
            invalid_value = selected_segment[invalid_offset]
            match_count = int(match_counts[invalid_offset].item())
            raise ValueError(
                f"selected_node_index value {int(invalid_value.item())} in pair "
                f"{pair_index} must occur exactly once in aligned_node_index; "
                f"found {match_count}."
            )
        result[aligned_start + permutation[positions]] = True
    return result


def segmented_pool(
    embeddings: Tensor,
    batch: Tensor,
    selection: Tensor,
    *,
    num_pairs: int,
    mode: str = "mean",
    embeddings_name: str = "embeddings",
    batch_name: str = "batch",
    selection_name: str = "selection",
) -> ScalePoolResult:
    """Pool selected rows by pair while preserving dtype, device, and gradients."""

    _validate_num_pairs(num_pairs)
    _validate_embeddings(embeddings, name=embeddings_name)
    _validate_batch(
        batch,
        name=batch_name,
        row_count=embeddings.shape[0],
        num_pairs=num_pairs,
        device=embeddings.device,
    )
    mask = _validate_mask(
        selection,
        name=selection_name,
        row_count=embeddings.shape[0],
        device=embeddings.device,
        required=True,
    )
    if mode not in _POOLING_MODES:
        raise ValueError(f"mode must be one of {sorted(_POOLING_MODES)}, got {mode!r}.")

    values = embeddings.new_zeros((num_pairs, embeddings.shape[1]))
    selected_batch = batch[mask]
    if selected_batch.numel():
        values.index_add_(0, selected_batch, embeddings[mask])
    counts = torch.bincount(selected_batch, minlength=num_pairs)
    valid_mask = counts > 0
    if mode == "mean":
        values = values / counts.clamp_min(1).to(dtype=embeddings.dtype).unsqueeze(1)
    return ScalePoolResult(values=values, valid_mask=valid_mask, counts=counts)


class ModelAMultiscalePooling(nn.Module):
    """Pool MUT, WT, and delta independently at explicitly selected scales."""

    def __init__(
        self,
        *,
        enabled_scales: tuple[str, ...] = ("mutation", "local", "global"),
        mode: str = "mean",
        allow_missing_mutation: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(enabled_scales, tuple):
            raise TypeError("enabled_scales must be a tuple of scale names.")
        if not enabled_scales:
            raise ValueError("enabled_scales must contain at least one scale.")
        if len(enabled_scales) != len(set(enabled_scales)):
            raise ValueError("enabled_scales must not contain duplicates.")
        unknown = set(enabled_scales) - set(_SCALE_ORDER)
        if unknown:
            raise ValueError(f"enabled_scales contains unknown scales: {sorted(unknown)}.")
        if mode not in _POOLING_MODES:
            raise ValueError(f"mode must be one of {sorted(_POOLING_MODES)}, got {mode!r}.")
        if not isinstance(allow_missing_mutation, bool):
            raise TypeError("allow_missing_mutation must be bool.")

        self.enabled_scales = tuple(scale for scale in _SCALE_ORDER if scale in enabled_scales)
        self.mode = mode
        self.allow_missing_mutation = allow_missing_mutation

    def _pool(
        self,
        embeddings: Tensor,
        batch: Tensor,
        mask: Tensor,
        *,
        num_pairs: int,
        branch: str,
        scale: str,
    ) -> ScalePoolResult:
        return segmented_pool(
            embeddings,
            batch,
            mask,
            num_pairs=num_pairs,
            mode=self.mode,
            embeddings_name=f"H_{branch}",
            batch_name=f"batch_{branch}",
            selection_name=f"{scale}_mask_{branch}",
        )

    @staticmethod
    def _validate_mutation_counts(
        result: ScalePoolResult,
        *,
        name: str,
        allow_missing: bool,
    ) -> None:
        if torch.any(result.counts > 1):
            raise ValueError(f"{name} must select at most one row per pair.")
        if not allow_missing and torch.any(result.counts != 1):
            raise ValueError(f"{name} must select exactly one row per pair.")

    def forward(
        self,
        *,
        H_MUT: Tensor,
        H_WT: Tensor,
        H_delta: Tensor,
        batch_MUT: Tensor,
        batch_WT: Tensor,
        aligned_pair_batch: Tensor,
        num_pairs: int,
        alignment_ptr: Tensor | None = None,
        mutation_mask_MUT: Tensor | None = None,
        mutation_mask_WT: Tensor | None = None,
        mutation_mask_delta: Tensor | None = None,
        local_mask_MUT: Tensor | None = None,
        local_mask_WT: Tensor | None = None,
        local_mask_delta: Tensor | None = None,
        domain_mask_MUT: Tensor | None = None,
        domain_mask_WT: Tensor | None = None,
        domain_mask_delta: Tensor | None = None,
    ) -> ModelAMultiscalePoolingOutput:
        _validate_num_pairs(num_pairs)
        _validate_embeddings(H_MUT, name="H_MUT")
        _validate_embeddings(H_WT, name="H_WT")
        _validate_embeddings(H_delta, name="H_delta")
        if H_MUT.device != H_WT.device or H_MUT.device != H_delta.device:
            raise ValueError(
                "H_MUT, H_WT, and H_delta must be on the same device, got "
                f"{H_MUT.device}, {H_WT.device}, and {H_delta.device}."
            )
        _validate_batch(
            batch_MUT,
            name="batch_MUT",
            row_count=H_MUT.shape[0],
            num_pairs=num_pairs,
            device=H_MUT.device,
        )
        _validate_batch(
            batch_WT,
            name="batch_WT",
            row_count=H_WT.shape[0],
            num_pairs=num_pairs,
            device=H_WT.device,
        )
        validate_delta_segmentation(
            aligned_pair_batch,
            alignment_ptr,
            alignment_count=H_delta.shape[0],
            num_pairs=num_pairs,
            device=H_delta.device,
        )

        branch_embeddings = {"MUT": H_MUT, "WT": H_WT, "delta": H_delta}
        branch_batches = {
            "MUT": batch_MUT,
            "WT": batch_WT,
            "delta": aligned_pair_batch,
        }
        masks = {
            "mutation": {
                "MUT": mutation_mask_MUT,
                "WT": mutation_mask_WT,
                "delta": mutation_mask_delta,
            },
            "local": {
                "MUT": local_mask_MUT,
                "WT": local_mask_WT,
                "delta": local_mask_delta,
            },
            "domain": {
                "MUT": domain_mask_MUT,
                "WT": domain_mask_WT,
                "delta": domain_mask_delta,
            },
            "global": {
                "MUT": torch.ones(H_MUT.shape[0], dtype=torch.bool, device=H_MUT.device),
                "WT": torch.ones(H_WT.shape[0], dtype=torch.bool, device=H_WT.device),
                "delta": torch.ones(
                    H_delta.shape[0], dtype=torch.bool, device=H_delta.device
                ),
            },
        }

        pooled: dict[str, dict[str, ScalePoolResult | None]] = {
            branch: {scale: None for scale in _SCALE_ORDER}
            for branch in branch_embeddings
        }
        for scale in self.enabled_scales:
            for branch in branch_embeddings:
                mask = _validate_mask(
                    masks[scale][branch],
                    name=f"{scale}_mask_{branch}",
                    row_count=branch_embeddings[branch].shape[0],
                    device=branch_embeddings[branch].device,
                    required=True,
                )
                result = self._pool(
                    branch_embeddings[branch],
                    branch_batches[branch],
                    mask,
                    num_pairs=num_pairs,
                    branch=branch,
                    scale=scale,
                )
                if scale == "mutation":
                    self._validate_mutation_counts(
                        result,
                        name=f"mutation_mask_{branch}",
                        allow_missing=self.allow_missing_mutation or branch == "delta",
                    )
                pooled[branch][scale] = result

        def branch_output(name: str) -> BranchMultiscalePooling:
            return BranchMultiscalePooling(
                mutation=pooled[name]["mutation"],
                local=pooled[name]["local"],
                domain=pooled[name]["domain"],
                global_=pooled[name]["global"],
            )

        return ModelAMultiscalePoolingOutput(
            MUT=branch_output("MUT"),
            WT=branch_output("WT"),
            delta=branch_output("delta"),
        )
