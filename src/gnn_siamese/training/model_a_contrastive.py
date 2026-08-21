"""Minimal A6 contrastive route for two complete Model A pair views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from gnn_siamese.losses import NTXentLoss, build_false_negative_mask
from gnn_siamese.models.model_a import ModelATwoViewOutput
from gnn_siamese.models.projection_a import ModelAProjectionHead


@dataclass(frozen=True)
class ModelAContrastiveOutput:
    """Structured A6 outputs and effective negatives for all 2B anchors."""

    z_instance_pair_view1: Tensor
    z_instance_pair_view2: Tensor
    loss: Tensor
    valid_negative_counts: Tensor


class ModelAContrastive(nn.Module):
    """Apply one shared pair projection head and same-position NT-Xent."""

    def __init__(
        self,
        *,
        projection_head: ModelAProjectionHead,
        nt_xent: NTXentLoss,
        mask_mode: str = "same_position",
        min_valid_negatives: int = 1,
        min_valid_fraction: float = 0.0,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(projection_head, ModelAProjectionHead):
            raise TypeError("projection_head must be a ModelAProjectionHead.")
        if not isinstance(nt_xent, NTXentLoss):
            raise TypeError("nt_xent must be an NTXentLoss.")
        if not nt_xent.normalize:
            raise ValueError(
                "Model A requires NTXentLoss(normalize=True); L2 normalization "
                "is performed exactly once inside NT-Xent."
            )
        self.projection_head = projection_head
        self.nt_xent = nt_xent
        if (mask_mode, min_valid_negatives, min_valid_fraction, strict) != (
            "same_position", 1, 0.0, True
        ):
            raise ValueError("Model A requires same_position/min=1/fraction=0/strict=true.")
        self.mask_mode = mask_mode
        self.min_valid_negatives = min_valid_negatives
        self.min_valid_fraction = min_valid_fraction
        self.strict = strict

    def forward(
        self,
        two_view_output: ModelATwoViewOutput,
        *,
        positions: Sequence[int] | Tensor,
    ) -> ModelAContrastiveOutput:
        """Consume only each structured view's ``z_delta_pair`` field."""

        if not isinstance(two_view_output, ModelATwoViewOutput):
            raise TypeError(
                "two_view_output must be a ModelATwoViewOutput containing "
                "z_delta_pair for both complete Mut-WT views."
            )
        z_delta_pair_view1 = getattr(two_view_output.view1, "z_delta_pair", None)
        z_delta_pair_view2 = getattr(two_view_output.view2, "z_delta_pair", None)
        if z_delta_pair_view1 is None or z_delta_pair_view2 is None:
            raise ValueError("Both Model A views must provide z_delta_pair.")

        z_instance_pair_view1 = self.projection_head(z_delta_pair_view1)
        z_instance_pair_view2 = self.projection_head(z_delta_pair_view2)
        batch_size = z_instance_pair_view1.shape[0]
        mask = build_false_negative_mask(
            batch_size=batch_size,
            mode=self.mask_mode,
            positions=positions,
            min_valid_negatives=self.min_valid_negatives,
            min_valid_fraction=self.min_valid_fraction,
            strict=self.strict,
        )
        loss_output = self.nt_xent(
            z_instance_pair_view1,
            z_instance_pair_view2,
            mask_output=mask,
        )
        valid_negative_counts = torch.tensor(
            [int(entry.valid_negatives) for entry in mask.per_anchor_stats],
            dtype=torch.long,
            device=z_instance_pair_view1.device,
        )
        return ModelAContrastiveOutput(
            z_instance_pair_view1=z_instance_pair_view1,
            z_instance_pair_view2=z_instance_pair_view2,
            loss=loss_output.loss,
            valid_negative_counts=valid_negative_counts,
        )
