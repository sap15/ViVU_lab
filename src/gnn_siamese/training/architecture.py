"""Small trainer-facing adapter shared by the independent final architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from gnn_siamese.losses import NTXentLoss
from gnn_siamese.models import ModelANodalMultiscalePair, ModelBContrastiveBaseline


@dataclass(frozen=True)
class ContrastiveBatchOutput:
    architecture: str
    loss: Tensor | None
    model_output: Any

    def to_dict(self) -> dict[str, Any]:
        payload = (
            self.model_output.to_dict()
            if hasattr(self.model_output, "to_dict")
            else dict(self.model_output)
        )
        return {"architecture": self.architecture, "loss": self.loss, **payload}


def forward_contrastive_batch(
    model: nn.Module,
    batch: Any,
    *,
    augmenter: Any,
    loss_fn: NTXentLoss,
    run_seed: int,
    epoch: int,
) -> ContrastiveBatchOutput:
    """Consume one official pair batch without conflating A and B internals."""

    if isinstance(model, ModelANodalMultiscalePair):
        output = model(batch, run_seed=run_seed, epoch=epoch)
        return ContrastiveBatchOutput(model.architecture_name, output.loss, output)
    if isinstance(model, ModelBContrastiveBaseline):
        view1_mut, view2_mut = augmenter.create_two_views(batch.graph_mut)
        output = model(
            view1_graph_mut=view1_mut,
            view1_graph_wt=batch.graph_wt,
            view2_graph_mut=view2_mut,
            view2_graph_wt=batch.graph_wt,
        )
        return ContrastiveBatchOutput(
            "model_b_graph_level_relational", None, output
        )
    raise TypeError(
        "Unsupported final model type for shared contrastive trainer interface: "
        f"{type(model).__name__}."
    )
