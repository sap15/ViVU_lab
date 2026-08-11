"""One-view end-to-end assembly of Model A through A4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from gnn_siamese.data.model_a_pair_augmentations import (
    ModelAAugmentationExampleMetadata,
    ModelAPairAugmenter,
)
from gnn_siamese.models.delta_block import NodeDeltaBlock
from gnn_siamese.models.multiscale_pooling_a import (
    ModelAMultiscalePooling,
    aligned_selection_mask,
    indices_to_mask,
)
from gnn_siamese.models.multiscale_relational_a import ModelAMultiscaleRelational


@dataclass(frozen=True)
class ModelAOneViewOutput:
    H_MUT: Tensor
    H_WT: Tensor
    H_delta: Tensor
    h_mutation_MUT: Tensor | None
    h_mutation_WT: Tensor | None
    h_mutation_delta: Tensor | None
    h_local_MUT: Tensor | None
    h_local_WT: Tensor | None
    h_local_delta: Tensor | None
    h_domain_MUT: Tensor | None
    h_domain_WT: Tensor | None
    h_domain_delta: Tensor | None
    h_global_MUT: Tensor | None
    h_global_WT: Tensor | None
    h_global_delta: Tensor | None
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
    alignment_metadata: dict[str, Tensor | None]
    variant_id: Any = None


class ModelAOneView(nn.Module):
    """Shared encoder -> node delta -> A3 pooling -> A4 relational fusion."""

    def __init__(
        self,
        *,
        shared_encoder: nn.Module,
        node_delta_block: NodeDeltaBlock,
        multiscale_pooling: ModelAMultiscalePooling,
        multiscale_relational: ModelAMultiscaleRelational,
    ) -> None:
        super().__init__()
        if not isinstance(shared_encoder, nn.Module):
            raise TypeError("shared_encoder must be an nn.Module.")
        if not isinstance(node_delta_block, NodeDeltaBlock):
            raise TypeError("node_delta_block must be a NodeDeltaBlock.")
        if not isinstance(multiscale_pooling, ModelAMultiscalePooling):
            raise TypeError("multiscale_pooling must be a ModelAMultiscalePooling.")
        if not isinstance(multiscale_relational, ModelAMultiscaleRelational):
            raise TypeError(
                "multiscale_relational must be a ModelAMultiscaleRelational."
            )
        if multiscale_pooling.enabled_scales != multiscale_relational.active_scales:
            raise ValueError(
                "A3 enabled_scales and A4 active_scales must match canonically, got "
                f"{multiscale_pooling.enabled_scales} and "
                f"{multiscale_relational.active_scales}."
            )
        if node_delta_block.output_dim != multiscale_relational.embedding_dim:
            raise ValueError(
                "NodeDeltaBlock output_dim must equal A4 embedding_dim, got "
                f"{node_delta_block.output_dim} and "
                f"{multiscale_relational.embedding_dim}."
            )

        self.shared_encoder = shared_encoder
        self.node_delta_block = node_delta_block
        self.multiscale_pooling = multiscale_pooling
        self.multiscale_relational = multiscale_relational

    @staticmethod
    def _values(pooling: object, branch: str, scale: str) -> Tensor | None:
        branch_output = getattr(pooling, branch)
        result = getattr(branch_output, "global_" if scale == "global" else scale)
        return None if result is None else result.values

    def forward(
        self,
        *,
        graph_mut: object,
        graph_wt: object,
        mut_aligned_index: Tensor,
        wt_aligned_index: Tensor,
        aligned_pair_batch: Tensor,
        alignment_ptr: Tensor,
        num_pairs: int,
        mutation_mask_MUT: Tensor | None = None,
        mutation_mask_WT: Tensor | None = None,
        mutation_mask_delta: Tensor | None = None,
        local_mask_MUT: Tensor | None = None,
        local_mask_WT: Tensor | None = None,
        local_mask_delta: Tensor | None = None,
        domain_mask_MUT: Tensor | None = None,
        domain_mask_WT: Tensor | None = None,
        domain_mask_delta: Tensor | None = None,
        variant_id: Any = None,
    ) -> ModelAOneViewOutput:
        mut_output = self.shared_encoder(graph_mut)
        wt_output = self.shared_encoder(graph_wt)
        H_MUT = mut_output.H
        H_WT = wt_output.H
        delta = self.node_delta_block(
            H_MUT,
            H_WT,
            mut_aligned_index,
            wt_aligned_index,
            aligned_pair_batch,
            alignment_ptr,
        )
        pooling = self.multiscale_pooling(
            H_MUT=H_MUT,
            H_WT=H_WT,
            H_delta=delta.H_delta,
            batch_MUT=graph_mut.batch,
            batch_WT=graph_wt.batch,
            aligned_pair_batch=aligned_pair_batch,
            alignment_ptr=alignment_ptr,
            num_pairs=num_pairs,
            mutation_mask_MUT=mutation_mask_MUT,
            mutation_mask_WT=mutation_mask_WT,
            mutation_mask_delta=mutation_mask_delta,
            local_mask_MUT=local_mask_MUT,
            local_mask_WT=local_mask_WT,
            local_mask_delta=local_mask_delta,
            domain_mask_MUT=domain_mask_MUT,
            domain_mask_WT=domain_mask_WT,
            domain_mask_delta=domain_mask_delta,
        )
        relational = self.multiscale_relational(pooling)
        values = lambda branch, scale: self._values(pooling, branch, scale)
        return ModelAOneViewOutput(
            H_MUT=H_MUT,
            H_WT=H_WT,
            H_delta=delta.H_delta,
            h_mutation_MUT=values("MUT", "mutation"),
            h_mutation_WT=values("WT", "mutation"),
            h_mutation_delta=values("delta", "mutation"),
            h_local_MUT=values("MUT", "local"),
            h_local_WT=values("WT", "local"),
            h_local_delta=values("delta", "local"),
            h_domain_MUT=values("MUT", "domain"),
            h_domain_WT=values("WT", "domain"),
            h_domain_delta=values("delta", "domain"),
            h_global_MUT=values("MUT", "global"),
            h_global_WT=values("WT", "global"),
            h_global_delta=values("delta", "global"),
            z_delta_mutation=relational.z_delta_mutation,
            z_delta_local=relational.z_delta_local,
            z_delta_domain=relational.z_delta_domain,
            z_delta_global=relational.z_delta_global,
            h_pair_delta=relational.h_pair_delta,
            z_delta_pair=relational.z_delta_pair,
            active_scales=relational.active_scales,
            scale_order=relational.scale_order,
            scale_dimensions=relational.scale_dimensions,
            pair_dimension=relational.pair_dimension,
            scale_valid_masks=relational.scale_valid_masks,
            scale_counts=relational.scale_counts,
            pair_valid_mask=relational.pair_valid_mask,
            pair_fusion_mode=relational.pair_fusion_mode,
            alignment_metadata={
                "mut_aligned_index": mut_aligned_index,
                "wt_aligned_index": wt_aligned_index,
                "aligned_pair_batch": aligned_pair_batch,
                "alignment_ptr": alignment_ptr,
            },
            variant_id=variant_id,
        )


@dataclass(frozen=True)
class ModelATwoViewOutput:
    """A5 output before any pair projection head or contrastive loss."""

    view1: ModelAOneViewOutput
    view2: ModelAOneViewOutput
    h_pair_delta_view1: Tensor
    h_pair_delta_view2: Tensor
    z_delta_pair_view1: Tensor
    z_delta_pair_view2: Tensor
    augmentation_metadata_view1: tuple[ModelAAugmentationExampleMetadata, ...]
    augmentation_metadata_view2: tuple[ModelAAugmentationExampleMetadata, ...]


class ModelATwoView(nn.Module):
    """Run one shared A1-A4 instance over two conservative paired A5 views."""

    def __init__(
        self,
        *,
        one_view_model: ModelAOneView,
        pair_augmenter: ModelAPairAugmenter,
    ) -> None:
        super().__init__()
        if not isinstance(one_view_model, ModelAOneView):
            raise TypeError("one_view_model must be a ModelAOneView.")
        if not isinstance(pair_augmenter, ModelAPairAugmenter):
            raise TypeError("pair_augmenter must be a ModelAPairAugmenter.")
        self.one_view_model = one_view_model
        self.pair_augmenter = pair_augmenter

    def forward(
        self,
        pair_batch: object,
        *,
        run_seed: int,
        epoch: int,
        mutation_mask_MUT: Tensor | None = None,
        mutation_mask_WT: Tensor | None = None,
        mutation_mask_delta: Tensor | None = None,
        local_mask_MUT: Tensor | None = None,
        local_mask_WT: Tensor | None = None,
        local_mask_delta: Tensor | None = None,
        domain_mask_MUT: Tensor | None = None,
        domain_mask_WT: Tensor | None = None,
        domain_mask_delta: Tensor | None = None,
    ) -> ModelATwoViewOutput:
        view1, view2 = self.pair_augmenter.create_two_views(
            pair_batch, run_seed=run_seed, epoch=epoch
        )
        masks = {
            "mutation_mask_MUT": mutation_mask_MUT,
            "mutation_mask_WT": mutation_mask_WT,
            "mutation_mask_delta": mutation_mask_delta,
            "local_mask_MUT": local_mask_MUT,
            "local_mask_WT": local_mask_WT,
            "local_mask_delta": local_mask_delta,
            "domain_mask_MUT": domain_mask_MUT,
            "domain_mask_WT": domain_mask_WT,
            "domain_mask_delta": domain_mask_delta,
        }

        def run(view: object) -> ModelAOneViewOutput:
            batch = view.pair_batch
            return self.one_view_model(
                graph_mut=batch.graph_mut,
                graph_wt=batch.graph_wt,
                mut_aligned_index=batch.mut_aligned_index,
                wt_aligned_index=batch.wt_aligned_index,
                aligned_pair_batch=batch.aligned_pair_batch,
                alignment_ptr=batch.alignment_ptr,
                num_pairs=batch.batch_size,
                variant_id=tuple(batch.variant_ids),
                **masks,
            )

        output1 = run(view1)
        output2 = run(view2)
        return ModelATwoViewOutput(
            view1=output1,
            view2=output2,
            h_pair_delta_view1=output1.h_pair_delta,
            h_pair_delta_view2=output2.h_pair_delta,
            z_delta_pair_view1=output1.z_delta_pair,
            z_delta_pair_view2=output2.z_delta_pair,
            augmentation_metadata_view1=view1.augmentation_metadata,
            augmentation_metadata_view2=view2.augmentation_metadata,
        )


@dataclass(frozen=True)
class ModelANodalMultiscalePairOutput:
    """Stable trainer-facing output of the complete Model A route."""

    h_pair_delta_view1: Tensor
    h_pair_delta_view2: Tensor
    z_delta_pair_view1: Tensor
    z_delta_pair_view2: Tensor
    z_instance_pair_view1: Tensor
    z_instance_pair_view2: Tensor
    z_delta_local_view1: Tensor | None
    z_delta_local_view2: Tensor | None
    active_scales: tuple[str, ...]
    scale_order: tuple[str, ...]
    loss: Tensor
    valid_negative_counts: Tensor
    two_view_output: ModelATwoViewOutput

    @property
    def h_pair_delta(self) -> Tensor:
        return self.h_pair_delta_view1

    @property
    def z_delta_pair(self) -> Tensor:
        return self.z_delta_pair_view1

    @property
    def z_instance_pair(self) -> Tensor:
        return self.z_instance_pair_view1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "h_pair_delta": self.h_pair_delta,
            "z_delta_pair": self.z_delta_pair,
            "z_instance_pair": self.z_instance_pair,
            "h_pair_delta_view1": self.h_pair_delta_view1,
            "h_pair_delta_view2": self.h_pair_delta_view2,
            "z_delta_pair_view1": self.z_delta_pair_view1,
            "z_delta_pair_view2": self.z_delta_pair_view2,
            "z_instance_pair_view1": self.z_instance_pair_view1,
            "z_instance_pair_view2": self.z_instance_pair_view2,
            "active_scales": self.active_scales,
            "scale_order": self.scale_order,
            "loss": self.loss,
            "valid_negative_counts": self.valid_negative_counts,
        }
        if self.z_delta_local_view1 is not None:
            payload["z_delta_local_view1"] = self.z_delta_local_view1
            payload["z_delta_local_view2"] = self.z_delta_local_view2
        return payload


class ModelANodalMultiscalePair(nn.Module):
    """Final Model A architecture, kept independent from Model B."""

    architecture_name = "model_a_nodal_multiscale_pair"

    def __init__(self, *, two_view_model: ModelATwoView, contrastive: nn.Module) -> None:
        super().__init__()
        self.two_view_model = two_view_model
        self.contrastive = contrastive

    @property
    def pair_fusion(self) -> nn.Module:
        return self.two_view_model.one_view_model.multiscale_relational.pair_fusion

    @property
    def projection_pair_a(self) -> nn.Module:
        return self.contrastive.projection_head

    def forward(self, pair_batch: object, *, run_seed: int, epoch: int) -> ModelANodalMultiscalePairOutput:
        mutation_mut = pair_batch.graph_mut.is_mutation.bool()
        mutation_delta = mutation_mut[pair_batch.mut_aligned_index]
        mutation_wt = mutation_mut.new_zeros(pair_batch.graph_wt.num_nodes)
        mutation_wt[pair_batch.wt_aligned_index[mutation_delta]] = True
        masks: dict[str, Tensor] = {
            "mutation_mask_MUT": mutation_mut,
            "mutation_mask_WT": mutation_wt,
            "mutation_mask_delta": mutation_delta,
        }
        active_scales = self.two_view_model.one_view_model.multiscale_pooling.enabled_scales
        if "local" in active_scales:
            masks.update(
                {
                    "local_mask_MUT": indices_to_mask(
                        pair_batch.local_mut_aligned_index,
                        row_count=pair_batch.graph_mut.num_nodes,
                        name="local_mut_aligned_index",
                    ),
                    "local_mask_WT": indices_to_mask(
                        pair_batch.local_wt_aligned_index,
                        row_count=pair_batch.graph_wt.num_nodes,
                        name="local_wt_aligned_index",
                    ),
                    "local_mask_delta": aligned_selection_mask(
                        pair_batch.mut_aligned_index,
                        pair_batch.alignment_ptr,
                        pair_batch.local_mut_aligned_index,
                        pair_batch.local_alignment_ptr,
                        num_pairs=pair_batch.batch_size,
                    ),
                }
            )
        two_view = self.two_view_model(
            pair_batch,
            run_seed=run_seed,
            epoch=epoch,
            **masks,
        )
        positions = [int(item.position) for item in pair_batch.pair_keys]
        contrastive = self.contrastive(two_view, positions=positions)
        return ModelANodalMultiscalePairOutput(
            h_pair_delta_view1=two_view.h_pair_delta_view1,
            h_pair_delta_view2=two_view.h_pair_delta_view2,
            z_delta_pair_view1=two_view.z_delta_pair_view1,
            z_delta_pair_view2=two_view.z_delta_pair_view2,
            z_instance_pair_view1=contrastive.z_instance_pair_view1,
            z_instance_pair_view2=contrastive.z_instance_pair_view2,
            z_delta_local_view1=two_view.view1.z_delta_local,
            z_delta_local_view2=two_view.view2.z_delta_local,
            active_scales=two_view.view1.active_scales,
            scale_order=two_view.view1.scale_order,
            loss=contrastive.loss,
            valid_negative_counts=contrastive.valid_negative_counts,
            two_view_output=two_view,
        )
