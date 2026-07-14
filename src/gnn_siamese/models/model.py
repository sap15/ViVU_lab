"""Siamese wrappers around the shared edge-aware graph encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor, nn

from gnn_siamese.models.encoder import EdgeAwareGraphEncoder, EncoderBranchOutput
from gnn_siamese.models.projection import InstanceProjectionHead, PairProjectionHead

if TYPE_CHECKING:
    from gnn_siamese.models.relational import RelationalOutput, RelationalRepresentation


@dataclass(frozen=True)
class SharedSiameseEncoderOutput:
    """Structured outputs for the paired mutant-WT encoder."""

    H_mut: Tensor
    H_WT: Tensor
    h_encoder_mut: Tensor
    h_encoder_wt: Tensor
    r_delta: Tensor | None = None
    z_delta: Tensor | None = None
    z_instance: Tensor | None = None
    z_instance_pair: Tensor | None = None
    severity: Tensor | None = None
    mechanism_direction: Tensor | None = None
    z_delta_status: str = "not_applicable"

    @property
    def h_mut(self) -> Tensor:
        return self.h_encoder_mut

    @property
    def h_wt(self) -> Tensor:
        return self.h_encoder_wt

    @property
    def z_delta_is_validated(self) -> bool:
        return self.z_delta is not None and self.z_delta_status == "validated"

    def to_dict(self) -> dict[str, Tensor | str]:
        output: dict[str, Tensor | str] = {
            "H_mut": self.H_mut,
            "H_WT": self.H_WT,
            "h_encoder_mut": self.h_encoder_mut,
            "h_encoder_wt": self.h_encoder_wt,
            "h_mut": self.h_mut,
            "h_wt": self.h_wt,
        }
        if self.r_delta is not None:
            output["r_delta"] = self.r_delta
        if self.severity is not None:
            output["severity"] = self.severity
        if self.mechanism_direction is not None:
            output["mechanism_direction"] = self.mechanism_direction
        if self.z_delta is not None:
            output["z_delta"] = self.z_delta
            output["z_delta_status"] = self.z_delta_status
        if self.z_instance is not None:
            output["z_instance"] = self.z_instance
        if self.z_instance_pair is not None:
            output["z_instance_pair"] = self.z_instance_pair
        return output


@dataclass(frozen=True)
class ModelBContrastiveOutput:
    """Operational two-view output for the end-to-end Model B baseline."""

    z1: Tensor
    z2: Tensor
    view1: SharedSiameseEncoderOutput
    view2: SharedSiameseEncoderOutput

    def to_dict(self) -> dict[str, Tensor]:
        return {
            "z1": self.z1,
            "z2": self.z2,
            "h_encoder_mut_view1": self.view1.h_encoder_mut,
            "h_encoder_mut_view2": self.view2.h_encoder_mut,
            "h_encoder_wt_view1": self.view1.h_encoder_wt,
            "h_encoder_wt_view2": self.view2.h_encoder_wt,
        }


class SharedSiameseEncoderModel(nn.Module):
    """Process mutant and WT graphs with one shared edge-aware encoder."""

    def __init__(
        self,
        shared_encoder: EdgeAwareGraphEncoder,
        relational_module: "RelationalRepresentation | None" = None,
        projection_instance: InstanceProjectionHead | None = None,
        projection_pair: PairProjectionHead | None = None,
        pair_projection_source: str = "r_delta",
    ) -> None:
        super().__init__()
        if pair_projection_source not in {"r_delta", "z_delta"}:
            raise ValueError("pair_projection_source must be either 'r_delta' or 'z_delta'.")
        self.shared_encoder = shared_encoder
        self.relational_module = relational_module
        self.projection_instance = projection_instance
        self.projection_pair = projection_pair
        self.pair_projection_source = pair_projection_source

    def encode_graph(self, graph: object) -> EncoderBranchOutput:
        return self.shared_encoder(graph)

    def forward(self, *, graph_mut: object, graph_wt: object) -> SharedSiameseEncoderOutput:
        mut_branch = self.encode_graph(graph_mut)
        wt_branch = self.encode_graph(graph_wt)
        relational_output: RelationalOutput | None = None
        if self.relational_module is not None:
            relational_output = self.relational_module(mut_branch.h_encoder, wt_branch.h_encoder)
        z_instance = None
        if self.projection_instance is not None:
            z_instance = self.projection_instance(mut_branch.h_encoder, input_name="h_encoder_mut")
        z_instance_pair = None
        if self.projection_pair is not None:
            if relational_output is None:
                raise ValueError("projection_pair requires relational outputs, but relational_module is None.")
            if self.pair_projection_source == "r_delta":
                z_instance_pair = self.projection_pair(relational_output.r_delta, input_name="r_delta")
            else:
                if not relational_output.z_delta_is_validated or relational_output.z_delta is None:
                    raise ValueError(
                        "projection_pair configured with source 'z_delta', but z_delta is not validated."
                    )
                z_instance_pair = self.projection_pair(relational_output.z_delta, input_name="z_delta")

        return SharedSiameseEncoderOutput(
            H_mut=mut_branch.H,
            H_WT=wt_branch.H,
            h_encoder_mut=mut_branch.h_encoder,
            h_encoder_wt=wt_branch.h_encoder,
            r_delta=None if relational_output is None else relational_output.r_delta,
            z_delta=None if relational_output is None else relational_output.z_delta,
            z_instance=z_instance,
            z_instance_pair=z_instance_pair,
            severity=None if relational_output is None else relational_output.severity,
            mechanism_direction=None if relational_output is None else relational_output.mechanism_direction,
            z_delta_status="not_applicable" if relational_output is None else relational_output.z_delta_status,
        )


class ModelBContrastiveBaseline(nn.Module):
    """Run the shared Mutant-WT encoder twice and expose `z1`/`z2` for NT-Xent."""

    def __init__(self, siamese_model: SharedSiameseEncoderModel) -> None:
        super().__init__()
        if siamese_model.projection_instance is None:
            raise ValueError("Model B baseline requires projection_instance to produce z1 and z2.")
        self.siamese_model = siamese_model
        self.architecture_name = "model_b"

    def forward(
        self,
        *,
        view1_graph_mut: object,
        view1_graph_wt: object,
        view2_graph_mut: object,
        view2_graph_wt: object,
    ) -> ModelBContrastiveOutput:
        view1 = self.siamese_model(graph_mut=view1_graph_mut, graph_wt=view1_graph_wt)
        view2 = self.siamese_model(graph_mut=view2_graph_mut, graph_wt=view2_graph_wt)
        if view1.z_instance is None or view2.z_instance is None:
            raise ValueError("Model B baseline requires z_instance in both augmented views.")
        return ModelBContrastiveOutput(
            z1=view1.z_instance,
            z2=view2.z_instance,
            view1=view1,
            view2=view2,
        )
