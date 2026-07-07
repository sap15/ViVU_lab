"""Siamese wrappers around the shared edge-aware graph encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor, nn

from gnn_siamese.models.encoder import EdgeAwareGraphEncoder, EncoderBranchOutput

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
        return output


class SharedSiameseEncoderModel(nn.Module):
    """Process mutant and WT graphs with one shared edge-aware encoder."""

    def __init__(
        self,
        shared_encoder: EdgeAwareGraphEncoder,
        relational_module: "RelationalRepresentation | None" = None,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        self.relational_module = relational_module

    def encode_graph(self, graph: object) -> EncoderBranchOutput:
        return self.shared_encoder(graph)

    def forward(self, *, graph_mut: object, graph_wt: object) -> SharedSiameseEncoderOutput:
        mut_branch = self.encode_graph(graph_mut)
        wt_branch = self.encode_graph(graph_wt)
        relational_output: RelationalOutput | None = None
        if self.relational_module is not None:
            relational_output = self.relational_module(mut_branch.h_encoder, wt_branch.h_encoder)

        return SharedSiameseEncoderOutput(
            H_mut=mut_branch.H,
            H_WT=wt_branch.H,
            h_encoder_mut=mut_branch.h_encoder,
            h_encoder_wt=wt_branch.h_encoder,
            r_delta=None if relational_output is None else relational_output.r_delta,
            z_delta=None if relational_output is None else relational_output.z_delta,
            severity=None if relational_output is None else relational_output.severity,
            mechanism_direction=None if relational_output is None else relational_output.mechanism_direction,
            z_delta_status="not_applicable" if relational_output is None else relational_output.z_delta_status,
        )
