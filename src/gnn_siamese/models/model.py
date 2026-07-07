"""Siamese wrappers around the shared edge-aware graph encoder."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from gnn_siamese.models.encoder import EdgeAwareGraphEncoder, EncoderBranchOutput


@dataclass(frozen=True)
class SharedSiameseEncoderOutput:
    """Structured outputs for the paired mutant-WT encoder."""

    H_mut: Tensor
    H_WT: Tensor
    h_encoder_mut: Tensor
    h_encoder_wt: Tensor

    @property
    def h_mut(self) -> Tensor:
        return self.h_encoder_mut

    @property
    def h_wt(self) -> Tensor:
        return self.h_encoder_wt


class SharedSiameseEncoderModel(nn.Module):
    """Process mutant and WT graphs with one shared edge-aware encoder."""

    def __init__(self, shared_encoder: EdgeAwareGraphEncoder) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder

    def encode_graph(self, graph: object) -> EncoderBranchOutput:
        return self.shared_encoder(graph)

    def forward(self, *, graph_mut: object, graph_wt: object) -> SharedSiameseEncoderOutput:
        mut_branch = self.encode_graph(graph_mut)
        wt_branch = self.encode_graph(graph_wt)
        return SharedSiameseEncoderOutput(
            H_mut=mut_branch.H,
            H_WT=wt_branch.H,
            h_encoder_mut=mut_branch.h_encoder,
            h_encoder_wt=wt_branch.h_encoder,
        )
