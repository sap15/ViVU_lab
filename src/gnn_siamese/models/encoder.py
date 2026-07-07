"""Shared edge-aware graph encoder for paired mutant-WT inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv

from gnn_siamese.models.pooling import MutationAwarePooling


@dataclass(frozen=True)
class EncoderBranchOutput:
    """Node and graph representations for one graph branch."""

    H: Tensor
    h_encoder: Tensor


class EdgeAwareGraphEncoder(nn.Module):
    """NNConv-based encoder with mutation-centric pooling and fusion."""

    def __init__(
        self,
        *,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        edge_mlp_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        graph_output_dim: int = 64,
        dropout: float = 0.1,
        local_hops: int = 1,
        regional_hops: int = 2,
    ) -> None:
        super().__init__()
        if node_input_dim <= 0:
            raise ValueError("node_input_dim must be positive.")
        if edge_input_dim <= 0:
            raise ValueError("edge_input_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.node_input_dim = int(node_input_dim)
        self.edge_input_dim = int(edge_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.graph_output_dim = int(graph_output_dim)
        self.fusion_input_dim = 5 * self.hidden_dim

        self.input_projection = nn.Linear(self.node_input_dim, self.hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            edge_network = nn.Sequential(
                nn.Linear(self.edge_input_dim, edge_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_mlp_hidden_dim, self.hidden_dim * self.hidden_dim),
            )
            self.convs.append(NNConv(self.hidden_dim, self.hidden_dim, edge_network, aggr="mean"))
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.pooling = MutationAwarePooling(local_hops=local_hops, regional_hops=regional_hops)
        self.pre_fusion_norm = nn.LayerNorm(self.fusion_input_dim)
        self.mlp_fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(self.fusion_hidden_dim, self.graph_output_dim),
        )
        self.post_fusion_norm = nn.LayerNorm(self.graph_output_dim)

    def forward(self, graph_batch: object) -> EncoderBranchOutput:
        if not hasattr(graph_batch, "x") or not hasattr(graph_batch, "edge_index") or not hasattr(graph_batch, "edge_attr"):
            raise ValueError("graph_batch must provide x, edge_index and edge_attr.")
        if not hasattr(graph_batch, "batch") or not hasattr(graph_batch, "is_mutation"):
            raise ValueError("graph_batch must provide batch and is_mutation.")

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        is_mutation = graph_batch.is_mutation

        H = self.input_projection(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = H
            H = conv(H, edge_index, edge_attr)
            H = norm(H)
            H = F.relu(H)
            H = self.dropout(H)
            H = H + residual

        pooling_blocks = self.pooling(
            H,
            edge_index=edge_index,
            batch=batch,
            is_mutation=is_mutation,
            node_availability_masks=getattr(graph_batch, "node_availability_masks", None),
            local_node_mask=getattr(graph_batch, "local_node_mask", None),
            regional_node_mask=getattr(graph_batch, "regional_node_mask", None),
        )
        fused_input = pooling_blocks.fusion_input()
        normalized_input = self.pre_fusion_norm(fused_input)
        h_encoder = self.post_fusion_norm(self.mlp_fusion(normalized_input))
        return EncoderBranchOutput(H=H, h_encoder=h_encoder)
