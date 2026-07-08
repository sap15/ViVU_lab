"""Pooling helpers for mutation-centric graph representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PoolingBlocks:
    """Named pooling blocks before fusion."""

    global_pool: Tensor
    mutation_node: Tensor
    immediate_neighbors: Tensor
    local_pool: Tensor
    regional_pool: Tensor
    availability_mask: Tensor

    def fusion_input(self) -> Tensor:
        return torch.cat(
            [
                self.global_pool,
                self.mutation_node,
                self.immediate_neighbors,
                self.local_pool,
                self.regional_pool,
                self.availability_mask,
            ],
            dim=-1,
        )

    def concatenate(self) -> Tensor:
        return self.fusion_input()


def _num_graphs_from_batch(batch: Tensor) -> int:
    if batch.numel() == 0:
        return 0
    return int(batch.max().item()) + 1


def mean_pool_with_mask(node_embeddings: Tensor, batch: Tensor, node_mask: Tensor, num_graphs: int) -> Tensor:
    """Mean-pool masked nodes per graph and return zeros for unavailable blocks."""

    hidden_dim = int(node_embeddings.size(-1))
    pooled = node_embeddings.new_zeros((num_graphs, hidden_dim))
    if node_embeddings.numel() == 0 or not bool(node_mask.any()):
        return pooled

    mask = node_mask.to(dtype=torch.bool, device=node_embeddings.device)
    selected_embeddings = node_embeddings[mask]
    selected_batch = batch[mask]
    pooled.index_add_(0, selected_batch, selected_embeddings)
    counts = torch.bincount(selected_batch, minlength=num_graphs).to(node_embeddings.dtype).unsqueeze(1)
    safe_counts = counts.clamp_min(1.0)
    pooled = pooled / safe_counts
    zero_graphs = counts.squeeze(1) == 0
    if bool(zero_graphs.any()):
        pooled[zero_graphs] = 0.0
    return pooled


def availability_mask_summary(node_availability_masks: dict[str, Tensor] | None, batch: Tensor) -> Tensor:
    """Summarize node-level availability masks into one graph-level block."""

    num_graphs = _num_graphs_from_batch(batch)
    if not node_availability_masks:
        return torch.zeros((num_graphs, 1), dtype=torch.float32, device=batch.device)

    summaries: list[Tensor] = []
    for mask in node_availability_masks.values():
        mask_tensor = mask.to(device=batch.device, dtype=torch.float32).reshape(-1)
        summaries.append(mean_pool_with_mask(mask_tensor.unsqueeze(1), batch, torch.ones_like(mask_tensor, dtype=torch.bool), num_graphs))
    stacked = torch.cat(summaries, dim=-1)
    return stacked.mean(dim=-1, keepdim=True)


def neighbor_mask(edge_index: Tensor, seed_mask: Tensor) -> Tensor:
    """Return the 1-hop undirected neighborhood of the seed nodes, excluding the seeds."""

    mask = seed_mask.to(dtype=torch.bool, device=edge_index.device)
    if edge_index.numel() == 0 or not bool(mask.any()):
        return mask.new_zeros(mask.shape)

    src, dst = edge_index
    outbound = mask[src]
    inbound = mask[dst]
    neighbors = mask.new_zeros(mask.shape)
    neighbors[dst[outbound]] = True
    neighbors[src[inbound]] = True
    neighbors = neighbors & ~mask
    return neighbors


def expand_hops(edge_index: Tensor, seed_mask: Tensor, hops: int) -> Tensor:
    """Expand an undirected node mask over ``hops`` message-passing steps."""

    mask = seed_mask.to(dtype=torch.bool, device=edge_index.device)
    if hops <= 0 or edge_index.numel() == 0 or not bool(mask.any()):
        return mask

    expanded = mask.clone()
    frontier = mask.clone()
    for _ in range(hops):
        frontier = neighbor_mask(edge_index, frontier)
        new_nodes = frontier & ~expanded
        expanded = expanded | new_nodes
        frontier = new_nodes
        if not bool(frontier.any()):
            break
    return expanded


class MutationAwarePooling(torch.nn.Module):
    """Pooling stack for global, mutation-centric and regional graph views."""

    def __init__(self, *, local_hops: int = 1, regional_hops: int = 2) -> None:
        super().__init__()
        self.local_hops = int(local_hops)
        self.regional_hops = int(regional_hops)

    def forward(
        self,
        node_embeddings: Tensor,
        *,
        edge_index: Tensor,
        batch: Tensor,
        is_mutation: Tensor,
        node_availability_masks: dict[str, Tensor] | None = None,
        local_node_mask: Tensor | None = None,
        regional_node_mask: Tensor | None = None,
    ) -> PoolingBlocks:
        num_graphs = _num_graphs_from_batch(batch)
        global_mask = torch.ones_like(is_mutation, dtype=torch.bool, device=node_embeddings.device)
        mutation_mask = is_mutation.to(dtype=torch.bool, device=node_embeddings.device)
        immediate_mask = neighbor_mask(edge_index, mutation_mask)

        if local_node_mask is None:
            local_mask = expand_hops(edge_index, mutation_mask, self.local_hops)
        else:
            local_mask = local_node_mask.to(dtype=torch.bool, device=node_embeddings.device)

        if regional_node_mask is None:
            regional_mask = expand_hops(edge_index, mutation_mask, self.regional_hops)
        else:
            regional_mask = regional_node_mask.to(dtype=torch.bool, device=node_embeddings.device)

        return PoolingBlocks(
            global_pool=mean_pool_with_mask(node_embeddings, batch, global_mask, num_graphs),
            mutation_node=mean_pool_with_mask(node_embeddings, batch, mutation_mask, num_graphs),
            immediate_neighbors=mean_pool_with_mask(node_embeddings, batch, immediate_mask, num_graphs),
            local_pool=mean_pool_with_mask(node_embeddings, batch, local_mask, num_graphs),
            regional_pool=mean_pool_with_mask(node_embeddings, batch, regional_mask, num_graphs),
            availability_mask=availability_mask_summary(node_availability_masks, batch),
        )
