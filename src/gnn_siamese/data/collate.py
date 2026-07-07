"""Minimal collate helpers for paired mutant-WT graph samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gnn_siamese.data.dataset import MutWtPairSample
from gnn_siamese.data.hdf5_loader import HDF5GraphComponents
from gnn_siamese.data.pairing import PairingKey


class MutWtPairCollateError(ValueError):
    """Raised when paired samples cannot be collated into a consistent batch."""


@dataclass(frozen=True)
class BatchedGraphComponents:
    """Minimal batched graph structure with PyG-style bookkeeping arrays."""

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    node_availability_masks: dict[str, np.ndarray]
    is_mutation: np.ndarray
    batch: np.ndarray
    ptr: np.ndarray
    graph_metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class MutWtPairBatch:
    """Minimal batch of paired mutant and WT graph components."""

    graph_mut: BatchedGraphComponents
    graph_wt: BatchedGraphComponents
    metadata: list[dict[str, Any]]
    variant_ids: list[str]
    mutant_keys: list[str]
    wt_keys: list[str]
    pair_keys: list[PairingKey]
    batch_size: int


def _validate_non_empty_samples(samples: Sequence[MutWtPairSample]) -> None:
    if not samples:
        raise MutWtPairCollateError("collate_mut_wt_pairs requires a non-empty sequence of samples.")


def _validate_graph_names(
    graphs: Sequence[HDF5GraphComponents],
    *,
    graph_role: str,
    attribute_name: str,
) -> tuple[str, ...]:
    reference = getattr(graphs[0], attribute_name)
    for index, graph in enumerate(graphs[1:], start=1):
        candidate = getattr(graph, attribute_name)
        if candidate != reference:
            raise MutWtPairCollateError(
                f"Incompatible {attribute_name} in {graph_role} graph at batch index {index}: "
                f"{candidate!r} != {reference!r}."
            )
    return reference


def _validate_mask_keys(
    graphs: Sequence[HDF5GraphComponents],
    *,
    graph_role: str,
) -> tuple[str, ...]:
    reference = tuple(graphs[0].node_availability_masks.keys())
    for index, graph in enumerate(graphs[1:], start=1):
        candidate = tuple(graph.node_availability_masks.keys())
        if candidate != reference:
            raise MutWtPairCollateError(
                f"Incompatible node_availability_masks keys in {graph_role} graph at batch index {index}: "
                f"{candidate!r} != {reference!r}."
            )
    return reference


def _validate_mutation_counts(
    graphs: Sequence[HDF5GraphComponents],
    *,
    graph_role: str,
    expected_sum: float,
) -> None:
    for index, graph in enumerate(graphs):
        mutation_sum = float(np.asarray(graph.is_mutation, dtype=np.float32).sum())
        if not np.isclose(mutation_sum, expected_sum):
            raise MutWtPairCollateError(
                f"{graph_role} graph at batch index {index} has invalid is_mutation sum "
                f"{mutation_sum}; expected {expected_sum}."
            )


def _validate_cross_graph_compatibility(
    graph_mut: Sequence[HDF5GraphComponents],
    graph_wt: Sequence[HDF5GraphComponents],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    mut_node_names = _validate_graph_names(
        graph_mut,
        graph_role="mutant",
        attribute_name="node_feature_names",
    )
    wt_node_names = _validate_graph_names(
        graph_wt,
        graph_role="WT",
        attribute_name="node_feature_names",
    )
    if mut_node_names != wt_node_names:
        raise MutWtPairCollateError(
            "Mutant and WT node_feature_names must match: "
            f"{mut_node_names!r} != {wt_node_names!r}."
        )

    mut_edge_names = _validate_graph_names(
        graph_mut,
        graph_role="mutant",
        attribute_name="edge_feature_names",
    )
    wt_edge_names = _validate_graph_names(
        graph_wt,
        graph_role="WT",
        attribute_name="edge_feature_names",
    )
    if mut_edge_names != wt_edge_names:
        raise MutWtPairCollateError(
            "Mutant and WT edge_feature_names must match: "
            f"{mut_edge_names!r} != {wt_edge_names!r}."
        )

    mut_mask_keys = _validate_mask_keys(graph_mut, graph_role="mutant")
    wt_mask_keys = _validate_mask_keys(graph_wt, graph_role="WT")
    if mut_mask_keys != wt_mask_keys:
        raise MutWtPairCollateError(
            "Mutant and WT node_availability_masks keys must match: "
            f"{mut_mask_keys!r} != {wt_mask_keys!r}."
        )
    return mut_node_names, mut_edge_names, mut_mask_keys


def _collate_graphs(
    graphs: Sequence[HDF5GraphComponents],
    *,
    mask_keys: Sequence[str],
) -> BatchedGraphComponents:
    x_parts: list[np.ndarray] = []
    edge_attr_parts: list[np.ndarray] = []
    edge_index_parts: list[np.ndarray] = []
    is_mutation_parts: list[np.ndarray] = []
    batch_parts: list[np.ndarray] = []
    graph_metadata: list[dict[str, Any]] = []
    mask_parts: dict[str, list[np.ndarray]] = {key: [] for key in mask_keys}
    ptr = [0]
    node_offset = 0
    expected_edge_features = len(graphs[0].edge_feature_names)

    for graph_index, graph in enumerate(graphs):
        num_nodes = int(graph.x.shape[0])
        x_parts.append(np.asarray(graph.x, dtype=np.float32))
        edge_attr = np.asarray(graph.edge_attr, dtype=np.float32)
        if edge_attr.ndim == 1:
            if edge_attr.shape[0] == 0:
                edge_attr = np.empty((0, expected_edge_features), dtype=np.float32)
            elif expected_edge_features == 1:
                edge_attr = edge_attr.reshape(-1, 1)
            else:
                raise MutWtPairCollateError(
                    f"Graph at batch index {graph_index} has 1D edge_attr with shape "
                    f"{edge_attr.shape!r}; expected 2D edge_attr with {expected_edge_features} features."
                )
        elif edge_attr.ndim != 2:
            raise MutWtPairCollateError(
                f"Graph at batch index {graph_index} has invalid edge_attr shape {edge_attr.shape!r}; "
                "expected a 2D array."
            )
        if edge_attr.shape[1] != expected_edge_features:
            raise MutWtPairCollateError(
                f"Graph at batch index {graph_index} has edge_attr width {edge_attr.shape[1]}, "
                f"expected {expected_edge_features} from edge_feature_names."
            )
        edge_attr_parts.append(edge_attr.astype(np.float32, copy=False))

        edge_index = np.asarray(graph.edge_index, dtype=np.int64)
        if edge_index.shape == (0, 2):
            edge_index = edge_index.T
        elif edge_index.size == 0 and edge_index.ndim == 1:
            edge_index = np.empty((2, 0), dtype=np.int64)
        if edge_index.shape != (2, edge_attr.shape[0]):
            raise MutWtPairCollateError(
                f"Graph at batch index {graph_index} has incompatible edge_index shape "
                f"{edge_index.shape!r} for {edge_attr.shape[0]} edges."
            )
        edge_index_parts.append(edge_index + node_offset)
        is_mutation_parts.append(np.asarray(graph.is_mutation, dtype=np.float32))
        batch_parts.append(np.full(num_nodes, graph_index, dtype=np.int64))
        graph_metadata.append(dict(graph.metadata))
        for mask_key in mask_keys:
            mask_parts[mask_key].append(
                np.asarray(graph.node_availability_masks[mask_key], dtype=np.float32)
            )
        node_offset += num_nodes
        ptr.append(node_offset)

    x = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)
    edge_attr = np.concatenate(edge_attr_parts, axis=0).astype(np.float32, copy=False)
    if edge_index_parts:
        edge_index = np.concatenate(edge_index_parts, axis=1).astype(np.int64, copy=False)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    is_mutation = np.concatenate(is_mutation_parts, axis=0).astype(np.float32, copy=False)
    batch = np.concatenate(batch_parts, axis=0).astype(np.int64, copy=False)
    masks = {
        mask_key: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for mask_key, parts in mask_parts.items()
    }

    return BatchedGraphComponents(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_feature_names=graphs[0].node_feature_names,
        edge_feature_names=graphs[0].edge_feature_names,
        node_availability_masks=masks,
        is_mutation=is_mutation,
        batch=batch,
        ptr=np.asarray(ptr, dtype=np.int64),
        graph_metadata=graph_metadata,
    )


def collate_mut_wt_pairs(samples: Sequence[MutWtPairSample]) -> MutWtPairBatch:
    """Collate a non-empty sequence of paired mutant-WT samples into one batch."""

    _validate_non_empty_samples(samples)
    graph_mut = [sample.graph_mut for sample in samples]
    graph_wt = [sample.graph_wt for sample in samples]

    _, _, mask_keys = _validate_cross_graph_compatibility(graph_mut, graph_wt)
    _validate_mutation_counts(graph_mut, graph_role="Mutant", expected_sum=1.0)
    _validate_mutation_counts(graph_wt, graph_role="WT", expected_sum=0.0)

    return MutWtPairBatch(
        graph_mut=_collate_graphs(graph_mut, mask_keys=mask_keys),
        graph_wt=_collate_graphs(graph_wt, mask_keys=mask_keys),
        metadata=[dict(sample.metadata) for sample in samples],
        variant_ids=[sample.variant_id for sample in samples],
        mutant_keys=[sample.mutant_key for sample in samples],
        wt_keys=[sample.wt_key for sample in samples],
        pair_keys=[sample.pair_key for sample in samples],
        batch_size=len(samples),
    )
