"""Collate helpers for paired mutant-WT graph samples using real PyG batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

try:
    from torch_geometric.data import Batch, Data
except ImportError:  # pragma: no cover - exercised only when PyG is unavailable.
    Batch = None
    Data = None

from gnn_siamese.data.dataset import MutWtPairSample
from gnn_siamese.data.hdf5_loader import (
    HDF5GraphComponents,
    HDF5GraphLoadError,
    NodeFeatureSlice,
    validate_node_feature_slices,
)
from gnn_siamese.data.pairing import PairingKey


class MutWtPairCollateError(ValueError):
    """Raised when paired samples cannot be collated into a consistent batch."""


BatchedGraphComponents = Batch


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
    mut_aligned_index: torch.Tensor
    wt_aligned_index: torch.Tensor
    aligned_pair_batch: torch.Tensor
    alignment_ptr: torch.Tensor
    exists_MUT: torch.Tensor
    exists_WT: torch.Tensor
    union_pair_batch: torch.Tensor
    union_ptr: torch.Tensor
    local_mut_aligned_index: torch.Tensor
    local_wt_aligned_index: torch.Tensor
    local_alignment_ptr: torch.Tensor

    def to(self, device: torch.device | str) -> MutWtPairBatch:
        """Return the complete paired batch on ``device``."""

        tensor_names = (
            "mut_aligned_index",
            "wt_aligned_index",
            "aligned_pair_batch",
            "alignment_ptr",
            "exists_MUT",
            "exists_WT",
            "union_pair_batch",
            "union_ptr",
            "local_mut_aligned_index",
            "local_wt_aligned_index",
            "local_alignment_ptr",
        )
        return replace(
            self,
            graph_mut=self.graph_mut.to(device),
            graph_wt=self.graph_wt.to(device),
            **{name: getattr(self, name).to(device) for name in tensor_names},
        )


def _validate_non_empty_samples(samples: Sequence[MutWtPairSample]) -> None:
    if not samples:
        raise MutWtPairCollateError("collate_mut_wt_pairs requires a non-empty sequence of samples.")


def _require_pyg() -> tuple[type[Batch], type[Data]]:
    if Batch is None or Data is None:
        raise ImportError(
            "collate_mut_wt_pairs requires torch_geometric. Install PyTorch Geometric "
            "or run the tests in an environment such as deeprank2_env where PyG is available."
        )
    return Batch, Data


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


def _validate_node_feature_slices(
    graphs: Sequence[HDF5GraphComponents],
    *,
    graph_role: str,
) -> tuple[NodeFeatureSlice, ...]:
    reference = graphs[0].node_feature_slices
    for index, graph in enumerate(graphs):
        try:
            validated = validate_node_feature_slices(
                graph.node_feature_slices,
                width=int(graph.x.shape[1]),
            )
        except HDF5GraphLoadError as exc:
            raise MutWtPairCollateError(
                f"Invalid node_feature_slices in {graph_role} graph at batch "
                f"index {index}: {exc}"
            ) from exc
        if tuple(item.name for item in validated) != graph.node_feature_names:
            raise MutWtPairCollateError(
                f"node_feature_slices names in {graph_role} graph at batch index "
                f"{index} do not match node_feature_names."
            )
        if validated != reference:
            raise MutWtPairCollateError(
                f"Incompatible node_feature_slices in {graph_role} graph at batch "
                f"index {index}: {validated!r} != {reference!r}."
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
    _validate_node_feature_slices(graph_mut, graph_role="mutant")
    _validate_node_feature_slices(graph_wt, graph_role="WT")
    return mut_node_names, mut_edge_names, mut_mask_keys


def _collate_graphs(
    graphs: Sequence[HDF5GraphComponents],
    *,
    mask_keys: Sequence[str],
) -> Batch:
    batch_cls, data_cls = _require_pyg()
    data_list: list[Data] = []
    graph_metadata: list[dict[str, Any]] = []
    mask_parts: dict[str, list[np.ndarray]] = {key: [] for key in mask_keys}
    expected_edge_features = len(graphs[0].edge_feature_names)

    for graph_index, graph in enumerate(graphs):
        x = np.asarray(graph.x, dtype=np.float32)
        if x.ndim != 2:
            raise MutWtPairCollateError(
                f"Graph at batch index {graph_index} has invalid x shape {x.shape!r}; "
                "expected a 2D array."
            )

        edge_index = np.asarray(graph.edge_index, dtype=np.int64)
        if edge_index.shape == (0, 2):
            edge_index = edge_index.T
        elif edge_index.size == 0 and edge_index.ndim == 1:
            edge_index = np.empty((2, 0), dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise MutWtPairCollateError(
                f"Graph at batch index {graph_index} has invalid edge_index shape "
                f"{edge_index.shape!r}; expected (2, E)."
            )

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
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise MutWtPairCollateError(
                f"Graph at batch index {graph_index} has incompatible edge_index shape "
                f"{edge_index.shape!r} for {edge_attr.shape[0]} edges."
            )

        data_list.append(
            data_cls(
                x=torch.as_tensor(x, dtype=torch.float32),
                edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                edge_attr=torch.as_tensor(edge_attr.astype(np.float32, copy=False)),
                is_mutation=torch.as_tensor(np.asarray(graph.is_mutation, dtype=np.float32)),
            )
        )
        graph_metadata.append(dict(graph.metadata))
        for mask_key in mask_keys:
            mask_parts[mask_key].append(
                np.asarray(graph.node_availability_masks[mask_key], dtype=np.float32)
            )

    batch = batch_cls.from_data_list(data_list)
    batch.node_feature_names = graphs[0].node_feature_names
    batch.node_feature_slices = graphs[0].node_feature_slices
    batch.edge_feature_names = graphs[0].edge_feature_names
    batch.node_availability_masks = {
        mask_key: torch.as_tensor(np.concatenate(parts, axis=0).astype(np.float32, copy=False))
        for mask_key, parts in mask_parts.items()
    }
    batch.graph_metadata = graph_metadata
    return batch


def _ptr(lengths: Sequence[int]) -> torch.Tensor:
    return torch.tensor([0, *np.cumsum(lengths, dtype=np.int64).tolist()], dtype=torch.long)


def _cat_long(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(parts)) if parts else torch.empty(0, dtype=torch.long)


def _collate_alignments(
    samples: Sequence[MutWtPairSample],
) -> dict[str, torch.Tensor]:
    mut_offsets = np.cumsum(
        [0, *(int(sample.graph_mut.x.shape[0]) for sample in samples[:-1])],
        dtype=np.int64,
    )
    wt_offsets = np.cumsum(
        [0, *(int(sample.graph_wt.x.shape[0]) for sample in samples[:-1])],
        dtype=np.int64,
    )
    alignment_lengths = [len(sample.mut_aligned_index) for sample in samples]
    union_lengths = [len(sample.exists_MUT) for sample in samples]
    local_lengths = [len(sample.local_mut_aligned_index) for sample in samples]

    def offset_parts(attribute: str, offsets: np.ndarray) -> list[torch.Tensor]:
        return [
            torch.as_tensor(getattr(sample, attribute), dtype=torch.long) + int(offset)
            for sample, offset in zip(samples, offsets)
        ]

    return {
        "mut_aligned_index": _cat_long(offset_parts("mut_aligned_index", mut_offsets)),
        "wt_aligned_index": _cat_long(offset_parts("wt_aligned_index", wt_offsets)),
        "aligned_pair_batch": torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.long),
            torch.tensor(alignment_lengths, dtype=torch.long),
        ),
        "alignment_ptr": _ptr(alignment_lengths),
        "exists_MUT": torch.as_tensor(
            [value for sample in samples for value in sample.exists_MUT],
            dtype=torch.bool,
        ),
        "exists_WT": torch.as_tensor(
            [value for sample in samples for value in sample.exists_WT],
            dtype=torch.bool,
        ),
        "union_pair_batch": torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.long),
            torch.tensor(union_lengths, dtype=torch.long),
        ),
        "union_ptr": _ptr(union_lengths),
        "local_mut_aligned_index": _cat_long(
            offset_parts("local_mut_aligned_index", mut_offsets)
        ),
        "local_wt_aligned_index": _cat_long(
            offset_parts("local_wt_aligned_index", wt_offsets)
        ),
        "local_alignment_ptr": _ptr(local_lengths),
    }


def collate_mut_wt_pairs(samples: Sequence[MutWtPairSample]) -> MutWtPairBatch:
    """Collate a non-empty sequence of paired mutant-WT samples into one batch."""

    _validate_non_empty_samples(samples)
    graph_mut = [sample.graph_mut for sample in samples]
    graph_wt = [sample.graph_wt for sample in samples]

    _, _, mask_keys = _validate_cross_graph_compatibility(graph_mut, graph_wt)
    _validate_mutation_counts(graph_mut, graph_role="Mutant", expected_sum=1.0)
    _validate_mutation_counts(graph_wt, graph_role="WT", expected_sum=0.0)
    alignment_batch = _collate_alignments(samples)

    return MutWtPairBatch(
        graph_mut=_collate_graphs(graph_mut, mask_keys=mask_keys),
        graph_wt=_collate_graphs(graph_wt, mask_keys=mask_keys),
        metadata=[dict(sample.metadata) for sample in samples],
        variant_ids=[sample.variant_id for sample in samples],
        mutant_keys=[sample.mutant_key for sample in samples],
        wt_keys=[sample.wt_key for sample in samples],
        pair_keys=[sample.pair_key for sample in samples],
        batch_size=len(samples),
        **alignment_batch,
    )
