"""Minimal HDF5 graph component loader for PyTorch Geometric-compatible graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gnn_siamese.data.pairing import SignatureParseError, parse_variant_signature
from gnn_siamese.data.validation import (
    CASE_KIND_MISSENSE,
    CASE_KIND_WT_COMPANION,
    classify_case,
)


class HDF5GraphLoadError(ValueError):
    """Raised when a graph cannot be loaded into valid graph components."""


@dataclass(frozen=True)
class NodeFeatureSlice:
    """Exact half-open column interval produced for one node feature."""

    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class HDF5GraphComponents:
    """Arrays and metadata needed to build one PyG graph."""

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    metadata: dict[str, Any]
    node_feature_names: tuple[str, ...]
    node_feature_slices: tuple[NodeFeatureSlice, ...]
    edge_feature_names: tuple[str, ...]
    node_availability_masks: dict[str, np.ndarray]
    mutation_node_index: int | None
    is_mutation: np.ndarray


DEFAULT_MUTATION_NODE_CONFIG: dict[str, Any] = {
    "source": "diff_features",
    "probes": ["diff_mass", "diff_charge", "diff_pI", "diff_size"],
    "epsilon": 1.0e-12,
    "require_exactly_one_for_missense": True,
    "wt_expected_count": 0,
    "create_is_mutation_channel": True,
}

_DISALLOWED_INPUT_FEATURES = {
    "custom_structure_energy",
    "custom_complex_energy_phenotype",
    "graph_features",
}


def _mutation_config_from(
    config: Mapping[str, Any] | None,
    mutation_node_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(DEFAULT_MUTATION_NODE_CONFIG)
    if config is not None:
        data_cfg = config.get("data")
        if isinstance(data_cfg, Mapping):
            nested = data_cfg.get("mutation_node")
            if isinstance(nested, Mapping):
                merged.update(nested)
    if mutation_node_config is not None:
        merged.update(mutation_node_config)
    return merged


def _read_group_datasets(group: h5py.Group) -> dict[str, np.ndarray]:
    return {name: dataset[()] for name, dataset in group.items()}


def _is_numeric_array(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number)


def _as_feature_matrix(
    value: Any,
    *,
    feature_name: str,
    expected_rows: int | None,
    feature_kind: str,
    reject_nan: bool,
    reject_inf: bool,
) -> np.ndarray:
    array = np.asarray(value)
    if not _is_numeric_array(array):
        raise HDF5GraphLoadError(f"{feature_kind} feature {feature_name!r} must be numeric.")
    if array.ndim == 0:
        raise HDF5GraphLoadError(
            f"{feature_kind} feature {feature_name!r} must have a row dimension."
        )
    if expected_rows is not None and int(array.shape[0]) != expected_rows:
        raise HDF5GraphLoadError(
            f"{feature_kind} feature {feature_name!r} first dimension {array.shape[0]} "
            f"does not match expected rows {expected_rows}."
        )

    matrix = array.reshape(int(array.shape[0]), -1).astype(np.float32, copy=False)
    if reject_nan and np.isnan(matrix).any():
        raise HDF5GraphLoadError(f"{feature_kind} feature {feature_name!r} contains NaN.")
    if reject_inf and np.isinf(matrix).any():
        raise HDF5GraphLoadError(f"{feature_kind} feature {feature_name!r} contains Inf.")
    return matrix


def _validate_selected_feature_names(feature_names: Sequence[str], *, feature_kind: str) -> tuple[str, ...]:
    selected: list[str] = []
    for name in feature_names:
        if not isinstance(name, str) or not name:
            raise HDF5GraphLoadError(f"{feature_kind} feature names must be non-empty strings.")
        if name.startswith("_") or name.startswith("mask_") or name in _DISALLOWED_INPUT_FEATURES:
            raise HDF5GraphLoadError(
                f"{feature_kind} feature {name!r} is not allowed as an encoder input."
            )
        selected.append(name)
    return tuple(selected)


def normalize_edge_index(edge_index: Any, *, num_nodes: int) -> np.ndarray:
    """Return edge indices in PyG orientation ``(2, E)`` and validate bounds."""

    array = np.asarray(edge_index)
    if array.ndim != 2:
        raise HDF5GraphLoadError(
            f"edge_index must be 2D with orientation (E, 2) or (2, E); got {array.shape!r}."
        )
    if array.shape[1] == 2:
        normalized = array.T
    elif array.shape[0] == 2:
        normalized = array
    else:
        raise HDF5GraphLoadError(
            f"edge_index must have orientation (E, 2) or (2, E); got {array.shape!r}."
        )

    if not np.issubdtype(normalized.dtype, np.integer):
        as_float = normalized.astype(np.float64, copy=False)
        if not np.all(np.equal(as_float, np.floor(as_float))):
            raise HDF5GraphLoadError("edge_index must contain integer-compatible values.")
        normalized = as_float.astype(np.int64)
    else:
        normalized = normalized.astype(np.int64, copy=False)

    if normalized.size:
        if (normalized < 0).any():
            raise HDF5GraphLoadError("edge_index contains negative node indices.")
        max_index = int(normalized.max())
        if max_index >= num_nodes:
            raise HDF5GraphLoadError(
                f"edge_index contains out-of-range node index {max_index} for num_nodes={num_nodes}."
            )
    return np.ascontiguousarray(normalized)


def _load_feature_block(
    features: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    expected_rows: int | None,
    feature_kind: str,
    reject_nan: bool,
    reject_inf: bool,
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for name in feature_names:
        if name not in features:
            raise HDF5GraphLoadError(f"{feature_kind} feature {name!r} is missing.")
        matrices.append(
            _as_feature_matrix(
                features[name],
                feature_name=name,
                expected_rows=expected_rows,
                feature_kind=feature_kind,
                reject_nan=reject_nan,
                reject_inf=reject_inf,
            )
        )
    if not matrices:
        rows = 0 if expected_rows is None else int(expected_rows)
        return np.empty((rows, 0), dtype=np.float32)
    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False)


def _load_node_feature_block(
    features: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    expected_rows: int,
    reject_nan: bool,
    reject_inf: bool,
) -> tuple[np.ndarray, tuple[NodeFeatureSlice, ...]]:
    """Load node columns and record their exact runtime concatenation slices."""

    matrices: list[np.ndarray] = []
    slices: list[NodeFeatureSlice] = []
    offset = 0
    for feature_name in feature_names:
        if feature_name not in features:
            raise HDF5GraphLoadError(f"node feature {feature_name!r} is missing.")
        matrix = _as_feature_matrix(
            features[feature_name],
            feature_name=feature_name,
            expected_rows=expected_rows,
            feature_kind="node",
            reject_nan=reject_nan,
            reject_inf=reject_inf,
        )
        stop = offset + int(matrix.shape[1])
        slices.append(NodeFeatureSlice(feature_name, offset, stop))
        matrices.append(matrix)
        offset = stop
    if not matrices:
        return np.empty((expected_rows, 0), dtype=np.float32), ()
    return (
        np.concatenate(matrices, axis=1).astype(np.float32, copy=False),
        tuple(slices),
    )


def validate_node_feature_slices(
    slices: Sequence[NodeFeatureSlice],
    *,
    width: int,
) -> tuple[NodeFeatureSlice, ...]:
    """Validate a complete, ordered, non-overlapping node-column layout."""

    resolved = tuple(slices)
    if not resolved:
        raise HDF5GraphLoadError("node_feature_slices must be non-empty.")
    names: set[str] = set()
    expected_start = 0
    for index, item in enumerate(resolved):
        if not isinstance(item, NodeFeatureSlice):
            raise HDF5GraphLoadError(
                f"node_feature_slices[{index}] must be a NodeFeatureSlice."
            )
        if not isinstance(item.name, str) or not item.name:
            raise HDF5GraphLoadError("Node feature slice names must be non-empty strings.")
        if item.name in names:
            raise HDF5GraphLoadError(
                f"Duplicate node feature slice name {item.name!r}."
            )
        if (
            not isinstance(item.start, int)
            or isinstance(item.start, bool)
            or not isinstance(item.stop, int)
            or isinstance(item.stop, bool)
        ):
            raise HDF5GraphLoadError(
                f"Slice for {item.name!r} must use integer start and stop."
            )
        if item.start != expected_start:
            relation = "overlaps" if item.start < expected_start else "leaves a gap"
            raise HDF5GraphLoadError(
                f"Slice for {item.name!r} {relation}: expected start "
                f"{expected_start}, got {item.start}."
            )
        if not 0 <= item.start < item.stop <= width:
            raise HDF5GraphLoadError(
                f"Slice for {item.name!r} is outside width {width}: "
                f"[{item.start}, {item.stop})."
            )
        names.add(item.name)
        expected_start = item.stop
    if expected_start != width:
        raise HDF5GraphLoadError(
            f"node_feature_slices are incomplete: covered {expected_start} of {width} columns."
        )
    return resolved


def _num_nodes_from_selected_features(
    node_features: Mapping[str, Any],
    node_feature_names: Sequence[str],
) -> int:
    if not node_feature_names:
        raise HDF5GraphLoadError("At least one selected node feature is required.")
    first_name = node_feature_names[0]
    if first_name not in node_features:
        raise HDF5GraphLoadError(f"Node feature {first_name!r} is missing.")
    shape = np.asarray(node_features[first_name]).shape
    if not shape:
        raise HDF5GraphLoadError(f"Node feature {first_name!r} must have a node dimension.")
    return int(shape[0])


def _num_nodes_from_available_features(
    node_features: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    context: str,
) -> int:
    for name in feature_names:
        if name not in node_features:
            continue
        shape = np.asarray(node_features[name]).shape
        if not shape:
            raise HDF5GraphLoadError(f"{context} feature {name!r} must have a node dimension.")
        return int(shape[0])
    raise HDF5GraphLoadError(f"No configured {context} features are available.")


def _coerce_mask_vector(value: Any, *, mask_name: str, num_nodes: int) -> np.ndarray:
    mask = _as_feature_matrix(
        value,
        feature_name=mask_name,
        expected_rows=num_nodes,
        feature_kind="node availability mask",
        reject_nan=True,
        reject_inf=True,
    )
    if mask.shape[1] != 1:
        raise HDF5GraphLoadError(
            f"Node availability mask {mask_name!r} must be scalar per node; got shape {mask.shape!r}."
        )
    return mask[:, 0].astype(np.float32, copy=False)


def load_node_availability_masks(
    node_features: Mapping[str, Any],
    node_availability_masks: Mapping[str, str] | None,
    *,
    num_nodes: int,
) -> dict[str, np.ndarray]:
    """Load requested masks as auxiliary arrays without adding them to ``x``."""

    loaded: dict[str, np.ndarray] = {}
    for feature_name, mask_name in (node_availability_masks or {}).items():
        if mask_name not in node_features:
            raise HDF5GraphLoadError(
                f"Availability mask {mask_name!r} for feature {feature_name!r} is missing."
            )
        loaded[feature_name] = _coerce_mask_vector(
            node_features[mask_name],
            mask_name=mask_name,
            num_nodes=num_nodes,
        )
    return loaded


def build_is_mutation_channel(
    node_features: Mapping[str, Any],
    *,
    graph_key: str,
    source_h5: str | Path | None = None,
    graph_features: Mapping[str, Any] | None = None,
    mutation_node_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, int | None]:
    """Infer and validate the mutation-node channel from configured ``diff_*`` probes."""

    cfg = _mutation_config_from(None, mutation_node_config)
    source = str(cfg.get("source", "diff_features"))
    if source != "diff_features":
        raise HDF5GraphLoadError(
            f"Unsupported mutation_node.source {source!r}; only 'diff_features' is supported."
        )

    probes = list(cfg.get("probes", []))
    if not probes:
        raise HDF5GraphLoadError("mutation_node.probes must contain at least one diff_* feature.")

    num_nodes = _num_nodes_from_available_features(
        node_features,
        probes,
        context="mutation probe",
    )
    epsilon = float(cfg.get("epsilon", 1.0e-12))
    is_mutation = np.zeros(num_nodes, dtype=np.float32)
    used_probe_count = 0

    for probe in probes:
        if probe not in node_features:
            continue
        values = _as_feature_matrix(
            node_features[probe],
            feature_name=probe,
            expected_rows=num_nodes,
            feature_kind="mutation probe",
            reject_nan=True,
            reject_inf=True,
        )
        used_probe_count += 1
        is_mutation[np.any(np.abs(values) > epsilon, axis=1)] = 1.0

    if used_probe_count == 0:
        raise HDF5GraphLoadError(
            "None of the configured mutation_node.probes are available to infer is_mutation."
        )

    case_kind = classify_case(
        graph_key,
        file_name=Path(source_h5).name if source_h5 is not None else None,
        graph_features=graph_features,
    )
    mutation_count = int(is_mutation.sum())
    if case_kind == CASE_KIND_MISSENSE and cfg.get("require_exactly_one_for_missense", True):
        if mutation_count != 1:
            raise HDF5GraphLoadError(
                f"missense graph {graph_key!r} expects exactly one mutated node, got {mutation_count}."
            )
    elif case_kind == CASE_KIND_WT_COMPANION:
        expected = int(cfg.get("wt_expected_count", 0))
        if mutation_count != expected:
            raise HDF5GraphLoadError(
                f"WT companion graph {graph_key!r} expects {expected} mutated nodes, got {mutation_count}."
            )

    indices = np.flatnonzero(is_mutation)
    mutation_node_index = int(indices[0]) if len(indices) == 1 else None
    return is_mutation, mutation_node_index


def extract_variant_metadata(
    graph_key: str,
    *,
    h5_path: str | Path,
    graph_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract stable metadata from the graph key and source path."""

    del graph_features
    metadata: dict[str, Any] = {
        "variant_id": graph_key,
        "source_h5": str(h5_path),
        "graph_key": graph_key,
    }
    try:
        signature = parse_variant_signature({"variant_id": graph_key, "source_path": str(h5_path)})
    except SignatureParseError:
        return {
            **metadata,
            "position": None,
            "wt_aa": None,
            "mut_aa": None,
        }
    return {
        **metadata,
        "variant_id": signature.variant_id,
        "position": signature.position,
        "wt_aa": signature.wt_aa,
        "mut_aa": signature.mut_aa,
        "chain_id": signature.chain_id,
        "wt_aa_full": signature.wt_aa_full,
        "mut_aa_full": signature.mut_aa_full,
    }


def validate_graph_components(components: HDF5GraphComponents) -> None:
    """Validate loaded graph component shapes and finite numeric values."""

    if components.x.ndim != 2:
        raise HDF5GraphLoadError(f"x must be 2D, got shape {components.x.shape!r}.")
    if components.edge_index.ndim != 2 or components.edge_index.shape[0] != 2:
        raise HDF5GraphLoadError(
            f"edge_index must have PyG shape (2, E), got {components.edge_index.shape!r}."
        )
    if components.edge_attr.ndim != 2:
        raise HDF5GraphLoadError(
            f"edge_attr must be 2D, got shape {components.edge_attr.shape!r}."
        )
    if components.edge_attr.shape[0] != components.edge_index.shape[1]:
        raise HDF5GraphLoadError(
            "edge_attr row count must match number of edges: "
            f"{components.edge_attr.shape[0]} != {components.edge_index.shape[1]}."
        )
    if components.is_mutation.shape != (components.x.shape[0],):
        raise HDF5GraphLoadError(
            f"is_mutation must have shape ({components.x.shape[0]},), "
            f"got {components.is_mutation.shape!r}."
        )
    validated_slices = validate_node_feature_slices(
        components.node_feature_slices,
        width=int(components.x.shape[1]),
    )
    if tuple(item.name for item in validated_slices) != components.node_feature_names:
        raise HDF5GraphLoadError(
            "node_feature_slices names must exactly match node_feature_names in order."
        )
    if not np.isfinite(components.x).all():
        raise HDF5GraphLoadError("x contains NaN or Inf.")
    if not np.isfinite(components.edge_attr).all():
        raise HDF5GraphLoadError("edge_attr contains NaN or Inf.")
    for feature_name, mask in components.node_availability_masks.items():
        if mask.shape != (components.x.shape[0],):
            raise HDF5GraphLoadError(
                f"Availability mask for {feature_name!r} has incompatible shape {mask.shape!r}."
            )


def load_hdf5_graph_components(
    h5_path: str | Path,
    graph_key: str,
    *,
    node_feature_names: Sequence[str],
    edge_feature_names: Sequence[str],
    node_availability_masks: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    mutation_node_config: Mapping[str, Any] | None = None,
    reject_nan: bool = True,
    reject_inf: bool = True,
) -> HDF5GraphComponents:
    """Load one HDF5 graph into arrays ready for a single PyG ``Data`` object."""

    node_names = _validate_selected_feature_names(node_feature_names, feature_kind="node")
    edge_names = _validate_selected_feature_names(edge_feature_names, feature_kind="edge")
    mutation_cfg = _mutation_config_from(config, mutation_node_config)

    with h5py.File(h5_path, "r") as handle:
        if graph_key not in handle:
            raise HDF5GraphLoadError(f"Graph key {graph_key!r} not found in {h5_path!s}.")
        graph = handle[graph_key]
        for group_name in ("node_features", "edge_features", "graph_features"):
            if group_name not in graph:
                raise HDF5GraphLoadError(f"Graph {graph_key!r} is missing {group_name!r}.")

        node_features = _read_group_datasets(graph["node_features"])
        edge_features = _read_group_datasets(graph["edge_features"])
        graph_features = _read_group_datasets(graph["graph_features"])

    num_nodes = _num_nodes_from_selected_features(node_features, node_names)
    x, node_feature_slices = _load_node_feature_block(
        node_features,
        node_names,
        expected_rows=num_nodes,
        reject_nan=reject_nan,
        reject_inf=reject_inf,
    )

    if "_index" not in edge_features:
        raise HDF5GraphLoadError("edge_features/_index is missing.")
    edge_index = normalize_edge_index(edge_features["_index"], num_nodes=num_nodes)
    num_edges = int(edge_index.shape[1])
    edge_attr = _load_feature_block(
        edge_features,
        edge_names,
        expected_rows=num_edges,
        feature_kind="edge",
        reject_nan=reject_nan,
        reject_inf=reject_inf,
    )

    masks = load_node_availability_masks(
        node_features,
        node_availability_masks,
        num_nodes=num_nodes,
    )
    is_mutation, mutation_node_index = build_is_mutation_channel(
        node_features,
        graph_key=graph_key,
        source_h5=h5_path,
        graph_features=graph_features,
        mutation_node_config=mutation_cfg,
    )

    output_node_names = node_names
    if bool(mutation_cfg.get("create_is_mutation_channel", True)):
        mutation_start = int(x.shape[1])
        x = np.concatenate([x, is_mutation.reshape(num_nodes, 1)], axis=1).astype(
            np.float32,
            copy=False,
        )
        output_node_names = (*node_names, "is_mutation")
        node_feature_slices = (
            *node_feature_slices,
            NodeFeatureSlice("is_mutation", mutation_start, mutation_start + 1),
        )
    node_feature_slices = validate_node_feature_slices(
        node_feature_slices,
        width=int(x.shape[1]),
    )

    components = HDF5GraphComponents(
        x=x.astype(np.float32, copy=False),
        edge_index=edge_index,
        edge_attr=edge_attr.astype(np.float32, copy=False),
        metadata=extract_variant_metadata(graph_key, h5_path=h5_path, graph_features=graph_features),
        node_feature_names=output_node_names,
        node_feature_slices=node_feature_slices,
        edge_feature_names=edge_names,
        node_availability_masks=masks,
        mutation_node_index=mutation_node_index,
        is_mutation=is_mutation,
    )
    validate_graph_components(components)
    return components
