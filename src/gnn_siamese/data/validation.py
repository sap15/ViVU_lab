"""Validation helpers for the observed HDF5 schema."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DIFF_PROBES_FOR_MUTATION = [
    "diff_mass",
    "diff_charge",
    "diff_pI",
    "diff_size",
    "diff_polarity",
    "diff_hb_donors",
    "diff_hb_acceptors",
]

CASE_KIND_MISSENSE = "missense"
CASE_KIND_WT_COMPANION = "wt_companion"
CASE_KIND_TRUNCATION = "truncation"

_STOP_PATTERN = re.compile(r"(?:^|[_:\-])(STOP|TER|TRUNC|TRUNCATION|X)(?:$|[_:\-])", re.IGNORECASE)
_WT_PATTERN = re.compile(r"(WT_COMPANION|WT_COMP|PKP2_WT)", re.IGNORECASE)


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        return tuple(shape)
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Sequence):
        if not value:
            return (0,)
        first = value[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return (len(value), len(first))
        return (len(value),)
    return ()


def _to_list(value: Any) -> list[Any]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _normalize_case_text(*parts: str | Path | None) -> str:
    return " ".join(str(part) for part in parts if part).upper()


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_numeric_values(value: Any) -> list[float]:
    numeric_values: list[float] = []
    stack = [_to_list(value)]
    while stack:
        current = stack.pop()
        for item in current:
            if isinstance(item, (list, tuple)):
                stack.append(list(item))
                continue
            numeric = _coerce_float(item)
            if numeric is not None:
                numeric_values.append(numeric)
    return numeric_values


def _num_nodes_from_node_features(node_features: Mapping[str, Any]) -> int:
    for value in node_features.values():
        shape = _shape_of(value)
        if shape:
            return int(shape[0])
    return 0


def _extract_scalar_bool(value: Any) -> bool:
    items = _to_list(value)
    if not items:
        return False
    numeric = _coerce_float(items[0])
    return bool(numeric) if numeric is not None else False


def _extract_numeric_node_vector(
    feature_name: str,
    value: Any,
    num_nodes: int,
) -> tuple[list[float] | None, str | None]:
    shape = _shape_of(value)
    values = _to_list(value)

    if shape and shape[0] != num_nodes:
        return None, (
            f"{feature_name} first dimension {shape[0]} does not match num_nodes={num_nodes}."
        )

    if len(shape) == 1:
        if len(values) != num_nodes:
            return None, (
                f"{feature_name} has incompatible shape for per-node numeric conversion: "
                f"expected length {num_nodes}, got {len(values)}."
            )
        converted: list[float] = []
        for index, raw_value in enumerate(values):
            numeric = _coerce_float(raw_value)
            if numeric is None:
                return None, f"{feature_name} contains non-numeric data at node index {index}."
            converted.append(numeric)
        return converted, None

    if len(shape) == 2 and shape[1] == 1:
        if len(values) != num_nodes:
            return None, (
                f"{feature_name} has incompatible shape for per-node numeric conversion: "
                f"expected length {num_nodes}, got {len(values)}."
            )
        converted = []
        for index, raw_value in enumerate(values):
            row = _to_list(raw_value)
            if len(row) != 1:
                return None, f"{feature_name} row {index} is not a scalar-valued node feature."
            numeric = _coerce_float(row[0])
            if numeric is None:
                return None, f"{feature_name} contains non-numeric data at node index {index}."
            converted.append(numeric)
        return converted, None

    return None, (
        f"{feature_name} is not convertible to a per-node numeric vector with shape (N,) or (N, 1); "
        f"observed shape {shape!r}."
    )


def _validate_numeric_node_feature(
    feature_name: str,
    value: Any,
    num_nodes: int,
) -> str | None:
    shape = _shape_of(value)
    if shape and shape[0] != num_nodes:
        return f"{feature_name} first dimension {shape[0]} does not match num_nodes={num_nodes}."

    values = _iter_numeric_values(value)
    if not values:
        return f"{feature_name} does not expose numeric node values."

    if any(math.isnan(numeric) or math.isinf(numeric) for numeric in values):
        return f"{feature_name} contains NaN or Inf."

    for item in _to_list(value):
        if isinstance(item, (list, tuple)):
            for nested in _to_list(item):
                if _coerce_float(nested) is None:
                    return f"{feature_name} contains non-numeric data."
            continue
        if _coerce_float(item) is None:
            return f"{feature_name} contains non-numeric data."
    return None


def _inspect_nonnumeric_node_features(
    node_features: Mapping[str, Any],
    *,
    include_metadata: bool = False,
) -> dict[str, str]:
    num_nodes = _num_nodes_from_node_features(node_features)
    nonnumeric: dict[str, str] = {}
    for name, value in node_features.items():
        if not include_metadata and name.startswith("_"):
            continue
        reason = _validate_numeric_node_feature(name, value, num_nodes)
        if reason is not None:
            nonnumeric[name] = reason
    return nonnumeric


def infer_is_mutation_from_diff(
    node_features: dict[str, Any],
    eps: float = 1e-12,
    return_metadata: bool = False,
) -> list[int] | dict[str, Any]:
    """Infer a binary mutation-node mask from available diff_* features."""
    available = [probe for probe in DIFF_PROBES_FOR_MUTATION if probe in node_features]
    num_nodes = _num_nodes_from_node_features(node_features)
    warnings: list[str] = []
    skipped_probes: dict[str, str] = {}
    used_numeric_probes: list[str] = []

    if not available:
        warning = "No diff_* mutation probes available; inferred is_mutation defaults to all zeros."
        LOGGER.warning(warning)
        metadata = {
            "is_mutation": [0] * num_nodes,
            "warnings": [warning],
            "used_numeric_probes": used_numeric_probes,
            "skipped_probes": skipped_probes,
            "nonnumeric_node_features": [],
        }
        return metadata if return_metadata else metadata["is_mutation"]

    mutation_mask = [0] * num_nodes
    for probe in available:
        values, reason = _extract_numeric_node_vector(probe, node_features[probe], num_nodes)
        if values is None:
            skipped_probes[probe] = reason or f"{probe} is not usable for mutation inference."
            warning = f"{probe} omitted from mutation inference: {skipped_probes[probe]}"
            LOGGER.warning(warning)
            warnings.append(warning)
            continue
        used_numeric_probes.append(probe)
        for index, numeric in enumerate(values):
            if abs(numeric) > eps:
                mutation_mask[index] = 1

    if not used_numeric_probes:
        warning = "No numeric diff_* probes available; inferred is_mutation defaults to all zeros."
        LOGGER.warning(warning)
        warnings.append(warning)

    metadata = {
        "is_mutation": mutation_mask,
        "warnings": warnings,
        "used_numeric_probes": used_numeric_probes,
        "skipped_probes": skipped_probes,
        "nonnumeric_node_features": sorted(skipped_probes.keys()),
    }
    return metadata if return_metadata else metadata["is_mutation"]


def detect_available_targets(graph_features: dict[str, Any]) -> dict[str, bool]:
    """Detect optional graph-level targets available in the case."""
    available = {
        "custom_structure_energy": "custom_structure_energy" in graph_features,
        "custom_complex_energy_phenotype": "custom_complex_energy_phenotype" in graph_features,
    }
    if not available["custom_complex_energy_phenotype"]:
        LOGGER.warning(
            "custom_complex_energy_phenotype not found; supervised phenotype training disabled."
        )
    return available


def is_wt_companion_case(case_key: str, file_name: str | None = None) -> bool:
    text = _normalize_case_text(case_key, file_name)
    return bool(_WT_PATTERN.search(text))


def is_truncation_case(
    case_key: str,
    file_name: str | None = None,
    graph_features: Mapping[str, Any] | None = None,
) -> bool:
    if graph_features and "is_truncation" in graph_features:
        if _extract_scalar_bool(graph_features["is_truncation"]):
            return True
    text = _normalize_case_text(case_key, file_name)
    return bool(_STOP_PATTERN.search(text))


def classify_case(
    case_key: str,
    file_name: str | None = None,
    graph_features: Mapping[str, Any] | None = None,
) -> str:
    if is_wt_companion_case(case_key, file_name=file_name):
        return CASE_KIND_WT_COMPANION
    if is_truncation_case(case_key, file_name=file_name, graph_features=graph_features):
        return CASE_KIND_TRUNCATION
    return CASE_KIND_MISSENSE


def list_var_features(node_features: Mapping[str, Any]) -> list[str]:
    return sorted(name for name in node_features if name.startswith("var_"))


def validate_encoder_feature_policy(
    *,
    node_features: Mapping[str, Any],
    selected_node_features: Sequence[str] | None,
    excluded_node_features: Sequence[str] | None,
    explicit_numeric_mappings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    selected = set(selected_node_features or [])
    excluded = set(excluded_node_features or [])
    mappings = explicit_numeric_mappings or {}
    nonnumeric_reasons = _inspect_nonnumeric_node_features(node_features)
    recommended_excluded_features = sorted(nonnumeric_reasons.keys())

    def has_explicit_mapping(feature_name: str) -> bool:
        direct = mappings.get(feature_name)
        if isinstance(direct, Mapping):
            return bool(direct.get("enabled", False))
        if direct is True:
            return True
        for mapping in mappings.values():
            if isinstance(mapping, Mapping):
                if mapping.get("source") == feature_name and mapping.get("enabled", False):
                    return True
        return False

    for feature_name, reason in sorted(nonnumeric_reasons.items()):
        if feature_name in selected and feature_name not in excluded and not has_explicit_mapping(feature_name):
            errors.append(
                f"{feature_name} is selected for encoder input but is non-numeric and has no explicit mapping: {reason}"
            )
            continue
        if feature_name in selected and has_explicit_mapping(feature_name):
            warnings.append(
                f"{feature_name} is non-numeric in raw HDF5 and requires explicit mapped feature handling."
            )
            continue
        warnings.append(
            f"{feature_name} is non-numeric in raw HDF5 and is recommended for exclusion from the encoder base."
        )

    return {
        "warnings": warnings,
        "errors": errors,
        "nonnumeric_node_features": sorted(nonnumeric_reasons.keys()),
        "nonnumeric_reasons": nonnumeric_reasons,
        "recommended_excluded_features": recommended_excluded_features,
    }


def validate_edge_index(edge_index: Any, num_nodes: int) -> dict[str, Any]:
    """Validate edge_features/_index with observed orientation (E, 2)."""
    shape = _shape_of(edge_index)
    if len(shape) != 2 or shape[1] != 2:
        raise ValueError(
            "edge_features/_index must have observed orientation (E, 2); "
            f"got shape {shape!r}."
        )

    rows = _to_list(edge_index)
    num_edges = len(rows)
    for edge_position, row in enumerate(rows):
        row_values = _to_list(row)
        if len(row_values) != 2:
            raise ValueError(
                f"edge_features/_index row {edge_position} must contain exactly 2 indices."
            )
        for index_value in row_values:
            if not isinstance(index_value, int):
                numeric = _coerce_float(index_value)
                if numeric is None or int(numeric) != numeric:
                    raise TypeError("edge_features/_index must contain integer-compatible values.")
                index_value = int(numeric)
            if index_value < 0 or index_value >= num_nodes:
                raise ValueError(
                    f"edge_features/_index contains out-of-range node index {index_value} "
                    f"for num_nodes={num_nodes}."
                )

    return {
        "shape": [shape[0], shape[1]],
        "orientation": "E,2",
        "pyg_shape": [2, num_edges],
        "num_edges": num_edges,
        "conversion_note": "Convert edge_features/_index from (E, 2) to (2, E) for PyG.",
    }


def _validate_feature_group(
    features: Mapping[str, Any],
    expected_length: int | None,
    group_name: str,
) -> list[str]:
    errors: list[str] = []
    for name, value in features.items():
        if name.startswith("_"):
            continue
        shape = _shape_of(value)
        if expected_length is not None and shape:
            if int(shape[0]) != expected_length:
                errors.append(
                    f"{group_name}.{name} has incompatible first dimension {shape[0]} "
                    f"(expected {expected_length})."
                )
        numeric_values = _iter_numeric_values(value)
        for numeric in numeric_values:
            if math.isnan(numeric) or math.isinf(numeric):
                errors.append(f"{group_name}.{name} contains NaN or Inf.")
                break
    return errors


def audit_hdf5_case(
    *,
    source_path: str,
    case_key: str,
    node_features: Mapping[str, Any],
    edge_features: Mapping[str, Any],
    graph_features: Mapping[str, Any],
    eps: float = 1e-12,
    require_explicit_is_mutation: bool = False,
    require_custom_complex_energy_phenotype: bool = False,
    diff_probes: Sequence[str] | None = None,
    expected_missense_mutation_nodes: int = 1,
    expected_wt_mutation_nodes: int = 0,
    expected_truncation_mutation_nodes: int = 0,
) -> dict[str, Any]:
    """Audit a single HDF5 case without mutating the source file."""
    warnings: list[str] = []
    errors: list[str] = []

    if diff_probes is not None:
        missing_requested = [probe for probe in diff_probes if probe not in node_features]
        if missing_requested:
            warnings.append(
                "Requested diff probes missing from node_features: "
                + ", ".join(sorted(missing_requested))
            )

    num_nodes = _num_nodes_from_node_features(node_features)
    if num_nodes <= 0:
        errors.append("node_features do not expose a valid number of nodes.")

    feature_policy = validate_encoder_feature_policy(
        node_features=node_features,
        selected_node_features=[],
        excluded_node_features=[],
        explicit_numeric_mappings={},
    )
    warnings.extend(feature_policy["warnings"])

    errors.extend(_validate_feature_group(node_features, num_nodes or None, "node_features"))
    errors.extend(_validate_feature_group(graph_features, None, "graph_features"))

    edge_index_info: dict[str, Any] | None = None
    if "_index" not in edge_features:
        errors.append("edge_features/_index is missing.")
        num_edges = 0
    else:
        try:
            edge_index_info = validate_edge_index(edge_features["_index"], num_nodes)
            num_edges = int(edge_index_info["num_edges"])
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            num_edges = 0
    errors.extend(_validate_feature_group(edge_features, num_edges or None, "edge_features"))

    case_kind = classify_case(case_key, file_name=Path(source_path).name, graph_features=graph_features)
    available_targets = detect_available_targets(dict(graph_features))
    if not available_targets["custom_complex_energy_phenotype"]:
        warnings.append(
            "custom_complex_energy_phenotype not found; supervised phenotype training disabled."
        )
    if require_custom_complex_energy_phenotype and not available_targets["custom_complex_energy_phenotype"]:
        errors.append("custom_complex_energy_phenotype is required by configuration but missing.")

    explicit_is_mutation_present = "is_mutation" in node_features
    if require_explicit_is_mutation and not explicit_is_mutation_present:
        errors.append("Explicit is_mutation dataset is required by configuration but missing.")

    inference = infer_is_mutation_from_diff(dict(node_features), eps=eps, return_metadata=True)
    inferred_is_mutation = inference["is_mutation"]
    warnings.extend(inference["warnings"])

    if not any(probe in node_features for probe in DIFF_PROBES_FOR_MUTATION):
        if case_kind == CASE_KIND_MISSENSE:
            errors.append("Missense case requires diff_* probes to infer a mutation node.")

    mutation_node_count = sum(inferred_is_mutation)
    expected_counts = {
        CASE_KIND_MISSENSE: expected_missense_mutation_nodes,
        CASE_KIND_WT_COMPANION: expected_wt_mutation_nodes,
        CASE_KIND_TRUNCATION: expected_truncation_mutation_nodes,
    }
    expected_count = expected_counts[case_kind]
    if mutation_node_count != expected_count:
        errors.append(
            f"{case_kind} case expects {expected_count} mutated nodes, got {mutation_node_count}."
        )

    return {
        "source_path": source_path,
        "case_key": case_key,
        "case_kind": case_kind,
        "valid": not errors,
        "warnings": warnings,
        "errors": errors,
        "node_feature_names": sorted(node_features.keys()),
        "edge_feature_names": sorted(edge_features.keys()),
        "graph_feature_names": sorted(graph_features.keys()),
        "available_var_features": list_var_features(node_features),
        "available_targets": available_targets,
        "nonnumeric_node_features": sorted(
            set(feature_policy["nonnumeric_node_features"]) | set(inference["nonnumeric_node_features"])
        ),
        "recommended_excluded_features": sorted(
            set(feature_policy["recommended_excluded_features"]) | set(inference["nonnumeric_node_features"])
        ),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "explicit_is_mutation_present": explicit_is_mutation_present,
        "inferred_is_mutation": inferred_is_mutation,
        "mutation_node_count": mutation_node_count,
        "used_numeric_mutation_probes": inference["used_numeric_probes"],
        "skipped_mutation_probes": inference["skipped_probes"],
        "edge_index": edge_index_info
        or {
            "shape": None,
            "orientation": "invalid",
            "pyg_shape": None,
            "conversion_note": "Convert edge_features/_index from (E, 2) to (2, E) for PyG.",
        },
    }
