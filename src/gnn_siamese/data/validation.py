"""Validation helpers for the observed HDF5 schema."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from glob import glob
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
_CASE_KEY_PATTERN = re.compile(
    r"^residue-srv:(?P<chain_id>[^:]+):(?P<position>\d+):(?P<wt_aa_full>[A-Za-z]+)->"
    r"(?P<mut_aa_full>[A-Za-z]+):(?P<variant_suffix>.+)$"
)


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


def _dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype)
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return "empty"
        return _dtype_of(value[0])
    return type(value).__name__


def _shape_as_list(value: Any) -> list[int]:
    return [int(dimension) for dimension in _shape_of(value)]


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


def _extract_scalar_int(value: Any) -> int | None:
    items = _to_list(value)
    if not items:
        return None
    numeric = _coerce_float(items[0])
    if numeric is None or int(numeric) != numeric:
        return None
    return int(numeric)


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


def parse_case_key(case_key: str) -> dict[str, Any]:
    match = _CASE_KEY_PATTERN.match(case_key)
    if not match:
        return {
            "valid": False,
            "chain_id": None,
            "position": None,
            "wt_aa_full": None,
            "mut_aa_full": None,
            "variant_suffix": None,
        }
    return {
        "valid": True,
        "chain_id": match.group("chain_id"),
        "position": int(match.group("position")),
        "wt_aa_full": match.group("wt_aa_full"),
        "mut_aa_full": match.group("mut_aa_full"),
        "variant_suffix": match.group("variant_suffix"),
    }


def summarize_feature_group(features: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, value in sorted(features.items()):
        summary[name] = {
            "shape": _shape_as_list(value),
            "dtype": _dtype_of(value),
        }
    return summary


def inspect_availability_masks(node_features: Mapping[str, Any]) -> dict[str, Any]:
    mask_names = sorted(
        name for name in node_features if name == "mask_diff" or name.startswith("mask_diff_")
    )
    masks: dict[str, dict[str, Any]] = {}
    for name in mask_names:
        values, reason = _extract_numeric_node_vector(
            name,
            node_features[name],
            _num_nodes_from_node_features(node_features),
        )
        if values is None:
            masks[name] = {"shape": _shape_as_list(node_features[name]), "error": reason}
            continue
        available = sum(1 for value in values if value != 0.0)
        unavailable = len(values) - available
        masks[name] = {
            "shape": _shape_as_list(node_features[name]),
            "available": available,
            "unavailable": unavailable,
            "available_fraction": (available / len(values)) if values else 0.0,
        }
    return {
        "mask_feature_names": mask_names,
        "masks": masks,
    }


def extract_explicit_is_mutation(
    node_features: Mapping[str, Any],
    num_nodes: int,
) -> tuple[list[int] | None, str | None]:
    if "is_mutation" not in node_features:
        return None, None
    values, reason = _extract_numeric_node_vector("is_mutation", node_features["is_mutation"], num_nodes)
    if values is None:
        return None, reason
    mask: list[int] = []
    for index, value in enumerate(values):
        if value not in (0.0, 1.0):
            return None, f"is_mutation contains non-binary value {value} at node index {index}."
        mask.append(int(value))
    return mask, None


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
    """Validate edge_features/_index accepting stored orientations (E, 2) and (2, E)."""
    shape = _shape_of(edge_index)
    if len(shape) != 2:
        raise ValueError(
            "edge_features/_index must be a 2D tensor with orientation (E, 2) or (2, E); "
            f"got shape {shape!r}."
        )

    rows = _to_list(edge_index)
    orientation: str
    normalized_rows: list[list[Any]]
    if shape[1] == 2:
        orientation = "E,2"
        normalized_rows = [_to_list(row) for row in rows]
        num_edges = len(normalized_rows)
        pyg_shape = [2, num_edges]
        conversion_note = "Convert edge_features/_index from (E, 2) to (2, E) for PyG."
    elif shape[0] == 2:
        orientation = "2,E"
        num_edges = int(shape[1])
        normalized_rows = []
        first_row = _to_list(rows[0])
        second_row = _to_list(rows[1])
        if len(first_row) != len(second_row):
            raise ValueError("edge_features/_index rows in (2, E) orientation must have equal length.")
        for source, target in zip(first_row, second_row, strict=True):
            normalized_rows.append([source, target])
        pyg_shape = [2, num_edges]
        conversion_note = "edge_features/_index already matches PyG orientation (2, E)."
    else:
        raise ValueError(
            "edge_features/_index must use orientation (E, 2) or (2, E); "
            f"got shape {shape!r}."
        )

    for edge_position, row_values in enumerate(normalized_rows):
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
        "shape": [int(shape[0]), int(shape[1])],
        "orientation": orientation,
        "pyg_shape": pyg_shape,
        "num_edges": num_edges,
        "conversion_note": conversion_note,
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
    parsed_case_key = parse_case_key(case_key)
    if not parsed_case_key["valid"]:
        errors.append("case_key does not match the documented residue-srv schema.")

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
    explicit_is_mutation, explicit_is_mutation_error = extract_explicit_is_mutation(node_features, num_nodes)
    if explicit_is_mutation_error is not None:
        errors.append(explicit_is_mutation_error)

    inference = infer_is_mutation_from_diff(dict(node_features), eps=eps, return_metadata=True)
    inferred_is_mutation = inference["is_mutation"]
    warnings.extend(inference["warnings"])
    mutation_mask_source = "inferred"
    effective_is_mutation = inferred_is_mutation
    if explicit_is_mutation is not None:
        mutation_mask_source = "explicit"
        effective_is_mutation = explicit_is_mutation
        if explicit_is_mutation != inferred_is_mutation and any(
            probe in node_features for probe in DIFF_PROBES_FOR_MUTATION
        ):
            warnings.append("Explicit is_mutation differs from diff_* reconstruction.")

    if not any(probe in node_features for probe in DIFF_PROBES_FOR_MUTATION):
        if case_kind == CASE_KIND_MISSENSE:
            errors.append("Missense case requires diff_* probes to infer a mutation node.")

    mutation_node_count = sum(effective_is_mutation)
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

    graph_num_nodes = _extract_scalar_int(graph_features.get("graph_num_nodes"))
    if graph_num_nodes is not None and graph_num_nodes != num_nodes:
        errors.append(
            f"graph_features.graph_num_nodes={graph_num_nodes} does not match inferred num_nodes={num_nodes}."
        )
    graph_num_edges = _extract_scalar_int(graph_features.get("graph_num_edges"))
    if graph_num_edges is not None and graph_num_edges != num_edges:
        errors.append(
            f"graph_features.graph_num_edges={graph_num_edges} does not match inferred num_edges={num_edges}."
        )

    availability_masks = inspect_availability_masks(node_features)

    return {
        "source_path": source_path,
        "case_key": case_key,
        "parsed_case_key": parsed_case_key,
        "case_kind": case_kind,
        "valid": not errors,
        "warnings": warnings,
        "errors": errors,
        "rejection_reasons": list(errors),
        "node_feature_names": sorted(node_features.keys()),
        "edge_feature_names": sorted(edge_features.keys()),
        "graph_feature_names": sorted(graph_features.keys()),
        "node_feature_summary": summarize_feature_group(node_features),
        "edge_feature_summary": summarize_feature_group(edge_features),
        "graph_feature_summary": summarize_feature_group(graph_features),
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
        "explicit_is_mutation": explicit_is_mutation,
        "inferred_is_mutation": inferred_is_mutation,
        "effective_is_mutation": effective_is_mutation,
        "mutation_mask_source": mutation_mask_source,
        "mutation_node_count": mutation_node_count,
        "used_numeric_mutation_probes": inference["used_numeric_probes"],
        "skipped_mutation_probes": inference["skipped_probes"],
        "availability_masks": availability_masks,
        "edge_index": edge_index_info
        or {
            "shape": None,
            "orientation": "invalid",
            "pyg_shape": None,
            "conversion_note": "Convert edge_features/_index from (E, 2) to (2, E) for PyG.",
        },
    }


def audit_mut_wt_pairing(
    mut_rows: Sequence[dict[str, Any]],
    wt_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    wt_by_signature: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in wt_rows:
        parsed = row.get("parsed_case_key", {})
        signature = (parsed.get("chain_id"), parsed.get("position"), parsed.get("wt_aa_full"))
        wt_by_signature.setdefault(signature, []).append(row)

    matched = 0
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for row in mut_rows:
        if row.get("case_kind") != CASE_KIND_MISSENSE:
            continue
        parsed = row.get("parsed_case_key", {})
        signature = (parsed.get("chain_id"), parsed.get("position"), parsed.get("wt_aa_full"))
        candidates = wt_by_signature.get(signature, [])
        if not candidates:
            row["valid"] = False
            row.setdefault("errors", []).append("No WT companion found for mutant position/wt_aa.")
            row.setdefault("rejection_reasons", []).append("No WT companion found for mutant position/wt_aa.")
            missing.append(
                {
                    "case_key": row["case_key"],
                    "chain_id": parsed.get("chain_id"),
                    "position": parsed.get("position"),
                    "wt_aa_full": parsed.get("wt_aa_full"),
                }
            )
            continue
        if len(candidates) > 1:
            row["valid"] = False
            row.setdefault("errors", []).append("Ambiguous WT companion pairing for mutant position/wt_aa.")
            row.setdefault("rejection_reasons", []).append(
                "Ambiguous WT companion pairing for mutant position/wt_aa."
            )
            ambiguous.append(
                {
                    "case_key": row["case_key"],
                    "chain_id": parsed.get("chain_id"),
                    "position": parsed.get("position"),
                    "wt_aa_full": parsed.get("wt_aa_full"),
                    "candidate_count": len(candidates),
                }
            )
            continue
        matched += 1

    return {
        "mutant_cases_checked": sum(1 for row in mut_rows if row.get("case_kind") == CASE_KIND_MISSENSE),
        "wt_cases_checked": len(wt_rows),
        "matched_pairs": matched,
        "missing_wt_companion": missing,
        "ambiguous_wt_companion": ambiguous,
        "coverage_complete": not missing and not ambiguous,
    }


def expand_hdf5_inputs(patterns: Sequence[str]) -> list[Path]:
    expanded: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob(pattern))
        expanded.extend(matches or [Path(pattern)])

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _read_hdf5_group(group: Any) -> dict[str, Any]:
    return {name: dataset[()] for name, dataset in group.items()}


def audit_hdf5_file(
    path: Path,
    *,
    dataset_role: str,
    diff_probes: list[str],
    require_explicit_is_mutation: bool,
    require_custom_complex_energy_phenotype: bool,
    expected_missense_mutation_nodes: int,
    expected_wt_mutation_nodes: int,
    expected_truncation_mutation_nodes: int,
) -> list[dict[str, Any]]:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("audit_hdf5_file requires h5py to inspect HDF5 inputs.") from exc

    records: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for case_key in sorted(handle.keys()):
            group = handle[case_key]
            if not isinstance(group, h5py.Group):
                continue
            if not {"node_features", "edge_features", "graph_features"}.issubset(group.keys()):
                records.append(
                    {
                        "source_path": str(path),
                        "dataset_role": dataset_role,
                        "case_key": case_key,
                        "case_kind": "unknown",
                        "valid": False,
                        "warnings": [],
                        "errors": ["Root group does not contain node_features, edge_features and graph_features."],
                        "rejection_reasons": [
                            "Root group does not contain node_features, edge_features and graph_features."
                        ],
                        "node_feature_names": [],
                        "edge_feature_names": [],
                        "graph_feature_names": [],
                        "node_feature_summary": {},
                        "edge_feature_summary": {},
                        "graph_feature_summary": {},
                        "available_var_features": [],
                        "nonnumeric_node_features": [],
                        "recommended_excluded_features": [],
                        "available_targets": {
                            "custom_structure_energy": False,
                            "custom_complex_energy_phenotype": False,
                        },
                        "availability_masks": {"mask_feature_names": [], "masks": {}},
                        "num_nodes": 0,
                        "num_edges": 0,
                        "explicit_is_mutation_present": False,
                        "explicit_is_mutation": None,
                        "inferred_is_mutation": [],
                        "effective_is_mutation": [],
                        "mutation_mask_source": "missing",
                        "mutation_node_count": 0,
                        "parsed_case_key": {
                            "valid": False,
                            "chain_id": None,
                            "position": None,
                            "wt_aa_full": None,
                            "mut_aa_full": None,
                            "variant_suffix": None,
                        },
                        "edge_index": {
                            "shape": None,
                            "orientation": "invalid",
                            "pyg_shape": None,
                            "conversion_note": "Convert edge_features/_index from (E, 2) to (2, E) for PyG.",
                        },
                    }
                )
                continue

            record = audit_hdf5_case(
                source_path=str(path),
                case_key=case_key,
                node_features=_read_hdf5_group(group["node_features"]),
                edge_features=_read_hdf5_group(group["edge_features"]),
                graph_features=_read_hdf5_group(group["graph_features"]),
                require_explicit_is_mutation=require_explicit_is_mutation,
                require_custom_complex_energy_phenotype=require_custom_complex_energy_phenotype,
                diff_probes=diff_probes,
                expected_missense_mutation_nodes=expected_missense_mutation_nodes,
                expected_wt_mutation_nodes=expected_wt_mutation_nodes,
                expected_truncation_mutation_nodes=expected_truncation_mutation_nodes,
            )
            record["dataset_role"] = dataset_role
            records.append(record)
    return records


def write_audit_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_audit_csv(path: Path, rows: list[dict[str, Any]], *, rejected_only: bool = False) -> None:
    filtered = [row for row in rows if not row["valid"]] if rejected_only else rows
    fieldnames = [
        "source_path",
        "dataset_role",
        "case_key",
        "case_kind",
        "valid",
        "num_nodes",
        "num_edges",
        "mutation_node_count",
        "mutation_mask_source",
        "explicit_is_mutation_present",
        "custom_structure_energy",
        "custom_complex_energy_phenotype",
        "available_var_features",
        "nonnumeric_node_features",
        "recommended_excluded_features",
        "mask_features",
        "warnings",
        "errors",
        "edge_index_shape",
        "edge_index_orientation",
        "pyg_edge_index_shape",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered:
            writer.writerow(
                {
                    "source_path": row["source_path"],
                    "dataset_role": row.get("dataset_role", "unknown"),
                    "case_key": row["case_key"],
                    "case_kind": row["case_kind"],
                    "valid": row["valid"],
                    "num_nodes": row["num_nodes"],
                    "num_edges": row["num_edges"],
                    "mutation_node_count": row["mutation_node_count"],
                    "mutation_mask_source": row.get("mutation_mask_source", "unknown"),
                    "explicit_is_mutation_present": row["explicit_is_mutation_present"],
                    "custom_structure_energy": row["available_targets"]["custom_structure_energy"],
                    "custom_complex_energy_phenotype": row["available_targets"][
                        "custom_complex_energy_phenotype"
                    ],
                    "available_var_features": ";".join(row["available_var_features"]),
                    "nonnumeric_node_features": ";".join(row["nonnumeric_node_features"]),
                    "recommended_excluded_features": ";".join(row["recommended_excluded_features"]),
                    "mask_features": ";".join(
                        row.get("availability_masks", {}).get("mask_feature_names", [])
                    ),
                    "warnings": " | ".join(row["warnings"]),
                    "errors": " | ".join(row["errors"]),
                    "edge_index_shape": row["edge_index"]["shape"],
                    "edge_index_orientation": row["edge_index"]["orientation"],
                    "pyg_edge_index_shape": row["edge_index"]["pyg_shape"],
                }
            )


def build_summary_by_reason(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    reason_counts: dict[str, int] = {}
    for row in rows:
        for field_name in ("errors", "warnings"):
            for reason in row.get(field_name, []):
                reason_text = str(reason).strip()
                if not reason_text:
                    continue
                reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1

    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items())
    ]


def write_summary_by_reason_csv(path: Path, summary_rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = ["reason", "count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "reason": row["reason"],
                    "count": row["count"],
                }
            )


def build_feature_summary(rows: list[dict[str, Any]], *, diff_probes: list[str]) -> dict[str, Any]:
    node_features = sorted({name for row in rows for name in row["node_feature_names"]})
    edge_features = sorted({name for row in rows for name in row["edge_feature_names"]})
    graph_features = sorted({name for row in rows for name in row["graph_feature_names"]})
    var_features = sorted({name for row in rows for name in row["available_var_features"]})
    nonnumeric_node_features = sorted({name for row in rows for name in row["nonnumeric_node_features"]})
    recommended_excluded_features = sorted(
        {name for row in rows for name in row["recommended_excluded_features"]}
    )

    def inventory_for(group_name: str) -> dict[str, list[str]]:
        feature_names = sorted({feature_name for row in rows for feature_name in row.get(group_name, {})})
        return {
            feature_name: sorted(
                {
                    f"shape={tuple(summary['shape'])},dtype={summary['dtype']}"
                    for row in rows
                    for candidate_name, summary in row.get(group_name, {}).items()
                    if candidate_name == feature_name
                }
            )
            for feature_name in feature_names
        }

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "graph_features": graph_features,
        "var_features": var_features,
        "nonnumeric_node_features": nonnumeric_node_features,
        "recommended_excluded_features": recommended_excluded_features,
        "mask_features": sorted(
            {
                mask_name
                for row in rows
                for mask_name in row.get("availability_masks", {}).get("mask_feature_names", [])
            }
        ),
        "detected_targets": {
            "custom_structure_energy": any(
                row["available_targets"]["custom_structure_energy"] for row in rows
            ),
            "custom_complex_energy_phenotype": any(
                row["available_targets"]["custom_complex_energy_phenotype"] for row in rows
            ),
        },
        "mutation_detection": {
            "diff_probes_checked": diff_probes,
            "edge_index_supported_orientations": ["(E, 2)", "(2, E)"],
            "pyg_conversion": "(2, E)",
        },
        "shape_dtype_inventory": {
            "node_features": inventory_for("node_feature_summary"),
            "edge_features": inventory_for("edge_feature_summary"),
            "graph_features": inventory_for("graph_feature_summary"),
        },
    }
