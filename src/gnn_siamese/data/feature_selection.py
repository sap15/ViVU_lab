"""Feature selection helpers driven by config and documented schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class FeatureSelectionError(ValueError):
    """Base error for feature selection failures."""


class MissingFeatureGroupError(FeatureSelectionError):
    """Raised when a requested feature group does not exist in config."""


class MissingSchemaFeatureError(FeatureSelectionError):
    """Raised when a requested feature is not available in the schema."""


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeatureSelectionError(f"{field_name} must be a mapping, got {type(value).__name__}.")
    return value


def _ensure_sequence_of_strings(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FeatureSelectionError(f"{field_name} must be a sequence of strings.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise FeatureSelectionError(f"{field_name} must contain only strings.")
        items.append(item)
    return items


def _unique_in_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _schema_feature_names(schema: Mapping[str, Any], *, feature_kind: str) -> list[str]:
    graph_layout = _require_mapping(schema.get("graph_layout"), field_name="schema.graph_layout")
    group_name = {
        "node": "node_features",
        "edge": "edge_features",
        "graph": "graph_features",
    }.get(feature_kind)
    if group_name is None:
        raise FeatureSelectionError(f"Unsupported feature_kind {feature_kind!r}.")

    feature_group = _require_mapping(
        graph_layout.get(group_name),
        field_name=f"schema.graph_layout.{group_name}",
    )
    feature_datasets = _require_mapping(
        feature_group.get("feature_datasets"),
        field_name=f"schema.graph_layout.{group_name}.feature_datasets",
    )
    return list(feature_datasets.keys())


def _requested_group_names(config: Mapping[str, Any], *, feature_kind: str) -> list[str]:
    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    group_list_key = {
        "node": "node_groups",
        "edge": "edge_groups",
    }.get(feature_kind)
    if group_list_key is None:
        raise FeatureSelectionError(f"Unsupported feature_kind {feature_kind!r}.")
    return _ensure_sequence_of_strings(
        features_cfg.get(group_list_key, []),
        field_name=f"config.features.{group_list_key}",
    )


def _collect_group_feature_names(config: Mapping[str, Any], *, feature_kind: str) -> list[str]:
    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    group_names = _requested_group_names(config, feature_kind=feature_kind)
    selected: list[str] = []

    for group_name in group_names:
        if group_name not in features_cfg:
            raise MissingFeatureGroupError(
                f"Requested {feature_kind} feature group {group_name!r} is not defined in config.features."
            )
        group_cfg = _require_mapping(
            features_cfg[group_name],
            field_name=f"config.features.{group_name}",
        )
        if group_cfg.get("enabled") is False:
            raise MissingFeatureGroupError(
                f"Requested {feature_kind} feature group {group_name!r} is disabled in config.features."
            )
        names = _ensure_sequence_of_strings(
            group_cfg.get("names", []),
            field_name=f"config.features.{group_name}.names",
        )
        if not names:
            raise MissingFeatureGroupError(
                f"Requested {feature_kind} feature group {group_name!r} defines no feature names."
            )
        selected.extend(names)

    return _unique_in_order(selected)


def _target_feature_names(config: Mapping[str, Any]) -> set[str]:
    targets_cfg = config.get("targets")
    if not isinstance(targets_cfg, Mapping):
        return set()
    target_names: set[str] = set()
    for value in targets_cfg.values():
        if not isinstance(value, Mapping):
            continue
        name = value.get("name")
        if isinstance(name, str) and name:
            target_names.add(name)
    return target_names


def _excluded_node_feature_names(config: Mapping[str, Any]) -> set[str]:
    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    excluded = set(
        _ensure_sequence_of_strings(
            features_cfg.get("excluded_from_encoder_base", []),
            field_name="config.features.excluded_from_encoder_base",
        )
    )
    excluded.update(
        _ensure_sequence_of_strings(
            features_cfg.get("node_metadata", []),
            field_name="config.features.node_metadata",
        )
    )
    excluded.update(
        _ensure_sequence_of_strings(
            features_cfg.get("confounders", []),
            field_name="config.features.confounders",
        )
    )
    excluded.update(_target_feature_names(config))
    return excluded


def _excluded_edge_feature_names(config: Mapping[str, Any]) -> set[str]:
    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    return set(
        _ensure_sequence_of_strings(
            features_cfg.get("edge_metadata", []),
            field_name="config.features.edge_metadata",
        )
    )


def resolve_node_feature_names(config: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """Resolve node features that enter the base encoder."""

    available = set(_schema_feature_names(schema, feature_kind="node"))
    requested = _collect_group_feature_names(config, feature_kind="node")
    missing = [name for name in requested if name not in available]
    if missing:
        raise MissingSchemaFeatureError(
            "Requested node feature(s) missing from schema: " + ", ".join(sorted(missing))
        )

    excluded = _excluded_node_feature_names(config)
    selected = [
        name
        for name in requested
        if not name.startswith("_") and not name.startswith("mask_") and name not in excluded
    ]
    return _unique_in_order(selected)


def resolve_edge_feature_names(config: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """Resolve edge features that enter the base encoder as edge_attr."""

    available = set(_schema_feature_names(schema, feature_kind="edge"))
    requested = _collect_group_feature_names(config, feature_kind="edge")
    missing = [name for name in requested if name not in available]
    if missing:
        raise MissingSchemaFeatureError(
            "Requested edge feature(s) missing from schema: " + ", ".join(sorted(missing))
        )

    excluded = _excluded_edge_feature_names(config)
    selected = [name for name in requested if not name.startswith("_") and name not in excluded]
    return _unique_in_order(selected)


def split_encoder_inputs_and_auxiliary_features(
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Split encoder inputs from auxiliary schema features."""

    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    node_available = _schema_feature_names(schema, feature_kind="node")
    edge_available = _schema_feature_names(schema, feature_kind="edge")
    graph_available = _schema_feature_names(schema, feature_kind="graph")

    encoder_node_features = resolve_node_feature_names(config, schema)
    encoder_edge_features = resolve_edge_feature_names(config, schema)

    mask_prefix = "mask_"
    diff_bioq_cfg = features_cfg.get("diff_bioq")
    if isinstance(diff_bioq_cfg, Mapping):
        candidate_prefix = diff_bioq_cfg.get("mask_prefix")
        if isinstance(candidate_prefix, str) and candidate_prefix:
            mask_prefix = candidate_prefix

    node_availability_masks: dict[str, str] = {}
    for name in node_available:
        if not name.startswith(mask_prefix):
            continue
        base_name = name[len(mask_prefix) :]
        node_availability_masks[base_name] = name

    auxiliary_graph_features = _unique_in_order(
        [
            name
            for name in graph_available
            if (
                name in _ensure_sequence_of_strings(
                    features_cfg.get("confounders", []),
                    field_name="config.features.confounders",
                )
                or name in _target_feature_names(config)
            )
        ]
    )

    return {
        "encoder_node_features": encoder_node_features,
        "encoder_edge_features": encoder_edge_features,
        "encoder_graph_features": [],
        "node_availability_masks": node_availability_masks,
        "node_metadata_features": [
            name
            for name in _ensure_sequence_of_strings(
                features_cfg.get("node_metadata", []),
                field_name="config.features.node_metadata",
            )
            if name in node_available
        ],
        "edge_metadata_features": [
            name
            for name in _ensure_sequence_of_strings(
                features_cfg.get("edge_metadata", []),
                field_name="config.features.edge_metadata",
            )
            if name in edge_available
        ],
        "auxiliary_graph_features": auxiliary_graph_features,
    }
