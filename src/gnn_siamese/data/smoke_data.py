"""Production smoke-test data generation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np

from gnn_siamese.data.pairing import parse_variant_signature


DEFAULT_SMOKE_VARIANTS: tuple[dict[str, Any], ...] = (
    {"position": 100, "wt_full": "Glycine", "mut_full": "Aspartate", "wt_aa": "G", "mut_aa": "D"},
    {"position": 101, "wt_full": "Glycine", "mut_full": "Serine", "wt_aa": "G", "mut_aa": "S"},
    {"position": 102, "wt_full": "Glycine", "mut_full": "Valine", "wt_aa": "G", "mut_aa": "V"},
    {"position": 103, "wt_full": "Glycine", "mut_full": "Alanine", "wt_aa": "G", "mut_aa": "A"},
    {"position": 563, "wt_full": "Cysteine", "mut_full": "Tryptophan", "wt_aa": "C", "mut_aa": "W"},
    {"position": 564, "wt_full": "Cysteine", "mut_full": "Phenylalanine", "wt_aa": "C", "mut_aa": "F"},
    {"position": 565, "wt_full": "Cysteine", "mut_full": "Tyrosine", "wt_aa": "C", "mut_aa": "Y"},
    {"position": 566, "wt_full": "Cysteine", "mut_full": "Leucine", "wt_aa": "C", "mut_aa": "L"},
)

_DEFAULT_NODE_METADATA = ("_chain_id", "_name", "_position")
_DEFAULT_EDGE_METADATA = ("_index",)
_DEFAULT_GRAPH_FEATURES = ("custom_structure_energy", "graph_num_nodes", "graph_num_edges")


@dataclass
class SmokeDataArtifacts:
    """Temporary smoke-test dataset artifacts kept alive for one run."""

    source: str
    mutants_hdf5: str
    wt_companion_hdf5: str
    schema_json: str
    split_json: str
    pair_count: int
    temp_dir: str
    _temporary_directory: TemporaryDirectory[str] | None = field(repr=False)

    def cleanup(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _resolve_path(config: Mapping[str, Any], raw_path: str | None) -> Path:
    if raw_path is None:
        raise ValueError("Required path is missing from config.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    config_path = config.get("__config_path__")
    if config_path is None:
        return candidate.resolve()
    return (Path(str(config_path)).resolve().parent / candidate).resolve()


def _unique_in_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _collect_feature_names(config: Mapping[str, Any], *, kind: str) -> list[str]:
    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    group_list_key = "node_groups" if kind == "node" else "edge_groups"
    group_names = features_cfg.get(group_list_key, [])
    if not isinstance(group_names, Sequence) or isinstance(group_names, (str, bytes)):
        raise ValueError(f"config.features.{group_list_key} must be a sequence.")
    selected: list[str] = []
    for group_name in group_names:
        if not isinstance(group_name, str):
            raise ValueError(f"config.features.{group_list_key} must contain only strings.")
        group_cfg = _require_mapping(features_cfg.get(group_name, {}), field_name=f"config.features.{group_name}")
        names = group_cfg.get("names", [])
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise ValueError(f"config.features.{group_name}.names must be a sequence.")
        selected.extend(str(name) for name in names)
    return _unique_in_order(selected)


def build_smoke_schema_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a schema payload compatible with the requested smoke-test config."""

    features_cfg = _require_mapping(config.get("features"), field_name="config.features")
    node_feature_names = list(_DEFAULT_NODE_METADATA) + _collect_feature_names(config, kind="node")
    edge_feature_names = list(_DEFAULT_EDGE_METADATA) + _collect_feature_names(config, kind="edge")
    graph_feature_names = list(_DEFAULT_GRAPH_FEATURES)

    diff_bioq_cfg = features_cfg.get("diff_bioq")
    if isinstance(diff_bioq_cfg, Mapping) and bool(diff_bioq_cfg.get("require_masks", False)):
        mask_prefix = str(diff_bioq_cfg.get("mask_prefix", "mask_"))
        for name in node_feature_names:
            if name.startswith("diff_"):
                node_feature_names.append(f"{mask_prefix}{name}")

    for name in features_cfg.get("node_metadata", []):
        if isinstance(name, str):
            node_feature_names.append(name)
    for name in features_cfg.get("edge_metadata", []):
        if isinstance(name, str):
            edge_feature_names.append(name)
    for name in features_cfg.get("confounders", []):
        if isinstance(name, str):
            graph_feature_names.append(name)
    targets_cfg = config.get("targets", {})
    if isinstance(targets_cfg, Mapping):
        for target_cfg in targets_cfg.values():
            if not isinstance(target_cfg, Mapping):
                continue
            name = target_cfg.get("name")
            if isinstance(name, str) and name:
                graph_feature_names.append(name)

    node_feature_names = _unique_in_order(node_feature_names)
    edge_feature_names = _unique_in_order(edge_feature_names)
    graph_feature_names = _unique_in_order(graph_feature_names)

    return {
        "schema_name": "gnn_siamese_smoke_schema",
        "schema_version": "1.0.0",
        "graph_layout": {
            "node_features": {"feature_datasets": {name: {} for name in node_feature_names}},
            "edge_features": {"feature_datasets": {name: {} for name in edge_feature_names}},
            "graph_features": {"feature_datasets": {name: {} for name in graph_feature_names}},
        },
    }


def write_smoke_schema_json(path: str | Path, config: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_smoke_schema_payload(config), indent=2, sort_keys=True), encoding="utf-8")


def _create_graph_group(
    handle: h5py.File,
    *,
    graph_key: str,
    residue_name: bytes,
    position: int,
    variant_index: int,
    is_wt: bool,
    node_feature_names: Sequence[str],
    edge_feature_names: Sequence[str],
    graph_feature_names: Sequence[str],
) -> None:
    num_nodes = 4
    mutation_node_index = variant_index % num_nodes
    graph = handle.create_group(graph_key)
    node_group = graph.create_group("node_features")
    edge_group = graph.create_group("edge_features")
    graph_group = graph.create_group("graph_features")
    node_group.create_dataset(
        "res_id",
        data=np.arange(position, position + num_nodes, dtype=np.int64),
    )

    edge_index = np.asarray([[0, 1], [1, 2], [2, 3], [0, 2]], dtype=np.int64)
    num_edges = int(edge_index.shape[0])
    node_positions = np.stack(
        [
            np.linspace(float(position), float(position) + 1.5, num_nodes, dtype=np.float32),
            np.linspace(0.0, 0.3 * (variant_index + 1), num_nodes, dtype=np.float32),
            np.linspace(1.0, 1.0 + 0.1 * variant_index, num_nodes, dtype=np.float32),
        ],
        axis=1,
    )

    numeric_node_defaults: dict[str, np.ndarray] = {}
    for feature_offset, feature_name in enumerate(node_feature_names):
        base = np.linspace(0.1, 0.1 * num_nodes, num_nodes, dtype=np.float32)
        numeric_node_defaults[feature_name] = base + np.float32(variant_index * 0.01 + feature_offset * 0.001)

    diff_vector = np.zeros(num_nodes, dtype=np.float32)
    if not is_wt:
        diff_vector[mutation_node_index] = np.float32(1.0 + 0.05 * variant_index)

    for feature_name in node_feature_names:
        if feature_name == "_chain_id":
            node_group.create_dataset(feature_name, data=np.asarray([b"A"] * num_nodes))
            continue
        if feature_name == "_name":
            node_group.create_dataset(feature_name, data=np.asarray([residue_name] * num_nodes))
            continue
        if feature_name == "_position":
            node_group.create_dataset(feature_name, data=node_positions)
            continue
        if feature_name.startswith("mask_diff_"):
            node_group.create_dataset(feature_name, data=np.ones(num_nodes, dtype=np.float32))
            continue
        if feature_name.startswith("diff_"):
            values = diff_vector.copy()
            if feature_name == "diff_polarity":
                values = diff_vector * np.float32(0.5)
            node_group.create_dataset(feature_name, data=values)
            continue
        node_group.create_dataset(feature_name, data=numeric_node_defaults[feature_name])

    edge_defaults: dict[str, np.ndarray] = {
        "distance": np.asarray([3.2, 4.1, 5.0, 4.4], dtype=np.float32) + np.float32(variant_index * 0.01),
        "covalent": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        "electrostatic": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "vanderwaals": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "seq_sep": np.asarray([1.0, 1.0, 1.0, 2.0], dtype=np.float32),
    }
    for feature_name in edge_feature_names:
        if feature_name == "_index":
            edge_group.create_dataset(feature_name, data=edge_index)
            continue
        values = edge_defaults.get(feature_name)
        if values is None:
            values = np.linspace(0.2, 0.2 * num_edges, num_edges, dtype=np.float32)
        edge_group.create_dataset(feature_name, data=values)

    graph_defaults: dict[str, float] = {
        "custom_structure_energy": float((1.0 if is_wt else 5.0) + position / 1000.0),
        "graph_num_nodes": float(num_nodes),
        "graph_num_edges": float(num_edges),
    }
    for feature_name in graph_feature_names:
        graph_group.create_dataset(feature_name, data=float(graph_defaults.get(feature_name, 0.0)))


def create_synthetic_mut_wt_hdf5(
    mutant_path: str | Path,
    wt_path: str | Path,
    config: Mapping[str, Any],
    *,
    variants: Sequence[Mapping[str, Any]] = DEFAULT_SMOKE_VARIANTS,
) -> int:
    """Create a deterministic paired synthetic dataset compatible with the config."""

    variant_records = [dict(variant) for variant in variants]
    schema = build_smoke_schema_payload(config)
    node_feature_names = tuple(schema["graph_layout"]["node_features"]["feature_datasets"].keys())
    edge_feature_names = tuple(schema["graph_layout"]["edge_features"]["feature_datasets"].keys())
    graph_feature_names = tuple(schema["graph_layout"]["graph_features"]["feature_datasets"].keys())

    mutant_path = Path(mutant_path)
    wt_path = Path(wt_path)
    mutant_path.parent.mkdir(parents=True, exist_ok=True)
    wt_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(mutant_path, "w") as mutant_handle, h5py.File(wt_path, "w") as wt_handle:
        for variant_index, variant in enumerate(variant_records):
            position = int(variant["position"])
            wt_full = str(variant["wt_full"])
            mut_full = str(variant["mut_full"])
            wt_aa = str(variant["wt_aa"])
            mut_aa = str(variant["mut_aa"])
            residue_name = wt_full[:3].upper().encode("ascii")
            mutant_key = f"residue-srv:A:{position}:{wt_full}->{mut_full}:pos_{position}_{wt_aa}_{mut_aa}"
            wt_key = f"residue-srv:A:{position}:{wt_full}->{wt_full}:PKP2_WT"
            _create_graph_group(
                mutant_handle,
                graph_key=mutant_key,
                residue_name=residue_name,
                position=position,
                variant_index=variant_index,
                is_wt=False,
                node_feature_names=node_feature_names,
                edge_feature_names=edge_feature_names,
                graph_feature_names=graph_feature_names,
            )
            if wt_key not in wt_handle:
                _create_graph_group(
                    wt_handle,
                    graph_key=wt_key,
                    residue_name=residue_name,
                    position=position,
                    variant_index=variant_index,
                    is_wt=True,
                    node_feature_names=node_feature_names,
                    edge_feature_names=edge_feature_names,
                    graph_feature_names=graph_feature_names,
                )
    return len(variant_records)


def _copy_group(source_group: h5py.Group, destination_group: h5py.Group) -> None:
    for key, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            destination_group.create_dataset(key, data=item[()])
        else:
            nested = destination_group.create_group(key)
            _copy_group(item, nested)


def _build_wt_key_from_variant(mutant_key: str) -> str:
    signature = parse_variant_signature(mutant_key)
    return (
        f"residue-srv:{signature.chain_id}:{signature.position}:"
        f"{signature.wt_aa_full}->{signature.wt_aa_full}:PKP2_WT"
    )


def _populate_wt_from_mutant(mutant_group: h5py.Group, wt_group: h5py.Group) -> None:
    _copy_group(mutant_group, wt_group)
    node_features = wt_group["node_features"]
    for feature_name, dataset in node_features.items():
        value = dataset[()]
        if feature_name.startswith("diff_") and hasattr(value, "dtype") and value.dtype.kind in {"f", "i", "u"}:
            dataset[...] = 0
        if feature_name.startswith("mask_") and hasattr(value, "dtype") and value.dtype.kind in {"f", "i", "u"}:
            dataset[...] = 1


def _build_repository_example_smoke_dataset(
    artifacts_dir: Path,
    config: Mapping[str, Any],
) -> SmokeDataArtifacts:
    paths_cfg = _require_mapping(config.get("paths"), field_name="config.paths")
    sample_root = _resolve_path(config, str(paths_cfg.get("sample_data_root", "sample_data")))
    schema_source = _resolve_path(config, str(paths_cfg.get("sample_schema", "sample_data/sample_schema.json")))
    examples = sorted((sample_root / "examples").glob("*.hdf5"))
    if not examples or not schema_source.exists():
        raise FileNotFoundError("Repository smoke examples are not available.")

    mutants_path = artifacts_dir / "mutants_smoke.hdf5"
    wt_path = artifacts_dir / "wt_companion_smoke.hdf5"
    schema_path = artifacts_dir / "sample_schema.json"
    split_path = artifacts_dir / "split_leave_position_out.json"
    shutil.copyfile(schema_source, schema_path)

    with h5py.File(mutants_path, "w") as mutant_handle, h5py.File(wt_path, "w") as wt_handle:
        for variant_index, variant in enumerate(DEFAULT_SMOKE_VARIANTS):
            source_path = examples[variant_index % len(examples)]
            with h5py.File(source_path, "r") as source_handle:
                source_key = next(iter(source_handle.keys()))
                source_group = source_handle[source_key]
                signature = parse_variant_signature(source_key)
                position = int(variant["position"])
                mutant_key = (
                    f"residue-srv:{signature.chain_id}:{position}:"
                    f"{signature.wt_aa_full}->{signature.mut_aa_full}:"
                    f"pos_{position}_{signature.wt_aa}_{signature.mut_aa}"
                )
                wt_key = _build_wt_key_from_variant(mutant_key)
                _copy_group(source_group, mutant_handle.create_group(mutant_key))
                _populate_wt_from_mutant(source_group, wt_handle.create_group(wt_key))

    temp_dir = str(artifacts_dir)
    return SmokeDataArtifacts(
        source="repository_examples",
        mutants_hdf5=str(mutants_path),
        wt_companion_hdf5=str(wt_path),
        schema_json=str(schema_path),
        split_json=str(split_path),
        pair_count=len(DEFAULT_SMOKE_VARIANTS),
        temp_dir=temp_dir,
        _temporary_directory=None,  # type: ignore[arg-type]
    )


def prepare_smoke_data(config: Mapping[str, Any]) -> SmokeDataArtifacts:
    """Create temporary smoke-test artifacts from repository examples or synthetic data."""

    temporary_directory = TemporaryDirectory(prefix="gnn_siamese_smoke_")
    artifacts_dir = Path(temporary_directory.name)
    try:
        try:
            artifacts = _build_repository_example_smoke_dataset(artifacts_dir, config)
            artifacts._temporary_directory = temporary_directory  # type: ignore[misc]
            return artifacts
        except FileNotFoundError:
            pass

        mutants_path = artifacts_dir / "mutants_smoke.hdf5"
        wt_path = artifacts_dir / "wt_companion_smoke.hdf5"
        schema_path = artifacts_dir / "sample_schema.json"
        split_path = artifacts_dir / "split_leave_position_out.json"
        write_smoke_schema_json(schema_path, config)
        pair_count = create_synthetic_mut_wt_hdf5(mutants_path, wt_path, config)
        return SmokeDataArtifacts(
            source="synthetic_temporary",
            mutants_hdf5=str(mutants_path),
            wt_companion_hdf5=str(wt_path),
            schema_json=str(schema_path),
            split_json=str(split_path),
            pair_count=pair_count,
            temp_dir=str(artifacts_dir),
            _temporary_directory=temporary_directory,
        )
    except Exception:
        temporary_directory.cleanup()
        raise
