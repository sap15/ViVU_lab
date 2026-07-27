"""Descriptive residue-level alignment audit for paired Mut--WT HDF5 graphs.

This module is intentionally independent from the production loader, dataset,
collate path and model.  It reads residue identity metadata, constructs an
audited ``(chain_id, residue_number)`` fallback alignment, and reports evidence
without imposing a minimum-coverage threshold.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from gnn_siamese.data.pairing import pair_mutants_with_wt, parse_variant_signature


KEY_POLICY = "audited_chain_residue_fallback"
IDENTITY_FIELD_NAMES = ("_chain_id", "res_id", "_name")
NAME_TAIL_RE = re.compile(
    r"(?:^|\s)(?P<chain>\S+)\s+(?P<number>[+-]?\d+)(?P<suffix>[A-Za-z]+)?\s*$"
)
ALIGNMENT_STATUSES = ("valid", "partial", "rejected")
COVERAGE_NAMES = (
    "coverage_union",
    "coverage_mut",
    "coverage_wt",
    "coverage_min",
    "coverage_max",
)


def _incident(
    code: str,
    *,
    role: str,
    node_indices: Iterable[int] = (),
    key: tuple[str, int] | None = None,
    detail: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "role": role,
        "node_indices": list(node_indices),
        "key": list(key) if key is not None else None,
        "detail": detail,
    }


def _strict_utf8(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8"), None
        except UnicodeDecodeError:
            return None, "undecodable_utf8"
    if isinstance(value, (str, np.str_)):
        return str(value), None
    return None, "not_a_string"


def _normalise_chain(value: Any) -> tuple[str | None, str | None]:
    decoded, error = _strict_utf8(value)
    if error is not None:
        return None, error
    normalised = decoded.strip().upper()
    if not normalised:
        return None, "empty_chain"
    return normalised, None


def _normalise_residue_number(value: Any) -> tuple[int | None, str | None]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_res_id"
    if not math.isfinite(number):
        return None, "nonfinite_res_id"
    if not number.is_integer():
        return None, "nonintegral_res_id"
    return int(number), None


def _parse_redundant_name(value: Any) -> tuple[tuple[str, int] | None, str | None]:
    decoded, error = _strict_utf8(value)
    if error is not None:
        return None, "undecodable_name"
    match = NAME_TAIL_RE.search(decoded.strip())
    if match is None:
        return None, "unparseable_name"
    return (match.group("chain").strip().upper(), int(match.group("number"))), None


def _field_description(dataset: h5py.Dataset | None, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "present": dataset is not None,
        "dtype": str(dataset.dtype) if dataset is not None else None,
        "shape": list(dataset.shape) if dataset is not None else None,
    }


def extract_graph_identity(
    graph: h5py.Group,
    *,
    role: str,
) -> dict[str, Any]:
    """Extract valid fallback keys and identity incidents from one graph."""

    node_group = graph.get("node_features")
    graph_path = graph.name
    if not isinstance(node_group, h5py.Group):
        return {
            "n_nodes": 0,
            "valid_key_to_indices": {},
            "valid_keys_by_index": [],
            "invalid_indices": [],
            "duplicate_keys": {},
            "incidents": [
                _incident(
                    "missing_node_features",
                    role=role,
                    detail=f"{graph_path}/node_features is absent or is not a group.",
                )
            ],
            "field_metadata": {
                name: _field_description(None, f"{graph_path}/node_features/{name}")
                for name in IDENTITY_FIELD_NAMES
            },
            "chain_frequency": {},
        }

    datasets = {
        name: node_group.get(name)
        if isinstance(node_group.get(name), h5py.Dataset)
        else None
        for name in IDENTITY_FIELD_NAMES
    }
    field_metadata = {
        name: _field_description(dataset, f"{graph_path}/node_features/{name}")
        for name, dataset in datasets.items()
    }
    incidents: list[dict[str, Any]] = []
    missing = [name for name, dataset in datasets.items() if dataset is None]
    for name in missing:
        incidents.append(
            _incident(
                f"missing_{name.lstrip('_')}",
                role=role,
                detail=f"Required audit field {name!r} is absent.",
            )
        )

    lengths = {
        name: int(dataset.shape[0])
        for name, dataset in datasets.items()
        if dataset is not None and dataset.ndim >= 1
    }
    if any(dataset is not None and dataset.ndim == 0 for dataset in datasets.values()):
        incidents.append(
            _incident(
                "scalar_identity_field",
                role=role,
                detail="Every identity field must have a node axis.",
            )
        )
    if len(set(lengths.values())) > 1:
        incidents.append(
            _incident(
                "identity_field_length_mismatch",
                role=role,
                detail=f"Identity field lengths differ: {lengths!r}.",
            )
        )
    n_nodes = max(lengths.values(), default=0)
    if missing or len(set(lengths.values())) > 1:
        return {
            "n_nodes": n_nodes,
            "valid_key_to_indices": {},
            "valid_keys_by_index": [None] * n_nodes,
            "invalid_indices": list(range(n_nodes)),
            "duplicate_keys": {},
            "incidents": incidents,
            "field_metadata": field_metadata,
            "chain_frequency": {},
        }

    chain_values = np.asarray(datasets["_chain_id"][()]).reshape(-1)
    residue_values = np.asarray(datasets["res_id"][()]).reshape(-1)
    name_values = np.asarray(datasets["_name"][()]).reshape(-1)
    valid_keys_by_index: list[tuple[str, int] | None] = []
    key_to_indices: dict[tuple[str, int], list[int]] = {}
    invalid_indices: list[int] = []
    chain_frequency: Counter[str] = Counter()

    for index, (chain_raw, residue_raw, name_raw) in enumerate(
        zip(chain_values, residue_values, name_values)
    ):
        chain, chain_error = _normalise_chain(chain_raw)
        residue, residue_error = _normalise_residue_number(residue_raw)
        if chain_error is not None:
            incidents.append(
                _incident(
                    chain_error,
                    role=role,
                    node_indices=[index],
                    detail="Node chain cannot be used in the fallback key.",
                )
            )
        if residue_error is not None:
            incidents.append(
                _incident(
                    residue_error,
                    role=role,
                    node_indices=[index],
                    detail="Node res_id cannot be used in the fallback key.",
                )
            )
        key = (chain, residue) if chain is not None and residue is not None else None
        valid_keys_by_index.append(key)
        if key is None:
            invalid_indices.append(index)
        else:
            key_to_indices.setdefault(key, []).append(index)
            chain_frequency[key[0]] += 1

        parsed_name, name_error = _parse_redundant_name(name_raw)
        if name_error is not None:
            incidents.append(
                _incident(
                    name_error,
                    role=role,
                    node_indices=[index],
                    key=key,
                    detail="_name could not validate the fallback identity.",
                )
            )
        elif key is not None and parsed_name != key:
            incidents.append(
                _incident(
                    "name_identity_mismatch",
                    role=role,
                    node_indices=[index],
                    key=key,
                    detail=f"_name identity {parsed_name!r} disagrees with fallback key {key!r}.",
                )
            )

    duplicate_keys = {
        key: indices for key, indices in key_to_indices.items() if len(indices) > 1
    }
    for key, indices in sorted(duplicate_keys.items()):
        incidents.append(
            _incident(
                "duplicate_key",
                role=role,
                node_indices=indices,
                key=key,
                detail="Fallback key occurs more than once and is not deterministically resolvable.",
            )
        )

    return {
        "n_nodes": n_nodes,
        "valid_key_to_indices": key_to_indices,
        "valid_keys_by_index": valid_keys_by_index,
        "invalid_indices": invalid_indices,
        "duplicate_keys": duplicate_keys,
        "incidents": incidents,
        "field_metadata": field_metadata,
        "chain_frequency": dict(sorted(chain_frequency.items())),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _coverage_metrics(aligned: int, n_mut: int, n_wt: int) -> dict[str, float | None]:
    union = n_mut + n_wt - aligned
    return {
        "coverage_union": _safe_ratio(aligned, union),
        "coverage_mut": _safe_ratio(aligned, n_mut),
        "coverage_wt": _safe_ratio(aligned, n_wt),
        "coverage_min": _safe_ratio(aligned, min(n_mut, n_wt)),
        "coverage_max": _safe_ratio(aligned, max(n_mut, n_wt)),
    }


def audit_pair_graphs(
    mut_graph: h5py.Group,
    wt_graph: h5py.Group,
    *,
    variant_id: str,
    mut_graph_id: str,
    wt_graph_id: str,
    mut_hdf5_path: str | None = None,
    wt_hdf5_path: str | None = None,
) -> dict[str, Any]:
    """Audit one already-resolved missense/WT graph pair."""

    signature = parse_variant_signature({"variant_id": variant_id})
    anchor_key = (signature.chain_id.strip().upper(), int(signature.position))
    mut = extract_graph_identity(mut_graph, role="mut")
    wt = extract_graph_identity(wt_graph, role="wt")
    incidents = [*mut["incidents"], *wt["incidents"]]

    mut_unique = {
        key: indices[0]
        for key, indices in mut["valid_key_to_indices"].items()
        if len(indices) == 1
    }
    wt_unique = {
        key: indices[0]
        for key, indices in wt["valid_key_to_indices"].items()
        if len(indices) == 1
    }
    aligned_keys = sorted(set(mut_unique).intersection(wt_unique))
    mut_only_keys = sorted(set(mut_unique).difference(wt_unique))
    wt_only_keys = sorted(set(wt_unique).difference(mut_unique))
    mut_aligned_index = [mut_unique[key] for key in aligned_keys]
    wt_aligned_index = [wt_unique[key] for key in aligned_keys]
    mut_only_indices = [mut_unique[key] for key in mut_only_keys]
    wt_only_indices = [wt_unique[key] for key in wt_only_keys]
    mut_alignment_mask = [False] * mut["n_nodes"]
    wt_alignment_mask = [False] * wt["n_nodes"]
    for index in mut_aligned_index:
        mut_alignment_mask[index] = True
    for index in wt_aligned_index:
        wt_alignment_mask[index] = True

    shared_nonunique = sorted(
        set(mut["valid_key_to_indices"]).intersection(wt["valid_key_to_indices"])
        - set(aligned_keys)
    )
    for key in shared_nonunique:
        incidents.append(
            _incident(
                "non_unique_correspondence",
                role="pair",
                key=key,
                detail="A shared key is duplicated in at least one branch.",
            )
        )

    anchor_mut_count = len(mut["valid_key_to_indices"].get(anchor_key, []))
    anchor_wt_count = len(wt["valid_key_to_indices"].get(anchor_key, []))
    anchor_aligned = anchor_key in aligned_keys
    if anchor_mut_count == 0:
        incidents.append(
            _incident(
                "mutation_anchor_missing_mut",
                role="mut",
                key=anchor_key,
                detail="Mutation anchor is absent from the mutant graph.",
            )
        )
    elif anchor_mut_count > 1:
        incidents.append(
            _incident(
                "mutation_anchor_ambiguous_mut",
                role="mut",
                key=anchor_key,
                node_indices=mut["valid_key_to_indices"][anchor_key],
                detail="Mutation anchor occurs more than once in the mutant graph.",
            )
        )
    if anchor_wt_count == 0:
        incidents.append(
            _incident(
                "mutation_anchor_missing_wt",
                role="wt",
                key=anchor_key,
                detail="Mutation anchor is absent from the WT graph.",
            )
        )
    elif anchor_wt_count > 1:
        incidents.append(
            _incident(
                "mutation_anchor_ambiguous_wt",
                role="wt",
                key=anchor_key,
                node_indices=wt["valid_key_to_indices"][anchor_key],
                detail="Mutation anchor occurs more than once in the WT graph.",
            )
        )
    if anchor_mut_count == 1 and anchor_wt_count == 1 and not anchor_aligned:
        incidents.append(
            _incident(
                "mutation_anchor_not_aligned",
                role="pair",
                key=anchor_key,
                detail="Unique mutation anchors are not part of the aligned intersection.",
            )
        )

    n_mut_unique = len(mut_unique)
    n_wt_unique = len(wt_unique)
    coverages = _coverage_metrics(len(aligned_keys), n_mut_unique, n_wt_unique)
    for metric, value in coverages.items():
        if value is None:
            incidents.append(
                _incident(
                    "zero_coverage_denominator",
                    role="pair",
                    detail=f"{metric} has a zero denominator and is reported as null.",
                )
            )

    rejection_codes = {
        "missing_node_features",
        "missing_chain_id",
        "missing_res_id",
        "missing_name",
        "scalar_identity_field",
        "identity_field_length_mismatch",
        "empty_chain",
        "undecodable_utf8",
        "not_a_string",
        "invalid_res_id",
        "nonfinite_res_id",
        "nonintegral_res_id",
        "undecodable_name",
        "unparseable_name",
        "name_identity_mismatch",
        "duplicate_key",
        "non_unique_correspondence",
        "mutation_anchor_missing_mut",
        "mutation_anchor_missing_wt",
        "mutation_anchor_ambiguous_mut",
        "mutation_anchor_ambiguous_wt",
        "mutation_anchor_not_aligned",
        "zero_coverage_denominator",
    }
    incident_codes = {incident["code"] for incident in incidents}
    if not aligned_keys:
        incidents.append(
            _incident(
                "empty_alignment",
                role="pair",
                detail="The pair has no one-to-one aligned fallback keys.",
            )
        )
        incident_codes.add("empty_alignment")
    if incident_codes.intersection(rejection_codes) or "empty_alignment" in incident_codes:
        alignment_status = "rejected"
    elif mut_only_keys or wt_only_keys:
        alignment_status = "partial"
    else:
        alignment_status = "valid"

    return {
        "variant_id": variant_id,
        "mut_graph_id": mut_graph_id,
        "wt_graph_id": wt_graph_id,
        "source_hdf5": {
            "mut": mut_hdf5_path,
            "wt": wt_hdf5_path,
        },
        "key_policy": KEY_POLICY,
        "conceptual_key_fields": ["chain_id", "residue_number", "insertion_code"],
        "fallback_key_fields": ["_chain_id", "res_id"],
        "insertion_code_status": "not_stored_not_assumed_empty",
        "field_metadata": {
            "mut": mut["field_metadata"],
            "wt": wt["field_metadata"],
        },
        "n_mut": mut["n_nodes"],
        "n_wt": wt["n_nodes"],
        "n_mut_valid_unique": n_mut_unique,
        "n_wt_valid_unique": n_wt_unique,
        "aligned_count": len(aligned_keys),
        "mut_aligned_index": mut_aligned_index,
        "wt_aligned_index": wt_aligned_index,
        "aligned_keys": [list(key) for key in aligned_keys],
        "mut_only_indices": mut_only_indices,
        "wt_only_indices": wt_only_indices,
        "mut_only_keys": [list(key) for key in mut_only_keys],
        "wt_only_keys": [list(key) for key in wt_only_keys],
        "mut_alignment_mask": mut_alignment_mask,
        "wt_alignment_mask": wt_alignment_mask,
        "mut_only_count": len(mut_only_keys),
        "wt_only_count": len(wt_only_keys),
        "invalid_key_count_mut": len(mut["invalid_indices"]),
        "invalid_key_count_wt": len(wt["invalid_indices"]),
        "duplicate_key_count_mut": sum(
            len(indices) - 1 for indices in mut["duplicate_keys"].values()
        ),
        "duplicate_key_count_wt": sum(
            len(indices) - 1 for indices in wt["duplicate_keys"].values()
        ),
        "chain_frequency_mut": mut["chain_frequency"],
        "chain_frequency_wt": wt["chain_frequency"],
        **coverages,
        "mutation_anchor_key": list(anchor_key),
        "mutation_anchor_in_mut": anchor_mut_count == 1,
        "mutation_anchor_in_wt": anchor_wt_count == 1,
        "mutation_anchor_count_mut": anchor_mut_count,
        "mutation_anchor_count_wt": anchor_wt_count,
        "mutation_anchor_aligned": anchor_aligned,
        "local_scale_status": "not_assessed",
        "domain_scale_status": "not_assessed",
        "incidents": incidents,
        "incident_codes": sorted({incident["code"] for incident in incidents}),
        "alignment_status": alignment_status,
    }


def audit_hdf5_pairs(
    mut_hdf5: str | Path,
    wt_hdf5: str | Path,
) -> list[dict[str, Any]]:
    """Resolve and audit every missense pair, opening both HDF5 files read-only."""

    mut_path = Path(mut_hdf5).resolve()
    wt_path = Path(wt_hdf5).resolve()
    with h5py.File(mut_path, "r") as mut_handle, h5py.File(wt_path, "r") as wt_handle:
        mutant_records = []
        for graph_id in sorted(mut_handle.keys()):
            signature = parse_variant_signature({"variant_id": graph_id})
            if signature.mut_aa != signature.wt_aa:
                mutant_records.append(
                    {"variant_id": graph_id, "graph_id": graph_id, "source_path": str(mut_path)}
                )
        wt_records = [
            {"variant_id": graph_id, "graph_id": graph_id, "source_path": str(wt_path)}
            for graph_id in sorted(wt_handle.keys())
        ]
        pairs = pair_mutants_with_wt(mutant_records, wt_records)
        records = []
        for pair in sorted(pairs, key=lambda value: str(value["variant_id"])):
            mut_graph_id = str(pair["graph_id"])
            wt_graph_id = str(pair["wt_companion_id"])
            records.append(
                audit_pair_graphs(
                    mut_handle[mut_graph_id],
                    wt_handle[wt_graph_id],
                    variant_id=str(pair["variant_id"]),
                    mut_graph_id=mut_graph_id,
                    wt_graph_id=wt_graph_id,
                    mut_hdf5_path=str(mut_path),
                    wt_hdf5_path=str(wt_path),
                )
            )
    return records


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    levels = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    calculated = np.percentile(np.asarray(values, dtype=float), levels)
    return {f"p{level}": float(value) for level, value in zip(levels, calculated)}


def build_alignment_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build deterministic aggregate statistics without acceptance thresholds."""

    rows = list(records)
    status_counts = Counter(str(row["alignment_status"]) for row in rows)
    chain_frequency = Counter()
    incident_frequency = Counter()
    for row in rows:
        chain_frequency.update(row["chain_frequency_mut"])
        chain_frequency.update(row["chain_frequency_wt"])
        incident_frequency.update(row["incident_codes"])

    coverage_summary: dict[str, Any] = {}
    for metric in COVERAGE_NAMES:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        counts, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, 21))
        coverage_summary[metric] = {
            "null_count": sum(row[metric] is None for row in rows),
            "percentiles": _percentiles(values),
            "histogram": {
                "bin_edges": [float(value) for value in edges],
                "counts": [int(value) for value in counts],
            },
        }

    size_differences = [int(row["n_mut"]) - int(row["n_wt"]) for row in rows]
    extreme_order = sorted(
        rows,
        key=lambda row: (
            float("inf") if row["coverage_union"] is None else float(row["coverage_union"]),
            -abs(int(row["n_mut"]) - int(row["n_wt"])),
            str(row["variant_id"]),
        ),
    )
    return {
        "schema_version": 1,
        "key_policy": KEY_POLICY,
        "primary_coverage_metric": "coverage_union",
        "minimum_coverage": None,
        "total_pairs": len(rows),
        "status_counts": {
            status: int(status_counts.get(status, 0)) for status in ALIGNMENT_STATUSES
        },
        "coverage_metrics": coverage_summary,
        "size_difference": {
            "values": size_differences,
            "min": min(size_differences, default=None),
            "max": max(size_differences, default=None),
            "percentiles": _percentiles([float(value) for value in size_differences]),
        },
        "exclusive_residues": {
            "mut_total": sum(int(row["mut_only_count"]) for row in rows),
            "wt_total": sum(int(row["wt_only_count"]) for row in rows),
            "pairs_with_mut_only": sum(int(row["mut_only_count"]) > 0 for row in rows),
            "pairs_with_wt_only": sum(int(row["wt_only_count"]) > 0 for row in rows),
        },
        "chain_frequency": dict(sorted(chain_frequency.items())),
        "duplicates": {
            "mut_total": sum(int(row["duplicate_key_count_mut"]) for row in rows),
            "wt_total": sum(int(row["duplicate_key_count_wt"]) for row in rows),
            "pairs_affected": sum(
                int(row["duplicate_key_count_mut"]) > 0
                or int(row["duplicate_key_count_wt"]) > 0
                for row in rows
            ),
        },
        "mutation_anchors": {
            "missing_or_ambiguous_mut": sum(
                int(row["mutation_anchor_count_mut"]) != 1 for row in rows
            ),
            "missing_or_ambiguous_wt": sum(
                int(row["mutation_anchor_count_wt"]) != 1 for row in rows
            ),
            "not_aligned": sum(not bool(row["mutation_anchor_aligned"]) for row in rows),
        },
        "incident_frequency": dict(sorted(incident_frequency.items())),
        "extreme_cases": [
            {
                "variant_id": row["variant_id"],
                "alignment_status": row["alignment_status"],
                "coverage_union": row["coverage_union"],
                "n_mut": row["n_mut"],
                "n_wt": row["n_wt"],
                "mut_only_count": row["mut_only_count"],
                "wt_only_count": row["wt_only_count"],
                "incident_codes": row["incident_codes"],
            }
            for row in extreme_order[:20]
        ],
    }


CSV_FIELDS = (
    "variant_id",
    "mut_graph_id",
    "wt_graph_id",
    "n_mut",
    "n_wt",
    "aligned_count",
    "mut_only_count",
    "wt_only_count",
    *COVERAGE_NAMES,
    "mutation_anchor_in_mut",
    "mutation_anchor_in_wt",
    "mutation_anchor_aligned",
    "duplicate_key_count_mut",
    "duplicate_key_count_wt",
    "invalid_key_count_mut",
    "invalid_key_count_wt",
    "key_policy",
    "alignment_status",
    "incident_codes",
)


def write_alignment_artifacts(
    output_dir: str | Path,
    records: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write the three requested deterministic audit artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "node_pair_alignment_audit.json",
        "csv": output / "node_pair_alignment_audit.csv",
        "summary": output / "node_pair_alignment_summary.json",
    }
    ordered = sorted(records, key=lambda row: str(row["variant_id"]))
    detail = {
        "schema_version": 1,
        "key_policy": KEY_POLICY,
        "primary_coverage_metric": "coverage_union",
        "minimum_coverage": None,
        "pairs": ordered,
    }
    paths["json"].write_text(
        json.dumps(detail, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["summary"].write_text(
        json.dumps(
            build_alignment_summary(ordered),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in ordered:
            csv_row = {field: row[field] for field in CSV_FIELDS}
            csv_row["incident_codes"] = ";".join(row["incident_codes"])
            writer.writerow(csv_row)
    return paths
