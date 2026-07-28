"""Read-only descriptive spatial audit of audited Mut--WT residue alignments."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from gnn_siamese.data.node_pair_alignment_audit import (
    KEY_POLICY,
    audit_pair_graphs,
)
from gnn_siamese.data.pairing import pair_mutants_with_wt

SCHEMA_VERSION = "1.1.0"
COORDINATE_FIELD = "_position"
DEFAULT_RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
SUFFICIENT_ALIGNED_NODES_DESCRIPTOR = 8
SUBGRAPH_RADIUS_ANGSTROM = 20.0
EDGE_DISTANCE_CUTOFF_ANGSTROM = 5.5


def _ratio(a: int, b: int, incidents: list[dict[str, Any]], name: str) -> float | None:
    if b == 0:
        incidents.append({"code": "zero_denominator", "metric": name})
        return None
    return float(a / b)


def _key_tuple(value: Sequence[Any]) -> tuple[str, int]:
    return str(value[0]), int(value[1])


def _read_coordinates(
    graph: h5py.Group, n_nodes: int, role: str
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    incidents: list[dict[str, Any]] = []
    path = f"{graph.name}/node_features/{COORDINATE_FIELD}"
    node_group = graph.get("node_features")
    dataset = node_group.get(COORDINATE_FIELD) if isinstance(node_group, h5py.Group) else None
    metadata = {
        "path": path,
        "dtype": str(dataset.dtype) if isinstance(dataset, h5py.Dataset) else None,
        "shape": list(dataset.shape) if isinstance(dataset, h5py.Dataset) else None,
        "unit": "angstrom",
        "atomic_semantics": "unknown",
        "row_correspondence": "same node axis/order as _chain_id and res_id",
    }
    if not isinstance(dataset, h5py.Dataset):
        incidents.append({"code": "missing_coordinates", "role": role, "path": path})
        return None, np.zeros(n_nodes, dtype=bool), metadata, incidents
    values = np.asarray(dataset[()])
    if values.ndim != 2 or values.shape[1] != 3:
        incidents.append(
            {"code": "invalid_coordinate_shape", "role": role, "shape": list(values.shape)}
        )
        return None, np.zeros(n_nodes, dtype=bool), metadata, incidents
    if values.shape[0] != n_nodes:
        incidents.append(
            {
                "code": "coordinate_node_count_mismatch",
                "role": role,
                "coordinate_count": int(values.shape[0]),
                "node_count": n_nodes,
            }
        )
        return None, np.zeros(n_nodes, dtype=bool), metadata, incidents
    try:
        values = values.astype(float, copy=False)
    except (TypeError, ValueError):
        incidents.append({"code": "nonnumeric_coordinates", "role": role})
        return None, np.zeros(n_nodes, dtype=bool), metadata, incidents
    valid = np.isfinite(values).all(axis=1)
    for index in np.flatnonzero(~valid):
        incidents.append({"code": "invalid_coordinate", "role": role, "node_index": int(index)})
    return values, valid, metadata, incidents


def _radial_bin(distance: float | None, maximum: float | None) -> str | None:
    if distance is None or maximum is None or maximum <= 0:
        return None
    fraction = distance / maximum
    if fraction <= 0.25:
        return "0_25pct"
    if fraction <= 0.50:
        return "25_50pct"
    if fraction <= 0.75:
        return "50_75pct"
    return "75_100pct"


def audit_spatial_pair(
    mut_graph: h5py.Group,
    wt_graph: h5py.Group,
    *,
    variant_id: str,
    mut_graph_id: str,
    wt_graph_id: str,
    radii: Sequence[float] = DEFAULT_RADII,
    mut_hdf5_path: str | None = None,
    wt_hdf5_path: str | None = None,
) -> dict[str, Any]:
    """Audit one pair without changing prior alignment eligibility/status."""

    prior = audit_pair_graphs(
        mut_graph,
        wt_graph,
        variant_id=variant_id,
        mut_graph_id=mut_graph_id,
        wt_graph_id=wt_graph_id,
        mut_hdf5_path=mut_hdf5_path,
        wt_hdf5_path=wt_hdf5_path,
    )
    incidents: list[dict[str, Any]] = []
    mut_xyz, mut_valid, mut_meta, mut_inc = _read_coordinates(mut_graph, prior["n_mut"], "mut")
    wt_xyz, wt_valid, wt_meta, wt_inc = _read_coordinates(wt_graph, prior["n_wt"], "wt")
    incidents.extend(mut_inc)
    incidents.extend(wt_inc)
    anchor = _key_tuple(prior["mutation_anchor_key"])
    aligned = {_key_tuple(k) for k in prior["aligned_keys"]}
    mut_only = {_key_tuple(k) for k in prior["mut_only_keys"]}
    wt_only = {_key_tuple(k) for k in prior["wt_only_keys"]}
    ordered_union_keys = sorted(aligned | mut_only | wt_only)
    residue_union = [
        {
            "key": list(key),
            "exists_MUT": key in aligned or key in mut_only,
            "exists_WT": key in aligned or key in wt_only,
            "support": (
                "aligned"
                if key in aligned
                else "graph_mut_only"
                if key in mut_only
                else "graph_wt_only"
            ),
        }
        for key in ordered_union_keys
    ]
    mut_index = {
        _key_tuple(key): int(index)
        for key, index in zip(prior["aligned_keys"], prior["mut_aligned_index"])
    }
    mut_index.update(
        {_key_tuple(key): int(index) for key, index in zip(prior["mut_only_keys"], prior["mut_only_indices"])}
    )
    wt_index = {
        _key_tuple(key): int(index)
        for key, index in zip(prior["aligned_keys"], prior["wt_aligned_index"])
    }
    wt_index.update(
        {_key_tuple(key): int(index) for key, index in zip(prior["wt_only_keys"], prior["wt_only_indices"])}
    )
    mut_anchor_i = mut_index.get(anchor)
    wt_anchor_i = wt_index.get(anchor)
    mut_anchor_ok = bool(
        mut_xyz is not None and mut_anchor_i is not None and mut_valid[mut_anchor_i]
    )
    wt_anchor_ok = bool(wt_xyz is not None and wt_anchor_i is not None and wt_valid[wt_anchor_i])
    if not mut_anchor_ok:
        incidents.append({"code": "invalid_anchor_coordinate", "role": "mut"})
    if not wt_anchor_ok:
        incidents.append({"code": "invalid_anchor_coordinate", "role": "wt"})

    def distances(
        xyz: np.ndarray | None, valid: np.ndarray, anchor_i: int | None, index: Mapping[tuple[str, int], int]
    ) -> dict[tuple[str, int], float]:
        if xyz is None or anchor_i is None or not valid[anchor_i]:
            return {}
        return {
            key: float(np.linalg.norm(xyz[i] - xyz[anchor_i]))
            for key, i in index.items()
            if valid[i]
        }

    mut_dist = distances(mut_xyz, mut_valid, mut_anchor_i, mut_index)
    wt_dist = distances(wt_xyz, wt_valid, wt_anchor_i, wt_index)
    max_observed = max([*mut_dist.values(), *wt_dist.values()], default=None)
    usable_radii = [float(r) for r in radii if r <= SUBGRAPH_RADIUS_ANGSTROM]
    omitted_radii = [
        {"radius_angstrom": float(r), "reason": "documented_subgraph_radius_15A"}
        for r in radii
        if r > SUBGRAPH_RADIUS_ANGSTROM
    ]
    metrics: dict[str, Any] = {}
    masks: dict[str, Any] = {}
    for radius in usable_radii:
        label = f"{radius:g}A"
        km = {key for key, value in mut_dist.items() if value <= radius}
        kw = {key for key, value in wt_dist.items() if value <= radius}
        intersection = km & kw
        union = km | kw
        local_aligned = aligned & union
        radial_mut_only_aligned = (km - kw) & aligned
        radial_wt_only_aligned = (kw - km) & aligned
        global_mut_only_in_radius = km & mut_only
        global_wt_only_in_radius = kw & wt_only
        radial_states = [
            {
                "key": list(key),
                "inside_radius_MUT": key in km,
                "inside_radius_WT": key in kw,
                "state": (
                    "inside_both"
                    if key in km and key in kw
                    else "radial_mut_only"
                    if key in km
                    else "radial_wt_only"
                    if key in kw
                    else "outside_both"
                ),
            }
            for key in sorted(aligned)
        ]
        metrics[label] = {
            "radius_angstrom": radius,
            "n_mut_r": len(km),
            "n_wt_r": len(kw),
            "aligned_count_r": len(intersection),
            "union_count_r": len(union),
            "radial_union_count": len(union),
            "radial_aligned_count": len(intersection),
            "radial_graph_mut_only_count": len(global_mut_only_in_radius),
            "radial_graph_wt_only_count": len(global_wt_only_in_radius),
            "radial_mut_only_count": len(radial_mut_only_aligned),
            "radial_wt_only_count": len(radial_wt_only_aligned),
            "local_aligned_count": len(local_aligned),
            "coverage_union_r": _ratio(len(intersection), len(union), incidents, f"coverage_union_{label}"),
            "coverage_mut_r": _ratio(len(intersection), len(km), incidents, f"coverage_mut_{label}"),
            "coverage_wt_r": _ratio(len(intersection), len(kw), incidents, f"coverage_wt_{label}"),
            "radial_coverage_union": _ratio(
                len(intersection), len(union), incidents, f"radial_coverage_union_{label}"
            ),
        }
        masks[label] = {
            "K_MUT_keys": [list(k) for k in sorted(km)],
            "K_WT_keys": [list(k) for k in sorted(kw)],
            "K_local_union_keys": [list(k) for k in sorted(union)],
            "mut": {
                "size": len(km),
                "aligned_keys": [list(k) for k in sorted(km & aligned)],
                "graph_mut_only_keys": [list(k) for k in sorted(km & mut_only)],
                "coverage": _ratio(len(km & aligned), len(km), incidents, f"mask_mut_{label}"),
            },
            "wt": {
                "size": len(kw),
                "aligned_keys": [list(k) for k in sorted(kw & aligned)],
                "graph_wt_only_keys": [list(k) for k in sorted(kw & wt_only)],
                "coverage": _ratio(len(kw & aligned), len(kw), incidents, f"mask_wt_{label}"),
            },
            "union": {
                "size": len(union),
                "aligned_keys": [list(k) for k in sorted(local_aligned)],
                "graph_only_keys": [list(k) for k in sorted(union - aligned)],
                "coverage": _ratio(len(local_aligned), len(union), incidents, f"mask_union_{label}"),
            },
            "local_aligned": {
                "definition": "(K_MUT union K_WT) intersection K_aligned",
                "size": len(local_aligned),
                "keys": [list(k) for k in sorted(local_aligned)],
                "mut_aligned_index": [mut_index[k] for k in sorted(local_aligned)],
                "wt_aligned_index": [wt_index[k] for k in sorted(local_aligned)],
            },
            "global_exclusivity": {
                "mut_only_keys_in_radius": [list(k) for k in sorted(global_mut_only_in_radius)],
                "wt_only_keys_in_radius": [list(k) for k in sorted(global_wt_only_in_radius)],
                "mut_only_count_in_radius": len(global_mut_only_in_radius),
                "wt_only_count_in_radius": len(global_wt_only_in_radius),
            },
            "radial_exclusivity_aligned": {
                "mut_geometry_only_keys": [list(k) for k in sorted(radial_mut_only_aligned)],
                "wt_geometry_only_keys": [list(k) for k in sorted(radial_wt_only_aligned)],
                "mut_geometry_only_count": len(radial_mut_only_aligned),
                "wt_geometry_only_count": len(radial_wt_only_aligned),
            },
            "aligned_radial_states": radial_states,
            "mut_geometry_only_count": len(radial_mut_only_aligned),
            "wt_geometry_only_count": len(radial_wt_only_aligned),
            "symmetric_difference_count": len(km ^ kw),
            "aligned_radial_symmetric_difference_count": len(
                radial_mut_only_aligned | radial_wt_only_aligned
            ),
        }

    graph_exclusive_rows: list[dict[str, Any]] = []
    for role, keys, index, dist in (
        ("mut", mut_only, mut_index, mut_dist),
        ("wt", wt_only, wt_index, wt_dist),
    ):
        for key in sorted(keys):
            value = dist.get(key)
            graph_exclusive_rows.append(
                {
                    "key": list(key),
                    "original_index": index[key],
                    "branch": role,
                    "distance_to_anchor_angstrom": value,
                    "normalized_distance_to_subgraph_radius": value / SUBGRAPH_RADIUS_ANGSTROM if value is not None else None,
                    "radial_bin": _radial_bin(value, SUBGRAPH_RADIUS_ANGSTROM),
                    "coordinate_valid": value is not None,
                    "known_exclusivity_cause": "unknown_not_in_other_graph",
                }
            )
    graph_exclusive_distances = [
        row["distance_to_anchor_angstrom"]
        for row in graph_exclusive_rows
        if row["distance_to_anchor_angstrom"] is not None
    ]
    minimum = min(graph_exclusive_distances, default=None)
    median = (
        float(np.median(graph_exclusive_distances))
        if graph_exclusive_distances
        else None
    )
    peripheral_fraction = (
        sum(value > 15.0 for value in graph_exclusive_distances)
        / len(graph_exclusive_distances)
        if graph_exclusive_distances
        else None
    )
    reference_label = "8A" if "8A" in metrics else min(
        metrics, key=lambda key: abs(metrics[key]["radius_angstrom"] - 8.0), default=None
    )
    local_coverage = metrics.get(reference_label, {}).get("coverage_union_r")
    coverage_difference = (
        local_coverage - prior["coverage_union"]
        if local_coverage is not None and prior["coverage_union"] is not None
        else None
    )
    minimum_sufficient_radius = next(
        (
            value["radius_angstrom"]
            for value in metrics.values()
            if value["aligned_count_r"] >= SUFFICIENT_ALIGNED_NODES_DESCRIPTOR
        ),
        None,
    )
    size_difference = abs(prior["n_mut"] - prior["n_wt"])
    flags = {
        "low_global_high_local": bool(prior["coverage_union"] is not None and prior["coverage_union"] < 0.75 and local_coverage is not None and local_coverage >= 0.9),
        "high_global_low_local": bool(prior["coverage_union"] is not None and prior["coverage_union"] >= 0.9 and local_coverage is not None and local_coverage < 0.75),
        "graph_exclusive_near_anchor": bool(minimum is not None and minimum <= 8.0),
        "graph_exclusive_only_peripheral": bool(
            graph_exclusive_distances
            and all(value > 15.0 for value in graph_exclusive_distances)
        ),
        "coordinate_issue": bool(incidents),
        "extreme_size_difference": bool(max(prior["n_mut"], prior["n_wt"]) and size_difference / max(prior["n_mut"], prior["n_wt"]) >= 0.5),
    }
    return {
        "variant_id": variant_id,
        "mut_graph_id": mut_graph_id,
        "wt_graph_id": wt_graph_id,
        "source_hdf5": {"mut": mut_hdf5_path, "wt": wt_hdf5_path},
        "key_policy": KEY_POLICY,
        "ordered_residue_union": residue_union,
        "union_keys": [list(key) for key in ordered_union_keys],
        "ordered_residue_union_fields": ["key", "exists_MUT", "exists_WT", "support"],
        "coordinate_field": {"mut": mut_meta, "wt": wt_meta},
        "alignment_status": prior["alignment_status"],
        "prior_alignment": prior,
        "n_mut": prior["n_mut"],
        "n_wt": prior["n_wt"],
        "aligned_count": prior["aligned_count"],
        "global_aligned_count": prior["aligned_count"],
        "aligned_keys": prior["aligned_keys"],
        "mut_aligned_index": prior["mut_aligned_index"],
        "wt_aligned_index": prior["wt_aligned_index"],
        "mut_only_count": prior["mut_only_count"],
        "wt_only_count": prior["wt_only_count"],
        "global_mut_only_count": prior["mut_only_count"],
        "global_wt_only_count": prior["wt_only_count"],
        "global_union_count": (
            prior["aligned_count"]
            + prior["mut_only_count"]
            + prior["wt_only_count"]
        ),
        "graph_mut_only_keys": prior["mut_only_keys"],
        "graph_wt_only_keys": prior["wt_only_keys"],
        "graph_mut_only_indices": prior["mut_only_indices"],
        "graph_wt_only_indices": prior["wt_only_indices"],
        "coverage_union": prior["coverage_union"],
        "global_coverage_union": prior["coverage_union"],
        "radii_metrics": metrics,
        "descriptive_masks": masks,
        "omitted_radii": omitted_radii,
        "graph_exclusive_residues": graph_exclusive_rows,
        "graph_exclusive_min_distance_angstrom": minimum,
        "graph_exclusive_median_distance_angstrom": median,
        "first_radius_with_graph_exclusive_angstrom": next(
            (r for r in usable_radii if minimum is not None and minimum <= r), None
        ),
        "graph_exclusive_outer_25pct_fraction": peripheral_fraction,
        "local_reference_radius_angstrom": metrics.get(reference_label, {}).get("radius_angstrom"),
        "local_minus_global_coverage": coverage_difference,
        "minimum_radius_with_8_aligned_nodes_descriptor": minimum_sufficient_radius,
        "invalid_coordinate_count": int((~mut_valid).sum() + (~wt_valid).sum()),
        "anchor_analysis": {
            "key": list(anchor),
            "mut_index": mut_anchor_i,
            "wt_index": wt_anchor_i,
            "mut_coordinate_valid": mut_anchor_ok,
            "wt_coordinate_valid": wt_anchor_ok,
            "aligned": prior["mutation_anchor_aligned"],
        },
        "flags": flags,
        "incidents": incidents,
        "local_scale_status": "not_decided",
        "training_eligibility": "pending",
    }


def audit_spatial_hdf5_pairs(
    mut_hdf5: str | Path,
    wt_hdf5: str | Path,
    *,
    radii: Sequence[float] = DEFAULT_RADII,
) -> list[dict[str, Any]]:
    mut_path, wt_path = Path(mut_hdf5).resolve(), Path(wt_hdf5).resolve()
    with h5py.File(mut_path, "r") as mut_handle, h5py.File(wt_path, "r") as wt_handle:
        mutant_records = [
            {"variant_id": graph_id, "graph_id": graph_id, "source_path": str(mut_path)}
            for graph_id in sorted(mut_handle)
            if "PKP2_WT" not in graph_id
        ]
        wt_records = [
            {"variant_id": graph_id, "graph_id": graph_id, "source_path": str(wt_path)}
            for graph_id in sorted(wt_handle)
        ]
        pairs = pair_mutants_with_wt(mutant_records, wt_records)
        return [
            audit_spatial_pair(
                mut_handle[str(pair["graph_id"])],
                wt_handle[str(pair["wt_companion_id"])],
                variant_id=str(pair["variant_id"]),
                mut_graph_id=str(pair["graph_id"]),
                wt_graph_id=str(pair["wt_companion_id"]),
                radii=radii,
                mut_hdf5_path=str(mut_path),
                wt_hdf5_path=str(wt_path),
            )
            for pair in sorted(pairs, key=lambda item: str(item["variant_id"]))
        ]


def _percentiles(values: Iterable[float | None]) -> dict[str, float | None]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    return {
        name: (float(np.percentile(array, q)) if array.size else None)
        for name, q in (("min", 0), ("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95), ("max", 100))
    }


def build_spatial_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = sorted(
        {label for row in rows for label in row["radii_metrics"]},
        key=lambda value: float(value[:-1]),
    )
    all_graph_exclusive = [
        item["distance_to_anchor_angstrom"]
        for row in rows
        for item in row["graph_exclusive_residues"]
        if item["distance_to_anchor_angstrom"] is not None
    ]
    r101h = next(
        (
            row
            for row in rows
            if ":101:" in str(row["variant_id"])
            and "->Histidine:" in str(row["variant_id"])
        ),
        None,
    )
    positions = [int(row["prior_alignment"]["mutation_anchor_key"][1]) for row in rows]
    position_edges = np.percentile(positions, [0, 25, 50, 75, 100]) if positions else []

    def stratum(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "pair_count": len(selected),
            "global_coverage_percentiles": _percentiles(row["coverage_union"] for row in selected),
            "coverage_8A_percentiles": _percentiles(
                row["radii_metrics"].get("8A", {}).get("coverage_union_r")
                for row in selected
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "total_pairs": len(rows),
        "coordinate_field": "node_features/_position",
        "coordinate_dtype": sorted({row["coordinate_field"]["mut"]["dtype"] for row in rows}),
        "coordinate_shapes": "[N, 3]",
        "distance_unit": "angstrom",
        "coordinate_atomic_semantics": "unknown_not_stored_or_proven_by_generation_code",
        "subgraph_radius_angstrom": SUBGRAPH_RADIUS_ANGSTROM,
        "subgraph_radius_evidence": "observed anchor-to-node maximum 19.9576203491298 A across current HDF5",
        "edge_distance_cutoff_angstrom": EDGE_DISTANCE_CUTOFF_ANGSTROM,
        "edge_distance_evidence": "observed edge_features/distance maximum 5.499991454538815 A",
        "future_local_pooling_radius": "not_decided",
        "radii": {
            label: {
                "coverage_union_percentiles": _percentiles(row["radii_metrics"].get(label, {}).get("coverage_union_r") for row in rows),
                "complete_local_coverage_pairs": sum(row["radii_metrics"].get(label, {}).get("coverage_union_r") == 1.0 for row in rows),
                "pairs_with_any_graph_or_radial_asymmetry": sum(
                    row["radii_metrics"].get(label, {}).get("radial_graph_mut_only_count", 0)
                    + row["radii_metrics"].get(label, {}).get("radial_graph_wt_only_count", 0)
                    + row["radii_metrics"].get(label, {}).get("radial_mut_only_count", 0)
                    + row["radii_metrics"].get(label, {}).get("radial_wt_only_count", 0)
                    > 0
                    for row in rows
                ),
            }
            for label in labels
        },
        "graph_exclusive_distance_percentiles": _percentiles(all_graph_exclusive),
        "local_minus_global_coverage_percentiles": _percentiles(row["local_minus_global_coverage"] for row in rows),
        "flag_frequency": dict(
            Counter(name for row in rows for name, value in row["flags"].items() if value)
        ),
        "mask_comparison_8A": {
            "pairs_with_mut_wt_difference": sum(
                row["descriptive_masks"].get("8A", {}).get("symmetric_difference_count", 0) > 0
                for row in rows
            ),
            "symmetric_difference_count_percentiles": _percentiles(
                row["descriptive_masks"].get("8A", {}).get("symmetric_difference_count")
                for row in rows
            ),
            "mut_geometry_only_total": sum(
                row["descriptive_masks"].get("8A", {}).get("mut_geometry_only_count", 0)
                for row in rows
            ),
            "wt_geometry_only_total": sum(
                row["descriptive_masks"].get("8A", {}).get("wt_geometry_only_count", 0)
                for row in rows
            ),
        },
        "coordinate_issue_pairs": sum(row["flags"]["coordinate_issue"] for row in rows),
        "geometry_problem_status": "present" if any(row["flags"]["coordinate_issue"] for row in rows) else "not_observed",
        "stratified_by_graph_size": {
            "small_le_20": stratum([row for row in rows if max(row["n_mut"], row["n_wt"]) <= 20]),
            "medium_21_40": stratum([row for row in rows if 20 < max(row["n_mut"], row["n_wt"]) <= 40]),
            "large_gt_40": stratum([row for row in rows if max(row["n_mut"], row["n_wt"]) > 40]),
        },
        "position_percentiles": _percentiles(positions),
        "stratified_by_position_quartile": {
            f"q{index + 1}_{low:g}_{high:g}": stratum(
                [
                    row
                    for row in rows
                    if low
                    <= int(row["prior_alignment"]["mutation_anchor_key"][1])
                    <= high
                ]
            )
            for index, (low, high) in enumerate(zip(position_edges[:-1], position_edges[1:]))
        },
        "r101h_analysis": (
            {
                "mut_graph_id": r101h["mut_graph_id"],
                "wt_graph_id": r101h["wt_graph_id"],
                "n_mut": r101h["n_mut"],
                "n_wt": r101h["n_wt"],
                "aligned_count": r101h["aligned_count"],
                "mut_only_count": r101h["mut_only_count"],
                "wt_only_count": r101h["wt_only_count"],
                "coverage_union": r101h["coverage_union"],
                "coverage_by_radius": {
                    label: metric["coverage_union_r"]
                    for label, metric in r101h["radii_metrics"].items()
                },
                "graph_exclusive_min_distance_angstrom": r101h["graph_exclusive_min_distance_angstrom"],
                "graph_exclusive_median_distance_angstrom": r101h["graph_exclusive_median_distance_angstrom"],
                "graph_exclusive_outer_25pct_fraction": r101h["graph_exclusive_outer_25pct_fraction"],
                "anchor_analysis": r101h["anchor_analysis"],
                "flags": r101h["flags"],
                "interpretation": "graph-exclusive residues occur both near the anchor and farther out; causality is undetermined",
                "unknowns": [
                    "atomic semantics of _position",
                    "cause of residues absent from the other graph",
                    "training eligibility",
                    "final local pooling radius",
                ],
            }
            if r101h is not None
            else None
        ),
    }


def build_extreme_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    groups.append(("lowest_global_coverage", sorted(rows, key=lambda r: (r["coverage_union"] is None, r["coverage_union"] or 0))[:20]))
    groups.append(("lowest_local_8A_coverage", sorted(rows, key=lambda r: (r["radii_metrics"].get("8A", {}).get("coverage_union_r") is None, r["radii_metrics"].get("8A", {}).get("coverage_union_r") or 0))[:20]))
    groups.append(
        (
            "nearest_graph_exclusive",
            sorted(
                (
                    r
                    for r in rows
                    if r["graph_exclusive_min_distance_angstrom"] is not None
                ),
                key=lambda r: r["graph_exclusive_min_distance_angstrom"],
            )[:20],
        )
    )
    for flag, name in (
        ("low_global_high_local", "low_global_high_local"),
        ("high_global_low_local", "high_global_low_local"),
        ("graph_exclusive_only_peripheral", "graph_exclusive_only_peripheral"),
        ("coordinate_issue", "coordinate_issues"),
    ):
        groups.append((name, [r for r in rows if r["flags"][flag]]))
    groups.append(
        (
            "mask_geometry_difference",
            sorted(
                rows,
                key=lambda r: r["descriptive_masks"].get("8A", {}).get("symmetric_difference_count", 0),
                reverse=True,
            )[:20],
        )
    )
    output = []
    for category, selected in groups:
        for rank, row in enumerate(selected, 1):
            output.append(
                {
                    "category": category,
                    "rank": rank,
                    "variant_id": row["variant_id"],
                    "mut_graph_id": row["mut_graph_id"],
                    "wt_graph_id": row["wt_graph_id"],
                    "coverage_union": row["coverage_union"],
                    "coverage_union_8A": row["radii_metrics"].get("8A", {}).get("coverage_union_r"),
                    "graph_exclusive_min_distance_angstrom": row["graph_exclusive_min_distance_angstrom"],
                    "mask_symmetric_difference_8A": row["descriptive_masks"].get("8A", {}).get("symmetric_difference_count"),
                    "flags": ";".join(name for name, value in row["flags"].items() if value),
                }
            )
    return output


def build_spatial_schema() -> dict[str, Any]:
    return {
        "contract_version": SCHEMA_VERSION,
        "key_policy": KEY_POLICY,
        "coordinate_field": "node_features/_position",
        "distance_unit": "angstrom",
        "atomic_semantics": "unknown",
        "radii_angstrom": list(DEFAULT_RADII),
        "subgraph_radius_angstrom": SUBGRAPH_RADIUS_ANGSTROM,
        "edge_distance_cutoff_angstrom": EDGE_DISTANCE_CUTOFF_ANGSTROM,
        "definitions": {
            "K_MUT(r)": "audited keys with finite Mut distance to Mut anchor <= r",
            "K_WT(r)": "audited keys with finite WT distance to WT anchor <= r",
            "coverage_union_r": "|K_MUT intersection K_WT| / |K_MUT union K_WT|",
            "coverage_mut_r": "|intersection| / |K_MUT|",
            "coverage_wt_r": "|intersection| / |K_WT|",
            "global_exclusivity": "key absent from the other graph",
            "radial_exclusivity": "aligned key present in both graphs but inside radius in only one branch",
            "support_states": {
                "aligned": "exists_MUT=1 and exists_WT=1",
                "graph_mut_only": "exists_MUT=1 and exists_WT=0",
                "graph_wt_only": "exists_MUT=0 and exists_WT=1",
                "forbidden": "exists_MUT=0 and exists_WT=0",
            },
            "K_local_union(r)": "K_MUT(r) union K_WT(r)",
            "K_local_aligned(r)": "(K_MUT(r) union K_WT(r)) intersection K_aligned",
        },
        "null_policy": "zero denominators and unavailable distances are null and generate incidents",
        "flags": {
            "low_global_high_local": "global < 0.75 and 8 A local >= 0.90",
            "high_global_low_local": "global >= 0.90 and 8 A local < 0.75",
            "graph_exclusive_near_anchor": "minimum graph-exclusive distance <= 8 A",
            "graph_exclusive_only_peripheral": "all finite graph-exclusive distances > 15 A (outer 25% of 20 A)",
            "coordinate_issue": "one or more spatial incidents",
            "extreme_size_difference": "|Nmut-Nwt|/max(Nmut,Nwt) >= 0.50",
        },
        "eligibility_effect": "none_descriptive_only",
        "availability_policy_mvp": "metadata_quality_audit_only; not an encoder, DeltaBlock or z_delta_local input",
        "missing_embedding_policy": "no zero-filled embeddings; only real aligned pairs may enter a future DeltaBlock",
        "minimum_coverage": None,
        "local_pooling_radius": "not_decided",
    }


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "variant_id": row["variant_id"],
        "mut_graph_id": row["mut_graph_id"],
        "wt_graph_id": row["wt_graph_id"],
        "n_mut": row["n_mut"],
        "n_wt": row["n_wt"],
        "aligned_count": row["aligned_count"],
        "global_aligned_count": row["global_aligned_count"],
        "mut_only_count": row["mut_only_count"],
        "wt_only_count": row["wt_only_count"],
        "global_mut_only_count": row["global_mut_only_count"],
        "global_wt_only_count": row["global_wt_only_count"],
        "global_union_count": row["global_union_count"],
        "coverage_union": row["coverage_union"],
        "global_coverage_union": row["global_coverage_union"],
        "graph_exclusive_min_distance_angstrom": row["graph_exclusive_min_distance_angstrom"],
        "graph_exclusive_median_distance_angstrom": row["graph_exclusive_median_distance_angstrom"],
        "graph_exclusive_outer_25pct_fraction": row["graph_exclusive_outer_25pct_fraction"],
        "local_minus_global_coverage": row["local_minus_global_coverage"],
        "invalid_coordinate_count": row["invalid_coordinate_count"],
        "alignment_status": row["alignment_status"],
        "incident_count": len(row["incidents"]),
        "flags": ";".join(name for name, value in row["flags"].items() if value),
        "union_keys": json.dumps(row["union_keys"], separators=(",", ":")),
        "exists_MUT": json.dumps(
            [int(item["exists_MUT"]) for item in row["ordered_residue_union"]],
            separators=(",", ":"),
        ),
        "exists_WT": json.dumps(
            [int(item["exists_WT"]) for item in row["ordered_residue_union"]],
            separators=(",", ":"),
        ),
        "global_support": json.dumps(
            [item["support"] for item in row["ordered_residue_union"]],
            separators=(",", ":"),
        ),
        "incidents": json.dumps(row["incidents"], separators=(",", ":")),
        "training_eligibility": row["training_eligibility"],
        "local_scale_status": row["local_scale_status"],
    }
    for label, metric in row["radii_metrics"].items():
        for name in (
            "coverage_union_r",
            "coverage_mut_r",
            "coverage_wt_r",
            "radial_coverage_union",
            "n_mut_r",
            "n_wt_r",
            "aligned_count_r",
            "radial_union_count",
            "radial_aligned_count",
            "radial_graph_mut_only_count",
            "radial_graph_wt_only_count",
            "radial_mut_only_count",
            "radial_wt_only_count",
            "local_aligned_count",
        ):
            result[f"{name.removesuffix('_r')}_{label}"] = metric[name]
        result[f"mask_symmetric_difference_{label}"] = row["descriptive_masks"][label]["symmetric_difference_count"]
        local = row["descriptive_masks"][label]["local_aligned"]
        radial = row["descriptive_masks"][label]["radial_exclusivity_aligned"]
        result[f"local_aligned_keys_{label}"] = json.dumps(
            local["keys"], separators=(",", ":")
        )
        result[f"local_mut_aligned_index_{label}"] = json.dumps(
            local["mut_aligned_index"], separators=(",", ":")
        )
        result[f"local_wt_aligned_index_{label}"] = json.dumps(
            local["wt_aligned_index"], separators=(",", ":")
        )
        result[f"radial_mut_only_keys_{label}"] = json.dumps(
            radial["mut_geometry_only_keys"], separators=(",", ":")
        )
        result[f"radial_wt_only_keys_{label}"] = json.dumps(
            radial["wt_geometry_only_keys"], separators=(",", ":")
        )
        result[f"K_MUT_keys_{label}"] = json.dumps(
            row["descriptive_masks"][label]["K_MUT_keys"], separators=(",", ":")
        )
        result[f"K_WT_keys_{label}"] = json.dumps(
            row["descriptive_masks"][label]["K_WT_keys"], separators=(",", ":")
        )
        result[f"K_local_union_keys_{label}"] = json.dumps(
            row["descriptive_masks"][label]["K_local_union_keys"],
            separators=(",", ":"),
        )
        result[f"aligned_radial_states_{label}"] = json.dumps(
            row["descriptive_masks"][label]["aligned_radial_states"],
            separators=(",", ":"),
        )
    return result


def write_spatial_artifacts(
    output_dir: str | Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "node_pair_alignment_spatial_audit.json",
        "csv": output / "node_pair_alignment_spatial_audit.csv",
        "summary": output / "node_pair_alignment_spatial_summary.json",
        "extremes": output / "node_pair_alignment_extreme_cases.csv",
        "schema": output / "node_pair_alignment_spatial_schema.json",
    }
    paths["json"].write_text(json.dumps({"schema_version": SCHEMA_VERSION, "pairs": list(rows)}, indent=2), encoding="utf-8")
    flat = [_csv_row(row) for row in rows]
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else ["variant_id"])
        writer.writeheader()
        writer.writerows(flat)
    paths["summary"].write_text(json.dumps(build_spatial_summary(rows), indent=2), encoding="utf-8")
    extremes = build_extreme_cases(rows)
    with paths["extremes"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(extremes[0]) if extremes else ["category"])
        writer.writeheader()
        writer.writerows(extremes)
    paths["schema"].write_text(json.dumps(build_spatial_schema(), indent=2), encoding="utf-8")
    return paths
