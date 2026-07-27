"""Descriptive audit of residue identity fields in DeepRank2 HDF5 graphs.

This module deliberately does not construct a Mut--WT alignment.  It only
reports whether the stored fields could support a future structural key.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


IDENTITY_FIELDS = ("_chain_id", "res_id", "_name", "is_mutation")
INSERTION_NORMALIZED_NAMES = {
    "insertioncode",
    "insertion",
    "icode",
    "inscode",
}
CASE_KEY_RE = re.compile(r"^residue-srv:(?P<chain>[^:]*):(?P<position>[+-]?\d+)(?::|$)")
NAME_TAIL_RE = re.compile(
    r"(?:^|\s)(?P<chain>\S+)\s+(?P<number>[+-]?\d+)(?P<suffix>[A-Za-z]+)?\s*$"
)
MAX_ANOMALY_EXAMPLES = 10


def _normalise_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def is_insertion_code_field(name: str) -> bool:
    """Return whether a dataset name reasonably denotes an insertion code."""

    normalised = _normalise_field_name(name)
    return normalised in INSERTION_NORMALIZED_NAMES or (
        "insertion" in normalised and "code" in normalised
    )


def _dataset_description(dataset: h5py.Dataset | None) -> dict[str, Any] | None:
    if dataset is None:
        return None
    return {"dtype": str(dataset.dtype), "shape": list(dataset.shape)}


def _decode_strings(values: np.ndarray) -> tuple[list[str | None], int]:
    decoded: list[str | None] = []
    failures = 0
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            try:
                decoded.append(bytes(value).decode("utf-8"))
            except UnicodeDecodeError:
                decoded.append(None)
                failures += 1
        elif isinstance(value, str):
            decoded.append(value)
        else:
            try:
                decoded.append(str(value))
            except Exception:  # pragma: no cover - defensive for exotic HDF5 types
                decoded.append(None)
                failures += 1
    return decoded, failures


def parse_case_key(case_key: str) -> tuple[str | None, int | None]:
    match = CASE_KEY_RE.match(case_key)
    if not match:
        return None, None
    return match.group("chain"), int(match.group("position"))


def _scalar_number(dataset: h5py.Dataset | None) -> float | None:
    if dataset is None:
        return None
    values = np.asarray(dataset[()]).reshape(-1)
    if values.size != 1:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return None


def _safe_int(value: float) -> int | None:
    return int(value) if np.isfinite(value) and float(value).is_integer() else None


def audit_graph(
    graph: h5py.Group,
    *,
    hdf5_path: str | Path,
    role: str,
    case_key: str,
) -> dict[str, Any]:
    """Audit one graph without making alignment or acceptance decisions."""

    node_group = graph.get("node_features")
    graph_features = graph.get("graph_features")
    node_fields = sorted(node_group.keys()) if isinstance(node_group, h5py.Group) else []
    insertion_fields = sorted(name for name in node_fields if is_insertion_code_field(name))

    datasets = {
        name: node_group.get(name) if isinstance(node_group, h5py.Group) else None
        for name in IDENTITY_FIELDS
    }
    field_metadata = {
        name: _dataset_description(dataset) for name, dataset in datasets.items()
    }
    field_metadata.update(
        {
            name: _dataset_description(node_group[name])
            for name in insertion_fields
            if isinstance(node_group, h5py.Group)
        }
    )

    lengths = [
        int(dataset.shape[0])
        for dataset in datasets.values()
        if dataset is not None and dataset.ndim > 0
    ]
    n_nodes = lengths[0] if lengths else 0

    chain_values: list[str | None] = []
    chain_decode_failures = 0
    if datasets["_chain_id"] is not None:
        chain_values, chain_decode_failures = _decode_strings(datasets["_chain_id"][()])
        n_nodes = len(chain_values)

    res_values = np.asarray([], dtype=float)
    if datasets["res_id"] is not None:
        raw_res = np.asarray(datasets["res_id"][()]).reshape(-1)
        n_nodes = len(raw_res) if not lengths else n_nodes
        try:
            res_values = raw_res.astype(float)
        except (TypeError, ValueError):
            res_values = np.full(raw_res.shape, np.nan, dtype=float)

    names: list[str | None] = []
    name_decode_failures = 0
    if datasets["_name"] is not None:
        names, name_decode_failures = _decode_strings(datasets["_name"][()])
        n_nodes = len(names) if not lengths else n_nodes

    finite = np.isfinite(res_values)
    integral = finite & (res_values == np.floor(res_values))
    candidate_keys: list[tuple[str, int]] = []
    if chain_values and res_values.size:
        for chain, residue, valid in zip(chain_values, res_values, integral):
            if chain is not None and valid:
                candidate_keys.append((chain, int(residue)))
    key_counts = Counter(candidate_keys)
    duplicate_key_count = sum(count - 1 for count in key_counts.values() if count > 1)

    name_uninterpretable = name_decode_failures
    chain_disagreements = 0
    residue_disagreements = 0
    suffixes: Counter[str] = Counter()
    parsed_name_count = 0
    for index, name in enumerate(names):
        if name is None:
            continue
        match = NAME_TAIL_RE.search(name)
        if not match:
            name_uninterpretable += 1
            continue
        parsed_name_count += 1
        parsed_chain = match.group("chain")
        parsed_residue = int(match.group("number"))
        suffix = match.group("suffix")
        if suffix:
            suffixes[suffix] += 1
        if index < len(chain_values) and chain_values[index] is not None:
            chain_disagreements += int(parsed_chain != chain_values[index])
        if index < len(res_values) and integral[index]:
            residue_disagreements += int(parsed_residue != int(res_values[index]))

    case_chain, case_position = parse_case_key(case_key)
    anchor_dataset = (
        graph_features.get("anchor_position")
        if isinstance(graph_features, h5py.Group)
        else None
    )
    anchor_position = _scalar_number(anchor_dataset)
    anchor_integer = _safe_int(anchor_position) if anchor_position is not None else None

    def count_matches(chain: str | None, position: int | None) -> int:
        if chain is None or position is None or not chain_values or not res_values.size:
            return 0
        return sum(
            candidate_chain == chain and valid and int(residue) == position
            for candidate_chain, residue, valid in zip(chain_values, res_values, integral)
        )

    variant_match_count = count_matches(case_chain, case_position)
    anchor_match_count = count_matches(case_chain, anchor_integer)
    positions_agree = (
        case_position is not None
        and anchor_integer is not None
        and case_position == anchor_integer
    )

    observed_chains = Counter(value for value in chain_values if value is not None)
    return {
        "hdf5_path": str(Path(hdf5_path).resolve()),
        "role": role,
        "case_key": case_key,
        "n_nodes": n_nodes,
        "node_datasets": node_fields,
        "has_chain_id": datasets["_chain_id"] is not None,
        "has_res_id": datasets["res_id"] is not None,
        "has_name": datasets["_name"] is not None,
        "insertion_code_fields": insertion_fields,
        "has_explicit_insertion_code": bool(insertion_fields),
        "has_is_mutation": datasets["is_mutation"] is not None,
        "identity_fields": field_metadata,
        "observed_chains": dict(sorted(observed_chains.items())),
        "empty_chain_count": sum(value == "" for value in chain_values),
        "chain_with_spaces_count": sum(
            value is not None and any(char.isspace() for char in value)
            for value in chain_values
        ),
        "undecodable_chain_count": chain_decode_failures,
        "res_id_count": int(res_values.size),
        "nonfinite_res_id_count": int((~finite).sum()),
        "noninteger_res_id_count": int((finite & ~integral).sum()),
        "negative_res_id_count": int((finite & (res_values < 0)).sum()),
        "zero_res_id_count": int((finite & (res_values == 0)).sum()),
        "min_res_id": float(res_values[finite].min()) if finite.any() else None,
        "max_res_id": float(res_values[finite].max()) if finite.any() else None,
        "duplicate_candidate_key_count": duplicate_key_count,
        "unique_candidate_key_count": len(key_counts),
        "uninterpretable_name_count": name_uninterpretable,
        "undecodable_name_count": name_decode_failures,
        "parsed_name_count": parsed_name_count,
        "name_chain_disagreement_count": chain_disagreements,
        "name_res_id_disagreement_count": residue_disagreements,
        "name_number_suffixes": dict(sorted(suffixes.items())),
        "anchor_position": anchor_position,
        "anchor_position_field": _dataset_description(anchor_dataset),
        "case_key_chain": case_chain,
        "case_key_position": case_position,
        "variant_chain_position_match_count": variant_match_count,
        "anchor_chain_position_match_count": anchor_match_count,
        "case_key_anchor_position_agree": positions_agree,
        "unique_variant_anchor": positions_agree and anchor_match_count == 1,
        "identity_fields_coherent": bool(
            datasets["_chain_id"] is not None
            and datasets["res_id"] is not None
            and datasets["_name"] is not None
            and name_uninterpretable == 0
            and chain_disagreements == 0
            and residue_disagreements == 0
            and len(chain_values) == len(res_values) == len(names)
        ),
    }


def audit_hdf5(path: str | Path, role: str) -> list[dict[str, Any]]:
    """Audit every root graph in an HDF5 opened explicitly read-only."""

    resolved = Path(path).resolve()
    records: list[dict[str, Any]] = []
    with h5py.File(resolved, "r") as handle:
        for case_key in sorted(handle.keys()):
            obj = handle[case_key]
            if not isinstance(obj, h5py.Group):
                continue
            records.append(
                audit_graph(obj, hdf5_path=resolved, role=role, case_key=case_key)
            )
    return records


def _anomaly_types(record: Mapping[str, Any]) -> list[str]:
    checks = {
        "missing_chain_id": not record["has_chain_id"],
        "missing_res_id": not record["has_res_id"],
        "missing_name": not record["has_name"],
        "empty_chain": record["empty_chain_count"] > 0,
        "chain_with_spaces": record["chain_with_spaces_count"] > 0,
        "undecodable_chain": record["undecodable_chain_count"] > 0,
        "nonfinite_res_id": record["nonfinite_res_id_count"] > 0,
        "noninteger_res_id": record["noninteger_res_id_count"] > 0,
        "negative_res_id": record["negative_res_id_count"] > 0,
        "zero_res_id": record["zero_res_id_count"] > 0,
        "duplicate_candidate_key": record["duplicate_candidate_key_count"] > 0,
        "uninterpretable_name": record["uninterpretable_name_count"] > 0,
        "name_chain_disagreement": record["name_chain_disagreement_count"] > 0,
        "name_res_id_disagreement": record["name_res_id_disagreement_count"] > 0,
        "case_key_anchor_disagreement": not record["case_key_anchor_position_agree"],
        "missing_anchor_node": record["anchor_chain_position_match_count"] == 0,
        "multiple_anchor_nodes": record["anchor_chain_position_match_count"] > 1,
    }
    return [name for name, applies in checks.items() if applies]


def _summarise(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    node_frequency: Counter[str] = Counter()
    chain_frequency: Counter[str] = Counter()
    suffix_frequency: Counter[str] = Counter()
    anomaly_examples: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        node_frequency.update(row["node_datasets"])
        chain_frequency.update(row["observed_chains"])
        suffix_frequency.update(row["name_number_suffixes"])
        for anomaly in _anomaly_types(row):
            if len(anomaly_examples[anomaly]) < MAX_ANOMALY_EXAMPLES:
                anomaly_examples[anomaly].append(
                    {"role": row["role"], "case_key": row["case_key"]}
                )
    return {
        "total_graphs": len(rows),
        "total_nodes": sum(row["n_nodes"] for row in rows),
        "min_nodes_per_graph": min((row["n_nodes"] for row in rows), default=None),
        "max_nodes_per_graph": max((row["n_nodes"] for row in rows), default=None),
        "mean_nodes_per_graph": (
            sum(row["n_nodes"] for row in rows) / len(rows) if rows else None
        ),
        "node_dataset_frequency": dict(sorted(node_frequency.items())),
        "graphs_without_chain_id": sum(not row["has_chain_id"] for row in rows),
        "graphs_without_res_id": sum(not row["has_res_id"] for row in rows),
        "graphs_without_name": sum(not row["has_name"] for row in rows),
        "graphs_with_explicit_insertion_code": sum(row["has_explicit_insertion_code"] for row in rows),
        "graphs_without_explicit_insertion_code": sum(not row["has_explicit_insertion_code"] for row in rows),
        "graphs_with_is_mutation": sum(row["has_is_mutation"] for row in rows),
        "chain_frequency": dict(sorted(chain_frequency.items())),
        "name_number_suffix_frequency": dict(sorted(suffix_frequency.items())),
        "empty_chain_values": sum(row["empty_chain_count"] for row in rows),
        "graphs_with_empty_chains": sum(row["empty_chain_count"] > 0 for row in rows),
        "graphs_with_multiple_chains": sum(len(row["observed_chains"]) > 1 for row in rows),
        "graphs_with_noninteger_res_id": sum(row["noninteger_res_id_count"] > 0 for row in rows),
        "graphs_with_nonfinite_res_id": sum(row["nonfinite_res_id_count"] > 0 for row in rows),
        "graphs_with_duplicate_candidate_keys": sum(row["duplicate_candidate_key_count"] > 0 for row in rows),
        "graphs_with_uninterpretable_names": sum(row["uninterpretable_name_count"] > 0 for row in rows),
        "graphs_with_name_chain_disagreements": sum(row["name_chain_disagreement_count"] > 0 for row in rows),
        "graphs_with_name_res_id_disagreements": sum(row["name_res_id_disagreement_count"] > 0 for row in rows),
        "graphs_with_coherent_identity_fields": sum(row["identity_fields_coherent"] for row in rows),
        "graphs_with_unique_variant_anchor": sum(row["unique_variant_anchor"] for row in rows),
        "graphs_without_anchor_node": sum(row["anchor_chain_position_match_count"] == 0 for row in rows),
        "graphs_with_multiple_anchor_nodes": sum(row["anchor_chain_position_match_count"] > 1 for row in rows),
        "anomaly_examples": dict(sorted(anomaly_examples.items())),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    roles = sorted({record["role"] for record in records})
    by_role = {role: _summarise(r for r in records if r["role"] == role) for role in roles}
    comparison_metrics = [
        "total_graphs",
        "graphs_with_explicit_insertion_code",
        "graphs_with_empty_chains",
        "graphs_with_noninteger_res_id",
        "graphs_with_nonfinite_res_id",
        "graphs_with_duplicate_candidate_keys",
        "graphs_with_coherent_identity_fields",
        "graphs_with_unique_variant_anchor",
        "graphs_without_anchor_node",
        "graphs_with_multiple_anchor_nodes",
    ]
    differences = {
        metric: {role: by_role[role][metric] for role in roles}
        for metric in comparison_metrics
    }
    return {"overall": _summarise(records), "by_role": by_role, "role_comparison": differences}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_fingerprint(
    inputs: Mapping[str, str | Path], *, command: list[str], cwd: str | Path
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for role, path in inputs.items():
        resolved = Path(path).resolve()
        with h5py.File(resolved, "r") as handle:
            root_groups = sum(isinstance(handle[key], h5py.Group) for key in handle.keys())
        files[role] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
            "root_group_count": root_groups,
        }
    return {
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(cwd)),
        "python_version": platform.python_version(),
        "h5py_version": h5py.__version__,
        "command": command,
        "files": files,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_audit_artifacts(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    fingerprint: Mapping[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output / "node_identity_audit.csv",
        "json": output / "node_identity_audit.json",
        "summary": output / "node_identity_summary.json",
        "anomalies": output / "node_identity_anomalies.csv",
        "fingerprint": output / "dataset_fingerprint.json",
    }
    ordered_records = sorted(records, key=lambda row: (row["role"], row["case_key"]))
    fieldnames = list(ordered_records[0]) if ordered_records else []
    with paths["csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in ordered_records)

    anomaly_rows = [
        {
            "hdf5_path": row["hdf5_path"],
            "role": row["role"],
            "case_key": row["case_key"],
            "anomaly": anomaly,
        }
        for row in ordered_records
        for anomaly in _anomaly_types(row)
    ]
    with paths["anomalies"].open("w", encoding="utf-8", newline="") as stream:
        fields = ["hdf5_path", "role", "case_key", "anomaly"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(anomaly_rows)

    payloads = {
        paths["json"]: {"schema_version": 1, "graphs": ordered_records},
        paths["summary"]: {"schema_version": 1, **build_summary(ordered_records)},
        paths["fingerprint"]: dict(fingerprint),
    }
    for path, payload in payloads.items():
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return paths
