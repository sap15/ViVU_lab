from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "scripts/audit_dataset.py requires h5py. Install the 'data' extras or run in the project environment."
    ) from exc

from gnn_siamese.data.validation import DIFF_PROBES_FOR_MUTATION, audit_hdf5_case

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _read_group(group: h5py.Group) -> dict[str, Any]:
    return {name: dataset[()] for name, dataset in group.items()}


def audit_file(
    path: Path,
    *,
    diff_probes: list[str],
    require_explicit_is_mutation: bool,
    require_custom_complex_energy_phenotype: bool,
    expected_missense_mutation_nodes: int,
    expected_wt_mutation_nodes: int,
    expected_truncation_mutation_nodes: int,
) -> list[dict[str, Any]]:
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
                        "case_key": case_key,
                        "case_kind": "unknown",
                        "valid": False,
                        "warnings": [],
                        "errors": ["Root group does not contain node_features, edge_features and graph_features."],
                        "node_feature_names": [],
                        "edge_feature_names": [],
                        "graph_feature_names": [],
                        "available_var_features": [],
                        "nonnumeric_node_features": [],
                        "recommended_excluded_features": [],
                        "available_targets": {
                            "custom_structure_energy": False,
                            "custom_complex_energy_phenotype": False,
                        },
                        "num_nodes": 0,
                        "num_edges": 0,
                        "explicit_is_mutation_present": False,
                        "inferred_is_mutation": [],
                        "mutation_node_count": 0,
                        "edge_index": {
                            "shape": None,
                            "orientation": "invalid",
                            "pyg_shape": None,
                            "conversion_note": "Convert edge_features/_index from (E, 2) to (2, E) for PyG.",
                        },
                    }
                )
                continue

            records.append(
                audit_hdf5_case(
                    source_path=str(path),
                    case_key=case_key,
                    node_features=_read_group(group["node_features"]),
                    edge_features=_read_group(group["edge_features"]),
                    graph_features=_read_group(group["graph_features"]),
                    require_explicit_is_mutation=require_explicit_is_mutation,
                    require_custom_complex_energy_phenotype=require_custom_complex_energy_phenotype,
                    diff_probes=diff_probes,
                    expected_missense_mutation_nodes=expected_missense_mutation_nodes,
                    expected_wt_mutation_nodes=expected_wt_mutation_nodes,
                    expected_truncation_mutation_nodes=expected_truncation_mutation_nodes,
                )
            )
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, rejected_only: bool = False) -> None:
    filtered = [row for row in rows if not row["valid"]] if rejected_only else rows
    fieldnames = [
        "source_path",
        "case_key",
        "case_kind",
        "valid",
        "num_nodes",
        "num_edges",
        "mutation_node_count",
        "explicit_is_mutation_present",
        "custom_structure_energy",
        "custom_complex_energy_phenotype",
        "available_var_features",
        "nonnumeric_node_features",
        "recommended_excluded_features",
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
                    "case_key": row["case_key"],
                    "case_kind": row["case_kind"],
                    "valid": row["valid"],
                    "num_nodes": row["num_nodes"],
                    "num_edges": row["num_edges"],
                    "mutation_node_count": row["mutation_node_count"],
                    "explicit_is_mutation_present": row["explicit_is_mutation_present"],
                    "custom_structure_energy": row["available_targets"]["custom_structure_energy"],
                    "custom_complex_energy_phenotype": row["available_targets"][
                        "custom_complex_energy_phenotype"
                    ],
                    "available_var_features": ";".join(row["available_var_features"]),
                    "nonnumeric_node_features": ";".join(row["nonnumeric_node_features"]),
                    "recommended_excluded_features": ";".join(row["recommended_excluded_features"]),
                    "warnings": " | ".join(row["warnings"]),
                    "errors": " | ".join(row["errors"]),
                    "edge_index_shape": row["edge_index"]["shape"],
                    "edge_index_orientation": row["edge_index"]["orientation"],
                    "pyg_edge_index_shape": row["edge_index"]["pyg_shape"],
                }
            )


def _build_feature_summary(rows: list[dict[str, Any]], *, diff_probes: list[str]) -> dict[str, Any]:
    node_features = sorted({name for row in rows for name in row["node_feature_names"]})
    edge_features = sorted({name for row in rows for name in row["edge_feature_names"]})
    graph_features = sorted({name for row in rows for name in row["graph_feature_names"]})
    var_features = sorted({name for row in rows for name in row["available_var_features"]})
    nonnumeric_node_features = sorted({name for row in rows for name in row["nonnumeric_node_features"]})
    recommended_excluded_features = sorted(
        {name for row in rows for name in row["recommended_excluded_features"]}
    )
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "graph_features": graph_features,
        "var_features": var_features,
        "nonnumeric_node_features": nonnumeric_node_features,
        "recommended_excluded_features": recommended_excluded_features,
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
            "edge_index_observed_orientation": "(E, 2)",
            "pyg_conversion": "(2, E)",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one or more project HDF5 files.")
    parser.add_argument("--hdf5", action="append", default=[], help="Generic HDF5 path to audit.")
    parser.add_argument("--mut-hdf5", action="append", default=[], help="Mutant HDF5 path to audit.")
    parser.add_argument("--wt-hdf5", action="append", default=[], help="WT companion HDF5 path to audit.")
    parser.add_argument(
        "--output-dir",
        default="reports/dataset_audit",
        help="Directory for dataset_audit.json/csv outputs.",
    )
    parser.add_argument(
        "--require-explicit-is-mutation",
        action="store_true",
        help="Fail if is_mutation is not explicitly stored in node_features.",
    )
    parser.add_argument(
        "--require-custom-complex-energy-phenotype",
        action="store_true",
        help="Fail if custom_complex_energy_phenotype is missing.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()

    input_paths = [Path(path) for path in [*args.hdf5, *args.mut_hdf5, *args.wt_hdf5]]
    if not input_paths:
        raise SystemExit("Provide at least one HDF5 via --hdf5, --mut-hdf5 or --wt-hdf5.")

    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"HDF5 path(s) not found: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for path in input_paths:
        LOGGER.info("Auditing %s", path)
        rows.extend(
            audit_file(
                path,
                diff_probes=DIFF_PROBES_FOR_MUTATION,
                require_explicit_is_mutation=args.require_explicit_is_mutation,
                require_custom_complex_energy_phenotype=args.require_custom_complex_energy_phenotype,
                expected_missense_mutation_nodes=1,
                expected_wt_mutation_nodes=0,
                expected_truncation_mutation_nodes=0,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_audit_json = output_dir / "dataset_audit.json"
    dataset_audit_csv = output_dir / "dataset_audit.csv"
    rejected_cases_csv = output_dir / "rejected_cases.csv"
    feature_summary_json = output_dir / "feature_summary.json"

    _write_json(dataset_audit_json, rows)
    _write_csv(dataset_audit_csv, rows)
    _write_csv(rejected_cases_csv, rows, rejected_only=True)
    _write_json(feature_summary_json, _build_feature_summary(rows, diff_probes=DIFF_PROBES_FOR_MUTATION))

    LOGGER.info("Wrote %s", dataset_audit_json)
    LOGGER.info("Wrote %s", dataset_audit_csv)
    LOGGER.info("Wrote %s", rejected_cases_csv)
    LOGGER.info("Wrote %s", feature_summary_json)


if __name__ == "__main__":
    main()
