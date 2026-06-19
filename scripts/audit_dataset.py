from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SRC_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import h5py  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "scripts/audit_dataset.py requires h5py. Install the 'data' extras or run in the project environment."
    ) from exc

from gnn_siamese.data.validation import (
    CASE_KIND_MISSENSE,
    DIFF_PROBES_FOR_MUTATION,
    audit_mut_wt_pairing,
    audit_hdf5_file,
    build_feature_summary,
    build_summary_by_reason,
    expand_hdf5_inputs,
    write_audit_csv,
    write_audit_json,
    write_summary_by_reason_csv,
)

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one or more project HDF5 files.")
    parser.add_argument("--hdf5", action="append", default=[], help="Generic HDF5 path or glob to audit.")
    parser.add_argument("--mut-hdf5", action="append", default=[], help="Mutant HDF5 path or glob to audit.")
    parser.add_argument(
        "--wt-hdf5", action="append", default=[], help="WT companion HDF5 path or glob to audit."
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dataset_audit",
        help="Directory for dataset_audit.json/csv outputs.",
    )
    parser.add_argument(
        "--schema",
        default="sample_data/sample_schema.json",
        help="Documented schema path included in the audit summary metadata.",
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

    generic_paths = expand_hdf5_inputs(args.hdf5)
    mut_paths = expand_hdf5_inputs(args.mut_hdf5)
    wt_paths = expand_hdf5_inputs(args.wt_hdf5)
    if not generic_paths and not mut_paths and not wt_paths:
        raise SystemExit("Provide at least one HDF5 via --hdf5, --mut-hdf5 or --wt-hdf5.")

    input_paths = [*generic_paths, *mut_paths, *wt_paths]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"HDF5 path(s) not found: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    mut_rows: list[dict[str, Any]] = []
    wt_rows: list[dict[str, Any]] = []
    for dataset_role, paths in [("generic", generic_paths), ("mutant", mut_paths), ("wt_companion", wt_paths)]:
        for path in paths:
            LOGGER.info("Auditing %s (%s)", path, dataset_role)
            file_rows = audit_hdf5_file(
                path,
                dataset_role=dataset_role,
                diff_probes=DIFF_PROBES_FOR_MUTATION,
                require_explicit_is_mutation=args.require_explicit_is_mutation,
                require_custom_complex_energy_phenotype=args.require_custom_complex_energy_phenotype,
                expected_missense_mutation_nodes=1,
                expected_wt_mutation_nodes=0,
                expected_truncation_mutation_nodes=0,
            )
            rows.extend(file_rows)
            if dataset_role == "mutant":
                mut_rows.extend(file_rows)
            elif dataset_role == "wt_companion":
                wt_rows.extend(file_rows)

    pairing_summary = audit_mut_wt_pairing(mut_rows, wt_rows) if (mut_rows or wt_rows) else {
        "mutant_cases_checked": 0,
        "wt_cases_checked": 0,
        "matched_pairs": 0,
        "missing_wt_companion": [],
        "ambiguous_wt_companion": [],
        "coverage_complete": True,
    }

    feature_summary = build_feature_summary(rows, diff_probes=DIFF_PROBES_FOR_MUTATION)
    feature_summary["schema_path"] = str(Path(args.schema))
    feature_summary["pairing"] = pairing_summary
    feature_summary["case_counts"] = {
        "total": len(rows),
        "valid": sum(1 for row in rows if row["valid"]),
        "rejected": sum(1 for row in rows if not row["valid"]),
        "missense": sum(1 for row in rows if row["case_kind"] == CASE_KIND_MISSENSE),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_audit_json = output_dir / "dataset_audit.json"
    dataset_audit_csv = output_dir / "dataset_audit.csv"
    rejected_cases_csv = output_dir / "rejected_cases.csv"
    summary_by_reason_csv = output_dir / "summary_by_reason.csv"
    feature_summary_json = output_dir / "feature_summary.json"
    pairing_summary_json = output_dir / "pairing_summary.json"

    write_audit_json(dataset_audit_json, rows)
    write_audit_csv(dataset_audit_csv, rows)
    write_audit_csv(rejected_cases_csv, rows, rejected_only=True)
    write_summary_by_reason_csv(summary_by_reason_csv, build_summary_by_reason(rows))
    write_audit_json(feature_summary_json, feature_summary)
    write_audit_json(pairing_summary_json, pairing_summary)

    LOGGER.info("Wrote %s", dataset_audit_json)
    LOGGER.info("Wrote %s", dataset_audit_csv)
    LOGGER.info("Wrote %s", rejected_cases_csv)
    LOGGER.info("Wrote %s", summary_by_reason_csv)
    LOGGER.info("Wrote %s", feature_summary_json)
    LOGGER.info("Wrote %s", pairing_summary_json)


if __name__ == "__main__":
    main()
