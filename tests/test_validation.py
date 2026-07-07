from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py

from gnn_siamese.data.validation import (
    CASE_KIND_MISSENSE,
    CASE_KIND_TRUNCATION,
    CASE_KIND_WT_COMPANION,
    DIFF_PROBES_FOR_MUTATION,
    audit_hdf5_case,
    audit_hdf5_file,
    audit_mut_wt_pairing,
    build_feature_summary,
    build_summary_by_reason,
    detect_available_targets,
    expand_hdf5_inputs,
    infer_is_mutation_from_diff,
    validate_encoder_feature_policy,
    write_audit_csv,
    write_audit_json,
    write_summary_by_reason_csv,
)


def _create_case(
    file_path: Path,
    case_key: str,
    *,
    diff_mutated_nodes: list[int],
    is_truncation: float = 0.0,
    include_custom_structure_energy: bool = True,
    include_custom_complex_energy_phenotype: bool = False,
    include_var_features: bool = True,
    diff_polarity_values: list[bytes] | None = None,
    diff_mass_as_column_vector: bool = False,
    explicit_is_mutation: list[float] | None = None,
    edge_index_data: list[list[int]] | None = None,
    edge_index_orientation: str = "E,2",
    inject_nan_in_bsa: bool = False,
) -> None:
    with h5py.File(file_path, "w") as handle:
        group = handle.create_group(case_key)
        node_group = group.create_group("node_features")
        edge_group = group.create_group("edge_features")
        graph_group = group.create_group("graph_features")

        num_nodes = 4
        num_edges = 4
        mutation_signal = [1.0 if index in diff_mutated_nodes else 0.0 for index in range(num_nodes)]

        node_group.create_dataset("_name", data=[b"ALA", b"GLY", b"SER", b"VAL"])
        node_group.create_dataset("_chain_id", data=[b"A", b"A", b"A", b"A"])
        node_group.create_dataset(
            "_position",
            data=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
        )
        bsa_values = [0.1, 0.2, 0.3, 0.4]
        if inject_nan_in_bsa:
            bsa_values[2] = float("nan")
        node_group.create_dataset("bsa", data=bsa_values)
        for probe in DIFF_PROBES_FOR_MUTATION:
            if probe == "diff_polarity" and diff_polarity_values is not None:
                node_group.create_dataset(probe, data=diff_polarity_values)
                continue
            if probe == "diff_mass" and diff_mass_as_column_vector:
                node_group.create_dataset(probe, data=[[value] for value in mutation_signal])
                continue
            node_group.create_dataset(probe, data=mutation_signal)
        if explicit_is_mutation is not None:
            node_group.create_dataset("is_mutation", data=explicit_is_mutation)
        node_group.create_dataset("mask_diff_mass", data=[1.0, 1.0, 1.0, 1.0])
        if include_var_features:
            node_group.create_dataset("var_HSE", data=[0.1, 0.1, 0.2, 0.2])
            node_group.create_dataset("var_SASA", data=[0.5, 0.5, 0.6, 0.7])

        edge_index = edge_index_data or [[0, 1], [1, 2], [2, 3], [3, 0]]
        if edge_index_orientation == "2,E":
            edge_index = [
                [row[0] for row in edge_index],
                [row[1] for row in edge_index],
            ]
        edge_group.create_dataset("_index", data=edge_index)
        edge_group.create_dataset("distance", data=[3.0, 4.0, 5.0, 6.0])
        edge_group.create_dataset("covalent", data=[1.0, 0.0, 0.0, 1.0])

        graph_group.create_dataset("graph_num_nodes", data=4.0)
        graph_group.create_dataset("graph_num_edges", data=4.0)
        graph_group.create_dataset("is_truncation", data=is_truncation)
        if include_custom_structure_energy:
            graph_group.create_dataset("custom_structure_energy", data=2.5)
        if include_custom_complex_energy_phenotype:
            graph_group.create_dataset("custom_complex_energy_phenotype", data=0.5)


def test_infer_is_mutation_from_diff_marks_single_node() -> None:
    node_features = {
        "diff_mass": [0.0, 1.0, 0.0],
        "diff_charge": [0.0, 0.0, 0.0],
    }
    assert infer_is_mutation_from_diff(node_features) == [0, 1, 0]


def test_infer_is_mutation_from_diff_accepts_column_vector() -> None:
    node_features = {
        "diff_mass": [[0.0], [1.0], [0.0]],
        "diff_charge": [0.0, 0.0, 0.0],
    }
    assert infer_is_mutation_from_diff(node_features) == [0, 1, 0]


def test_detect_available_targets_warns_but_does_not_fail(caplog) -> None:
    with caplog.at_level("WARNING"):
        targets = detect_available_targets({"custom_structure_energy": 1.0})
    assert targets == {
        "custom_structure_energy": True,
        "custom_complex_energy_phenotype": False,
    }
    assert (
        "custom_complex_energy_phenotype not found; supervised phenotype training disabled."
        in caplog.text
    )


def test_audit_hdf5_case_accepts_missense_with_single_mutated_node(tmp_path: Path) -> None:
    path = tmp_path / "proc_483p.hdf5"
    case_key = "residue-srv:A:483:Proline->Serine:pos_483_P_S"
    _create_case(path, case_key, diff_mutated_nodes=[2])

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is True
    assert result["case_kind"] == CASE_KIND_MISSENSE
    assert result["mutation_node_count"] == 1
    assert result["edge_index"]["shape"] == [4, 2]
    assert result["edge_index"]["pyg_shape"] == [2, 4]
    assert "var_HSE" in result["available_var_features"]


def test_audit_hdf5_case_accepts_wt_companion_with_zero_mutations(tmp_path: Path) -> None:
    path = tmp_path / "wt_companion.hdf5"
    case_key = "residue-srv:A:483:Proline->Proline:PKP2_WT"
    _create_case(path, case_key, diff_mutated_nodes=[])

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is True
    assert result["case_kind"] == CASE_KIND_WT_COMPANION
    assert result["mutation_node_count"] == 0


def test_audit_hdf5_case_accepts_truncation_with_zero_mutations(tmp_path: Path) -> None:
    path = tmp_path / "proc_stop.hdf5"
    case_key = "residue-srv:A:510:Glutamine->STOP:pos_510_Q_STOP"
    _create_case(path, case_key, diff_mutated_nodes=[], is_truncation=1.0)

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is True
    assert result["case_kind"] == CASE_KIND_TRUNCATION
    assert result["mutation_node_count"] == 0


def test_audit_hdf5_case_rejects_invalid_missense_with_two_mutated_nodes(tmp_path: Path) -> None:
    path = tmp_path / "proc_invalid.hdf5"
    case_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_case(path, case_key, diff_mutated_nodes=[1, 2])

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is False
    assert "missense case expects 1 mutated nodes, got 2.".lower() in " ".join(
        message.lower() for message in result["errors"]
    )


def test_audit_hdf5_case_rejects_nan_node_feature(tmp_path: Path) -> None:
    path = tmp_path / "proc_nan.hdf5"
    case_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_case(path, case_key, diff_mutated_nodes=[1], inject_nan_in_bsa=True)

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is False
    assert any("node_features.bsa contains NaN or Inf." in error for error in result["errors"])


def test_audit_hdf5_case_accepts_edge_index_in_pyg_orientation(tmp_path: Path) -> None:
    path = tmp_path / "proc_pyg.hdf5"
    case_key = "residue-srv:A:200:Alanine->Valine:pos_200_A_V"
    _create_case(path, case_key, diff_mutated_nodes=[0], edge_index_orientation="2,E")

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is True
    assert result["edge_index"]["orientation"] == "2,E"
    assert result["edge_index"]["pyg_shape"] == [2, 4]


def test_audit_hdf5_case_rejects_invalid_edge_index(tmp_path: Path) -> None:
    path = tmp_path / "proc_bad_edge.hdf5"
    case_key = "residue-srv:A:101:Glycine->Aspartate:pos_101_G_D"
    _create_case(path, case_key, diff_mutated_nodes=[1], edge_index_data=[[0, 1], [1, 9]])

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is False
    assert any("out-of-range node index" in error for error in result["errors"])


def test_audit_hdf5_case_rejects_wt_companion_with_nonzero_explicit_is_mutation(tmp_path: Path) -> None:
    path = tmp_path / "wt_invalid.hdf5"
    case_key = "residue-srv:A:483:Proline->Proline:PKP2_WT"
    _create_case(path, case_key, diff_mutated_nodes=[], explicit_is_mutation=[0.0, 1.0, 0.0, 0.0])

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is False
    assert result["mutation_mask_source"] == "explicit"
    assert any("wt_companion case expects 0 mutated nodes, got 1." in error for error in result["errors"])


def test_wt_companion_with_nonnumeric_diff_polarity_is_valid_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "wt_companion.hdf5"
    case_key = "residue-srv:A:483:Proline->Proline:PKP2_WT"
    _create_case(
        path,
        case_key,
        diff_mutated_nodes=[],
        diff_polarity_values=[b"same", b"same", b"same", b"same"],
    )

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is True
    assert result["mutation_node_count"] == 0
    assert "diff_polarity" in result["nonnumeric_node_features"]
    assert "diff_polarity" in result["recommended_excluded_features"]
    assert any("diff_polarity omitted from mutation inference" in warning for warning in result["warnings"])


def test_missense_with_nonnumeric_diff_polarity_and_numeric_diff_mass_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "proc_483p.hdf5"
    case_key = "residue-srv:A:483:Proline->Serine:pos_483_P_S"
    _create_case(
        path,
        case_key,
        diff_mutated_nodes=[1],
        diff_polarity_values=[b"same", b"changed", b"same", b"same"],
        diff_mass_as_column_vector=True,
    )

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features={name: dataset[()] for name, dataset in group["node_features"].items()},
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is True
    assert result["mutation_node_count"] == 1
    assert "diff_mass" in result["used_numeric_mutation_probes"]
    assert result["skipped_mutation_probes"]["diff_polarity"].startswith(
        "diff_polarity contains non-numeric data"
    )


def test_missense_with_only_nonnumeric_diff_polarity_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "proc_483p.hdf5"
    case_key = "residue-srv:A:483:Proline->Serine:pos_483_P_S"
    _create_case(
        path,
        case_key,
        diff_mutated_nodes=[],
        diff_polarity_values=[b"same", b"changed", b"same", b"same"],
    )

    with h5py.File(path, "r") as handle:
        group = handle[case_key]
        node_features = {name: dataset[()] for name, dataset in group["node_features"].items()}
        for probe in ("diff_mass", "diff_charge", "diff_pI", "diff_size", "diff_hb_donors", "diff_hb_acceptors"):
            del node_features[probe]
        result = audit_hdf5_case(
            source_path=str(path),
            case_key=case_key,
            node_features=node_features,
            edge_features={name: dataset[()] for name, dataset in group["edge_features"].items()},
            graph_features={name: dataset[()] for name, dataset in group["graph_features"].items()},
        )

    assert result["valid"] is False
    assert result["mutation_node_count"] == 0
    assert "diff_polarity" in result["nonnumeric_node_features"]
    assert any("missense case expects 1 mutated nodes, got 0." in error.lower() for error in result["errors"])


def test_validate_encoder_feature_policy_errors_on_selected_nonnumeric_feature() -> None:
    node_features = {
        "diff_mass": [0.0, 1.0, 0.0],
        "diff_polarity": [b"same", b"changed", b"same"],
    }
    policy = validate_encoder_feature_policy(
        node_features=node_features,
        selected_node_features=["diff_polarity"],
        excluded_node_features=[],
        explicit_numeric_mappings={},
    )

    assert policy["errors"]
    assert "diff_polarity" in policy["nonnumeric_node_features"]
    assert "diff_polarity" in policy["recommended_excluded_features"]


def test_validate_encoder_feature_policy_warns_on_unselected_nonnumeric_feature() -> None:
    node_features = {
        "diff_mass": [0.0, 1.0, 0.0],
        "diff_polarity": [b"same", b"changed", b"same"],
    }
    policy = validate_encoder_feature_policy(
        node_features=node_features,
        selected_node_features=["diff_mass"],
        excluded_node_features=["diff_polarity"],
        explicit_numeric_mappings={},
    )

    assert policy["errors"] == []
    assert policy["warnings"]
    assert "diff_polarity" in policy["recommended_excluded_features"]


def test_audit_file_reports_missing_supervised_target_as_warning(tmp_path: Path) -> None:
    path = tmp_path / "proc_483p.hdf5"
    case_key = "residue-srv:A:483:Proline->Serine:pos_483_P_S"
    _create_case(path, case_key, diff_mutated_nodes=[0], include_custom_complex_energy_phenotype=False)

    rows = audit_hdf5_file(
        path,
        dataset_role="mutant",
        diff_probes=DIFF_PROBES_FOR_MUTATION,
        require_explicit_is_mutation=False,
        require_custom_complex_energy_phenotype=False,
        expected_missense_mutation_nodes=1,
        expected_wt_mutation_nodes=0,
        expected_truncation_mutation_nodes=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["valid"] is True
    assert row["available_targets"]["custom_complex_energy_phenotype"] is False
    assert any(
        "custom_complex_energy_phenotype not found; supervised phenotype training disabled."
        in warning
        for warning in row["warnings"]
    )
    assert "nonnumeric_node_features" in row
    assert "recommended_excluded_features" in row


def test_audit_pairing_rejects_missing_wt_companion(tmp_path: Path) -> None:
    mut_path = tmp_path / "proc_missing_wt.hdf5"
    case_key = "residue-srv:A:250:Alanine->Valine:pos_250_A_V"
    _create_case(mut_path, case_key, diff_mutated_nodes=[1])

    mut_rows = audit_hdf5_file(
        mut_path,
        dataset_role="mutant",
        diff_probes=DIFF_PROBES_FOR_MUTATION,
        require_explicit_is_mutation=False,
        require_custom_complex_energy_phenotype=False,
        expected_missense_mutation_nodes=1,
        expected_wt_mutation_nodes=0,
        expected_truncation_mutation_nodes=0,
    )
    pairing = audit_mut_wt_pairing(mut_rows, [])

    assert pairing["coverage_complete"] is False
    assert pairing["missing_wt_companion"][0]["position"] == 250
    assert mut_rows[0]["valid"] is False
    assert any("No WT companion found" in error for error in mut_rows[0]["errors"])


def test_audit_outputs_are_generated(tmp_path: Path) -> None:
    mut_path = tmp_path / "proc_483p.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    _create_case(mut_path, "residue-srv:A:483:Proline->Serine:pos_483_P_S", diff_mutated_nodes=[1])
    _create_case(wt_path, "residue-srv:A:483:Proline->Proline:PKP2_WT", diff_mutated_nodes=[])

    mut_rows = audit_hdf5_file(
        mut_path,
        dataset_role="mutant",
        diff_probes=DIFF_PROBES_FOR_MUTATION,
        require_explicit_is_mutation=False,
        require_custom_complex_energy_phenotype=False,
        expected_missense_mutation_nodes=1,
        expected_wt_mutation_nodes=0,
        expected_truncation_mutation_nodes=0,
    )
    wt_rows = audit_hdf5_file(
        wt_path,
        dataset_role="wt_companion",
        diff_probes=DIFF_PROBES_FOR_MUTATION,
        require_explicit_is_mutation=False,
        require_custom_complex_energy_phenotype=False,
        expected_missense_mutation_nodes=1,
        expected_wt_mutation_nodes=0,
        expected_truncation_mutation_nodes=0,
    )
    rows = [*mut_rows, *wt_rows]
    pairing = audit_mut_wt_pairing(mut_rows, wt_rows)

    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    dataset_json = output_dir / "dataset_audit.json"
    dataset_csv = output_dir / "dataset_audit.csv"
    rejected_csv = output_dir / "rejected_cases.csv"
    summary_by_reason_csv = output_dir / "summary_by_reason.csv"
    feature_summary_json = output_dir / "feature_summary.json"
    pairing_summary_json = output_dir / "pairing_summary.json"

    write_audit_json(dataset_json, rows)
    write_audit_csv(dataset_csv, rows)
    write_audit_csv(rejected_csv, rows, rejected_only=True)
    write_summary_by_reason_csv(summary_by_reason_csv, build_summary_by_reason(rows))
    summary = build_feature_summary(rows, diff_probes=DIFF_PROBES_FOR_MUTATION)
    summary["pairing"] = pairing
    write_audit_json(feature_summary_json, summary)
    write_audit_json(pairing_summary_json, pairing)

    assert dataset_json.exists()
    assert dataset_csv.exists()
    assert rejected_csv.exists()
    assert summary_by_reason_csv.exists()
    assert feature_summary_json.exists()
    assert pairing_summary_json.exists()

    summary = json.loads(feature_summary_json.read_text(encoding="utf-8"))
    assert summary["detected_targets"]["custom_structure_energy"] is True
    assert summary["detected_targets"]["custom_complex_energy_phenotype"] is False
    assert "nonnumeric_node_features" in summary
    assert "recommended_excluded_features" in summary
    assert "mask_features" in summary
    assert summary["pairing"]["coverage_complete"] is True

    with dataset_csv.open(encoding="utf-8", newline="") as handle:
        rows_csv = list(csv.DictReader(handle))
    assert len(rows_csv) == 2
    assert "nonnumeric_node_features" in rows_csv[0]
    assert "recommended_excluded_features" in rows_csv[0]
    assert "dataset_role" in rows_csv[0]

    with rejected_csv.open(encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))
    assert rejected_rows == []

    with summary_by_reason_csv.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    assert set(summary_rows[0]) == {"reason", "count"}


def test_write_summary_by_reason_csv_generates_header_without_rejections(tmp_path: Path) -> None:
    output_path = tmp_path / "summary_by_reason.csv"

    write_summary_by_reason_csv(output_path, build_summary_by_reason([]))

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").splitlines() == ["reason,count"]


def test_build_summary_by_reason_counts_synthetic_errors_and_warnings() -> None:
    rows = [
        {
            "errors": ["missing wt", "bad edge"],
            "warnings": ["soft issue"],
        },
        {
            "errors": ["missing wt"],
            "warnings": ["soft issue", "watch this"],
        },
    ]

    summary_rows = build_summary_by_reason(rows)

    assert summary_rows == [
        {"reason": "bad edge", "count": 1},
        {"reason": "missing wt", "count": 2},
        {"reason": "soft issue", "count": 2},
        {"reason": "watch this", "count": 1},
    ]


def test_expand_hdf5_inputs_supports_globs_without_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "a.hdf5"
    second = tmp_path / "b.hdf5"
    first.touch()
    second.touch()

    expanded = expand_hdf5_inputs([str(tmp_path / "*.hdf5"), str(first)])

    assert expanded == [first, second]


def test_cli_audits_individual_sample_hdf5_without_wt_pairing(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit_sample"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_dataset.py",
            "--hdf5",
            "sample_data/examples/*.hdf5",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "feature_summary.json").read_text(encoding="utf-8"))
    pairing = json.loads((output_dir / "pairing_summary.json").read_text(encoding="utf-8"))
    with (output_dir / "summary_by_reason.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert summary["case_counts"]["total"] == 3
    assert pairing["mutant_cases_checked"] == 0
    assert pairing["wt_cases_checked"] == 0
    assert pairing["coverage_complete"] is True
    assert summary_rows
    assert set(summary_rows[0]) == {"reason", "count"}
