from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from gnn_siamese.data.node_identity_audit import (
    audit_graph,
    audit_hdf5,
    build_summary,
    is_insertion_code_field,
    write_audit_artifacts,
)


def _graph(
    file: h5py.File,
    key: str,
    *,
    chains=(b"A", b"A", b"A"),
    residues=(9.0, 10.0, 11.0),
    names=(b"sample A 9", b"sample A 10", b"sample A 11"),
    anchor=10.0,
    insertion_field: str | None = None,
    include_chain=True,
    include_residue=True,
    diff_polarity_shape=None,
) -> h5py.Group:
    graph = file.create_group(key)
    nodes = graph.create_group("node_features")
    if include_chain:
        nodes.create_dataset("_chain_id", data=np.asarray(chains))
    if include_residue:
        nodes.create_dataset("res_id", data=np.asarray(residues, dtype=float))
    nodes.create_dataset("_name", data=np.asarray(names))
    if insertion_field:
        nodes.create_dataset(insertion_field, data=np.asarray([b"", b"", b""]))
    if diff_polarity_shape:
        nodes.create_dataset("diff_polarity", data=np.zeros(diff_polarity_shape))
    features = graph.create_group("graph_features")
    features.create_dataset("anchor_position", data=anchor)
    return graph


def _audit(tmp_path: Path, **kwargs):
    path = tmp_path / "synthetic.hdf5"
    with h5py.File(path, "w") as file:
        graph = _graph(file, "residue-srv:A:10:Glycine->Serine:pos_10_G_S", **kwargs)
        return audit_graph(graph, hdf5_path=path, role="mutant", case_key=graph.name[1:])


def test_normal_graph_bytes_and_float_integer_residue(tmp_path):
    row = _audit(tmp_path)
    assert row["identity_fields_coherent"] is True
    assert row["observed_chains"] == {"A": 3}
    assert row["noninteger_res_id_count"] == 0
    assert row["unique_variant_anchor"] is True
    assert row["anchor_chain_position_match_count"] == 1


def test_empty_chain_and_missing_identity_fields(tmp_path):
    row = _audit(tmp_path, chains=(b"A", b"", b"A"), include_residue=False)
    assert row["empty_chain_count"] == 1
    assert row["has_res_id"] is False
    assert row["anchor_chain_position_match_count"] == 0

    row = _audit(tmp_path, include_chain=False)
    assert row["has_chain_id"] is False


def test_noninteger_nan_and_duplicate_candidate_key(tmp_path):
    row = _audit(
        tmp_path,
        residues=(10.5, np.nan, 10.0),
        chains=(b"A", b"A", b"A"),
    )
    assert row["noninteger_res_id_count"] == 1
    assert row["nonfinite_res_id_count"] == 1

    row = _audit(
        tmp_path,
        residues=(9.0, 10.0, 10.0),
        names=(b"sample A 9", b"sample A 10", b"sample A 10"),
    )
    assert row["duplicate_candidate_key_count"] == 1
    assert row["anchor_chain_position_match_count"] == 2
    assert row["unique_variant_anchor"] is False


def test_insertion_code_detection_is_descriptive(tmp_path):
    absent = _audit(tmp_path)
    present = _audit(tmp_path, insertion_field="ins_code")
    assert absent["has_explicit_insertion_code"] is False
    assert present["insertion_code_fields"] == ["ins_code"]
    assert is_insertion_code_field("Insertion-Code")
    assert not is_insertion_code_field("position")


def test_name_disagreement_and_anchor_absence(tmp_path):
    row = _audit(
        tmp_path,
        names=(b"sample A 9", b"sample A 99", b"sample A 11"),
        anchor=99.0,
    )
    assert row["name_res_id_disagreement_count"] == 1
    assert row["case_key_anchor_position_agree"] is False
    assert row["anchor_chain_position_match_count"] == 0
    assert row["unique_variant_anchor"] is False


def test_wt_zero_diff_matrix_is_not_used_for_anchor(tmp_path):
    path = tmp_path / "wt.hdf5"
    key = "residue-srv:A:10:Glycine->Glycine:PKP2_WT"
    with h5py.File(path, "w") as file:
        _graph(file, key, diff_polarity_shape=(3, 4))
    rows = audit_hdf5(path, "wt_companion")
    assert len(rows) == 1
    assert rows[0]["anchor_chain_position_match_count"] == 1
    assert "diff_polarity" not in rows[0]["identity_fields"]
    assert rows[0]["has_is_mutation"] is False


def test_summary_and_artifacts_are_deterministic(tmp_path):
    path = tmp_path / "input.hdf5"
    with h5py.File(path, "w") as file:
        _graph(file, "residue-srv:A:10:G->S:x")
        _graph(
            file,
            "residue-srv:A:20:G->S:y",
            residues=(19.0, 21.0, 22.0),
            names=(b"sample A 19", b"sample A 21", b"sample A 22"),
            anchor=20.0,
        )
    records = audit_hdf5(path, "mutant")
    summary = build_summary(records)
    assert summary["by_role"]["mutant"]["total_graphs"] == 2
    assert summary["by_role"]["mutant"]["graphs_with_unique_variant_anchor"] == 1
    assert summary["by_role"]["mutant"]["graphs_without_anchor_node"] == 1

    paths = write_audit_artifacts(tmp_path / "out", records, {"files": {}})
    assert {path.name for path in paths.values()} == {
        "node_identity_audit.csv",
        "node_identity_audit.json",
        "node_identity_summary.json",
        "node_identity_anomalies.csv",
        "dataset_fingerprint.json",
    }
    payload = json.loads(paths["json"].read_text())
    assert [row["case_key"] for row in payload["graphs"]] == sorted(
        row["case_key"] for row in payload["graphs"]
    )

def test_node_identity_audit_cli_help_runs_from_repository_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_node_identity.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "traceback" not in result.stderr.lower()

