from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from gnn_siamese.data.node_pair_alignment_audit import (
    KEY_POLICY,
    audit_hdf5_pairs,
    audit_pair_graphs,
    build_alignment_summary,
    write_alignment_artifacts,
)


MUT_KEY = "residue-srv:A:10:Glycine->Serine:pos_10_G_S"
WT_KEY = "residue-srv:A:10:Glycine->Glycine:PKP2_WT"


def _names(chains, residues):
    values = []
    for chain, residue in zip(chains, residues):
        if isinstance(chain, bytes):
            try:
                chain_text = chain.decode("utf-8")
            except UnicodeDecodeError:
                chain_text = "A"
        else:
            chain_text = str(chain)
        residue_text = (
            str(int(residue))
            if np.isfinite(residue) and float(residue).is_integer()
            else str(residue)
        )
        values.append(f"sample {chain_text.strip() or 'A'} {residue_text}".encode())
    return values


def _create_graph(
    handle: h5py.File,
    key: str,
    *,
    chains=(b"A", b"A", b"A"),
    residues=(9.0, 10.0, 11.0),
    names=None,
) -> h5py.Group:
    graph = handle.create_group(key)
    nodes = graph.create_group("node_features")
    nodes.create_dataset("_chain_id", data=np.asarray(chains))
    nodes.create_dataset("res_id", data=np.asarray(residues))
    nodes.create_dataset(
        "_name",
        data=np.asarray(_names(chains, residues) if names is None else names),
    )
    return graph


def _audit(
    tmp_path: Path,
    *,
    mut_chains=(b"A", b"A", b"A"),
    mut_residues=(9.0, 10.0, 11.0),
    mut_names=None,
    wt_chains=(b"A", b"A", b"A"),
    wt_residues=(9.0, 10.0, 11.0),
    wt_names=None,
):
    path = tmp_path / "pair.hdf5"
    with h5py.File(path, "w") as handle:
        mut = _create_graph(
            handle,
            "mut",
            chains=mut_chains,
            residues=mut_residues,
            names=mut_names,
        )
        wt = _create_graph(
            handle,
            "wt",
            chains=wt_chains,
            residues=wt_residues,
            names=wt_names,
        )
        return audit_pair_graphs(
            mut,
            wt,
            variant_id=MUT_KEY,
            mut_graph_id=MUT_KEY,
            wt_graph_id=WT_KEY,
        )


def test_perfect_alignment(tmp_path):
    row = _audit(tmp_path)
    assert row["alignment_status"] == "valid"
    assert row["aligned_keys"] == [["A", 9], ["A", 10], ["A", 11]]
    assert row["mut_aligned_index"] == [0, 1, 2]
    assert row["wt_aligned_index"] == [0, 1, 2]
    assert all(row[name] == 1.0 for name in (
        "coverage_union", "coverage_mut", "coverage_wt", "coverage_min", "coverage_max"
    ))
    assert row["key_policy"] == KEY_POLICY
    assert row["local_scale_status"] == "not_assessed"
    assert row["domain_scale_status"] == "not_assessed"


def test_same_keys_in_different_row_order(tmp_path):
    row = _audit(
        tmp_path,
        wt_residues=(11.0, 9.0, 10.0),
    )
    assert row["alignment_status"] == "valid"
    assert row["aligned_keys"] == [["A", 9], ["A", 10], ["A", 11]]
    assert row["mut_aligned_index"] == [0, 1, 2]
    assert row["wt_aligned_index"] == [1, 2, 0]


def test_graphs_with_different_lengths(tmp_path):
    row = _audit(
        tmp_path,
        wt_chains=(b"A", b"A", b"A", b"A"),
        wt_residues=(8.0, 9.0, 10.0, 11.0),
    )
    assert row["alignment_status"] == "partial"
    assert row["wt_only_keys"] == [["A", 8]]
    assert row["coverage_union"] == 0.75
    assert row["coverage_mut"] == 1.0


def test_residue_only_in_wt(tmp_path):
    row = _audit(
        tmp_path,
        wt_chains=(b"A", b"A", b"A", b"A"),
        wt_residues=(9.0, 10.0, 11.0, 12.0),
    )
    assert row["wt_only_indices"] == [3]
    assert row["wt_only_keys"] == [["A", 12]]
    assert row["alignment_status"] == "partial"


def test_residue_only_in_mut(tmp_path):
    row = _audit(
        tmp_path,
        mut_chains=(b"A", b"A", b"A", b"A"),
        mut_residues=(9.0, 10.0, 11.0, 12.0),
    )
    assert row["mut_only_indices"] == [3]
    assert row["mut_only_keys"] == [["A", 12]]
    assert row["alignment_status"] == "partial"


def test_shifted_numbering_is_not_corrected(tmp_path):
    row = _audit(
        tmp_path,
        mut_residues=(9.0, 10.0, 11.0),
        wt_residues=(10.0, 11.0, 12.0),
    )
    assert row["aligned_keys"] == [["A", 10], ["A", 11]]
    assert row["mut_only_keys"] == [["A", 9]]
    assert row["wt_only_keys"] == [["A", 12]]
    assert row["alignment_status"] == "partial"


def test_empty_chain_is_rejected_without_defaulting_to_a(tmp_path):
    row = _audit(
        tmp_path,
        mut_chains=(b"A", b" ", b"A"),
        mut_names=(b"sample A 9", b"sample A 10", b"sample A 11"),
    )
    assert row["invalid_key_count_mut"] == 1
    assert "empty_chain" in row["incident_codes"]
    assert row["mutation_anchor_in_mut"] is False
    assert row["alignment_status"] == "rejected"


def test_duplicate_key_in_mut_is_rejected(tmp_path):
    row = _audit(
        tmp_path,
        mut_residues=(10.0, 10.0, 11.0),
    )
    assert row["duplicate_key_count_mut"] == 1
    assert "duplicate_key" in row["incident_codes"]
    assert row["alignment_status"] == "rejected"


def test_duplicate_key_in_wt_is_rejected(tmp_path):
    row = _audit(
        tmp_path,
        wt_residues=(9.0, 10.0, 10.0),
    )
    assert row["duplicate_key_count_wt"] == 1
    assert row["alignment_status"] == "rejected"


def test_non_unique_correspondence_is_reported(tmp_path):
    row = _audit(
        tmp_path,
        mut_residues=(9.0, 10.0, 10.0),
        wt_residues=(9.0, 10.0, 11.0),
    )
    assert "non_unique_correspondence" in row["incident_codes"]
    assert row["mutation_anchor_count_mut"] == 2
    assert row["alignment_status"] == "rejected"


def test_missing_mutation_anchor_is_rejected(tmp_path):
    row = _audit(
        tmp_path,
        mut_residues=(8.0, 9.0, 11.0),
    )
    assert row["mutation_anchor_in_mut"] is False
    assert row["mutation_anchor_aligned"] is False
    assert "mutation_anchor_missing_mut" in row["incident_codes"]
    assert row["alignment_status"] == "rejected"


def test_high_coverage_does_not_hide_missing_mutation_anchor(tmp_path):
    common = tuple(float(value) for value in range(1, 101) if value != 10)
    row = _audit(
        tmp_path,
        mut_chains=(b"A",) * len(common),
        mut_residues=common,
        wt_chains=(b"A",) * 100,
        wt_residues=tuple(float(value) for value in range(1, 101)),
    )
    assert row["coverage_union"] == 0.99
    assert row["mutation_anchor_in_mut"] is False
    assert row["alignment_status"] == "rejected"


def test_partial_coverage_with_aligned_anchor(tmp_path):
    row = _audit(
        tmp_path,
        mut_residues=(8.0, 9.0, 10.0),
        wt_residues=(9.0, 10.0, 11.0),
    )
    assert row["mutation_anchor_aligned"] is True
    assert row["alignment_status"] == "partial"


def test_nan_res_id_is_rejected_and_coverage_denominators_remain_explicit(tmp_path):
    row = _audit(tmp_path, mut_residues=(9.0, np.nan, 11.0))
    assert "nonfinite_res_id" in row["incident_codes"]
    assert row["invalid_key_count_mut"] == 1
    assert row["alignment_status"] == "rejected"


def test_infinite_res_id_is_rejected(tmp_path):
    row = _audit(tmp_path, mut_residues=(9.0, np.inf, 11.0))
    assert "nonfinite_res_id" in row["incident_codes"]
    assert row["alignment_status"] == "rejected"


def test_nonintegral_res_id_is_rejected(tmp_path):
    row = _audit(tmp_path, mut_residues=(9.0, 10.5, 11.0))
    assert "nonintegral_res_id" in row["incident_codes"]
    assert row["alignment_status"] == "rejected"


def test_undecodable_bytes_are_rejected(tmp_path):
    row = _audit(
        tmp_path,
        mut_chains=(b"A", b"\xff", b"A"),
        mut_names=(b"sample A 9", b"sample A 10", b"sample A 11"),
    )
    assert "undecodable_utf8" in row["incident_codes"]
    assert row["alignment_status"] == "rejected"


def test_semantic_result_is_deterministic_when_both_row_orders_change(tmp_path):
    first = _audit(
        tmp_path,
        mut_residues=(11.0, 9.0, 10.0),
        wt_residues=(10.0, 11.0, 9.0),
    )
    second_path = tmp_path / "second"
    second_path.mkdir()
    second = _audit(
        second_path,
        mut_residues=(9.0, 10.0, 11.0),
        wt_residues=(11.0, 9.0, 10.0),
    )
    assert first["aligned_keys"] == second["aligned_keys"]
    assert first["alignment_status"] == second["alignment_status"] == "valid"
    assert {name: first[name] for name in (
        "coverage_union", "coverage_mut", "coverage_wt", "coverage_min", "coverage_max"
    )} == {name: second[name] for name in (
        "coverage_union", "coverage_mut", "coverage_wt", "coverage_min", "coverage_max"
    )}


def test_hdf5_orchestration_excludes_native_wt_and_writes_three_artifacts(tmp_path):
    mut_path = tmp_path / "proc.hdf5"
    wt_path = tmp_path / "wt.hdf5"
    native_key = "residue-srv:A:20:Alanine->Alanine:PKP2_WT"
    with h5py.File(mut_path, "w") as handle:
        _create_graph(handle, MUT_KEY)
        _create_graph(handle, native_key, residues=(19.0, 20.0, 21.0))
    with h5py.File(wt_path, "w") as handle:
        _create_graph(handle, WT_KEY)

    rows = audit_hdf5_pairs(mut_path, wt_path)
    assert len(rows) == 1
    assert rows[0]["variant_id"] == MUT_KEY

    paths = write_alignment_artifacts(tmp_path / "out", rows)
    assert {path.name for path in paths.values()} == {
        "node_pair_alignment_audit.json",
        "node_pair_alignment_audit.csv",
        "node_pair_alignment_summary.json",
    }
    detail = json.loads(paths["json"].read_text())
    summary = json.loads(paths["summary"].read_text())
    assert detail["minimum_coverage"] is None
    assert summary["total_pairs"] == 1
    assert summary["status_counts"] == {"valid": 1, "partial": 0, "rejected": 0}
    assert build_alignment_summary(rows)["primary_coverage_metric"] == "coverage_union"
