from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import h5py
import numpy as np

from gnn_siamese.data.node_pair_alignment_spatial_audit import (
    audit_spatial_hdf5_pairs,
    audit_spatial_pair,
    write_spatial_artifacts,
)
import gnn_siamese.data.node_pair_alignment_spatial_audit as spatial_module

MUT = "residue-srv:A:10:Glycine->Serine:pos_10_G_S"
WT = "residue-srv:A:10:Glycine->Glycine:PKP2_WT"


def _graph(handle, key, residues, xyz):
    graph = handle.create_group(key)
    nodes = graph.create_group("node_features")
    nodes.create_dataset("_chain_id", data=np.asarray([b"A"] * len(residues)))
    nodes.create_dataset("res_id", data=np.asarray(residues, dtype=float))
    nodes.create_dataset("_name", data=np.asarray([f"x A {r}".encode() for r in residues]))
    nodes.create_dataset("_position", data=np.asarray(xyz, dtype=float))
    return graph


def _audit(tmp_path, mut_res=(10, 11, 12), wt_res=(10, 11, 12), mut_xyz=None, wt_xyz=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mut_xyz = mut_xyz if mut_xyz is not None else [(0, 0, 0), (4, 0, 0), (12, 0, 0)]
    wt_xyz = wt_xyz if wt_xyz is not None else [(0, 0, 0), (4, 0, 0), (12, 0, 0)]
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as h:
        mg = _graph(h, "mut", mut_res, mut_xyz)
        wg = _graph(h, "wt", wt_res, wt_xyz)
        return audit_spatial_pair(mg, wg, variant_id=MUT, mut_graph_id=MUT, wt_graph_id=WT)


def test_complete_coverage_all_radii(tmp_path):
    row = _audit(tmp_path)
    assert all(value["coverage_union_r"] == 1 for value in row["radii_metrics"].values())


def test_peripheral_and_near_exclusives(tmp_path):
    peripheral = _audit(tmp_path, mut_res=(10, 11, 12, 13), mut_xyz=[(0,0,0),(4,0,0),(12,0,0),(19,0,0)])
    assert peripheral["flags"]["graph_exclusive_only_peripheral"]
    near = _audit(tmp_path / "near", mut_res=(10, 11, 12, 13), mut_xyz=[(0,0,0),(4,0,0),(12,0,0),(2,0,0)])
    assert near["flags"]["graph_exclusive_near_anchor"]


def test_local_high_global_low_and_reverse(tmp_path):
    row = _audit(tmp_path, mut_res=(10,11,12,13,14), wt_res=(10,11), mut_xyz=[(0,0,0),(2,0,0),(12,0,0),(13,0,0),(14,0,0)], wt_xyz=[(0,0,0),(2,0,0)])
    assert row["flags"]["low_global_high_local"]
    reverse = _audit(tmp_path / "reverse", mut_xyz=[(0,0,0),(2,0,0),(3,0,0)], wt_xyz=[(0,0,0),(12,0,0),(13,0,0)])
    assert reverse["flags"]["high_global_low_local"]


def test_different_coordinates_masks_and_union(tmp_path):
    row = _audit(tmp_path, mut_xyz=[(0,0,0),(4,0,0),(12,0,0)], wt_xyz=[(0,0,0),(12,0,0),(4,0,0)])
    mask = row["descriptive_masks"]["8A"]
    assert mask["mut"]["size"] == mask["wt"]["size"] == 2
    assert mask["symmetric_difference_count"] == 2
    assert mask["union"]["size"] == 3
    assert mask["local_aligned"]["keys"] == [["A", 10], ["A", 11], ["A", 12]]
    assert mask["radial_exclusivity_aligned"]["mut_geometry_only_keys"] == [["A", 11]]
    assert mask["radial_exclusivity_aligned"]["wt_geometry_only_keys"] == [["A", 12]]
    assert mask["K_local_union_keys"] == [["A", 10], ["A", 11], ["A", 12]]
    assert {item["state"] for item in mask["aligned_radial_states"]} == {
        "inside_both",
        "radial_mut_only",
        "radial_wt_only",
    }


def test_ordered_union_presence_and_global_vs_radial_exclusivity(tmp_path):
    row = _audit(
        tmp_path,
        mut_res=(10, 11, 13),
        wt_res=(10, 11, 14),
        mut_xyz=[(0, 0, 0), (4, 0, 0), (3, 0, 0)],
        wt_xyz=[(0, 0, 0), (12, 0, 0), (3, 0, 0)],
    )
    assert row["ordered_residue_union"] == [
        {"key": ["A", 10], "exists_MUT": True, "exists_WT": True, "support": "aligned"},
        {"key": ["A", 11], "exists_MUT": True, "exists_WT": True, "support": "aligned"},
        {"key": ["A", 13], "exists_MUT": True, "exists_WT": False, "support": "graph_mut_only"},
        {"key": ["A", 14], "exists_MUT": False, "exists_WT": True, "support": "graph_wt_only"},
    ]
    assert row["global_aligned_count"] == 2
    assert row["global_mut_only_count"] == 1
    assert row["global_wt_only_count"] == 1
    assert row["global_union_count"] == 4
    assert row["global_coverage_union"] == 0.5
    mask = row["descriptive_masks"]["8A"]
    assert mask["global_exclusivity"]["mut_only_keys_in_radius"] == [["A", 13]]
    assert mask["global_exclusivity"]["wt_only_keys_in_radius"] == [["A", 14]]
    assert mask["radial_exclusivity_aligned"]["mut_geometry_only_keys"] == [["A", 11]]
    assert mask["local_aligned"]["keys"] == [["A", 10], ["A", 11]]
    assert mask["local_aligned"]["mut_aligned_index"] == [0, 1]
    assert mask["local_aligned"]["wt_aligned_index"] == [0, 1]


def test_union_and_alignment_order_do_not_depend_on_hdf5_row_order(tmp_path):
    first = _audit(
        tmp_path,
        mut_res=(13, 10, 11),
        wt_res=(11, 14, 10),
        mut_xyz=[(3, 0, 0), (0, 0, 0), (4, 0, 0)],
        wt_xyz=[(12, 0, 0), (3, 0, 0), (0, 0, 0)],
    )
    second = _audit(
        tmp_path / "second",
        mut_res=(10, 11, 13),
        wt_res=(10, 11, 14),
        mut_xyz=[(0, 0, 0), (4, 0, 0), (3, 0, 0)],
        wt_xyz=[(0, 0, 0), (12, 0, 0), (3, 0, 0)],
    )
    assert first["union_keys"] == second["union_keys"] == [
        ["A", 10], ["A", 11], ["A", 13], ["A", 14]
    ]
    assert first["aligned_keys"] == second["aligned_keys"] == [["A", 10], ["A", 11]]
    for row in (first, second):
        assert len(row["mut_aligned_index"]) == len(row["wt_aligned_index"]) == 2
        assert all(isinstance(index, int) and index >= 0 for index in row["mut_aligned_index"])
        assert all(isinstance(index, int) and index >= 0 for index in row["wt_aligned_index"])


def test_spatial_audit_never_constructs_or_zero_fills_embedding_unions():
    source = inspect.getsource(spatial_module)
    forbidden = ("H_MUT_union", "H_WT_union", "zeros_like", "sentinel embedding")
    assert not any(token in source for token in forbidden)
    assert "no zero-filled embeddings" in source


def test_zero_denominator_is_null(tmp_path):
    row = _audit(tmp_path, mut_xyz=[(np.nan,0,0)]*3, wt_xyz=[(np.nan,0,0)]*3)
    assert row["radii_metrics"]["4A"]["coverage_union_r"] is None
    assert any(i["code"] == "zero_denominator" for i in row["incidents"])


def test_nan_inf_invalid_shape_and_anchor(tmp_path):
    nan = _audit(tmp_path, mut_xyz=[(np.nan,0,0),(1,0,0),(2,0,0)])
    assert nan["flags"]["coordinate_issue"]
    inf = _audit(tmp_path / "inf", wt_xyz=[(np.inf,0,0),(1,0,0),(2,0,0)])
    assert not inf["anchor_analysis"]["wt_coordinate_valid"]
    path = tmp_path / "shape.h5"
    with h5py.File(path, "w") as h:
        mg = _graph(h, "mut", (10,11,12), [(0,0,0),(1,0,0),(2,0,0)])
        del mg["node_features"]["_position"]
        mg["node_features"].create_dataset("_position", data=np.zeros((3,2)))
        wg = _graph(h, "wt", (10,11,12), [(0,0,0),(1,0,0),(2,0,0)])
        row = audit_spatial_pair(mg, wg, variant_id=MUT, mut_graph_id=MUT, wt_graph_id=WT)
    assert any(i["code"] == "invalid_coordinate_shape" for i in row["incidents"])


def test_deterministic_and_preserves_prior_alignment(tmp_path):
    a = _audit(tmp_path)
    b = _audit(tmp_path / "b")
    assert a["radii_metrics"] == b["radii_metrics"]
    assert a["alignment_status"] == a["prior_alignment"]["alignment_status"] == "valid"
    assert a["training_eligibility"] == "pending"


def test_hdf5_read_only_and_five_artifacts(tmp_path):
    mut, wt = tmp_path / "proc.h5", tmp_path / "wt.h5"
    with h5py.File(mut, "w") as h:
        _graph(h, MUT, (10,11), [(0,0,0),(4,0,0)])
    with h5py.File(wt, "w") as h:
        _graph(h, WT, (10,11), [(0,0,0),(4,0,0)])
    before = hashlib.sha256(mut.read_bytes()).hexdigest(), hashlib.sha256(wt.read_bytes()).hexdigest()
    rows = audit_spatial_hdf5_pairs(mut, wt)
    after = hashlib.sha256(mut.read_bytes()).hexdigest(), hashlib.sha256(wt.read_bytes()).hexdigest()
    assert before == after
    paths = write_spatial_artifacts(tmp_path / "out", rows)
    assert len(paths) == 5 and all(path.exists() for path in paths.values())
    assert len(json.loads(paths["json"].read_text())["pairs"]) == 1
    csv_header = paths["csv"].read_text().splitlines()[0]
    for column in (
        "union_keys",
        "exists_MUT",
        "exists_WT",
        "global_support",
        "local_aligned_keys_8A",
        "local_mut_aligned_index_8A",
        "local_wt_aligned_index_8A",
        "training_eligibility",
        "local_scale_status",
        "global_aligned_count",
        "global_mut_only_count",
        "global_wt_only_count",
        "global_union_count",
        "global_coverage_union",
        "K_MUT_keys_8A",
        "K_WT_keys_8A",
        "K_local_union_keys_8A",
        "aligned_radial_states_8A",
    ):
        assert column in csv_header


def test_r101h_real_integration_if_available():
    mut = Path("/home/sartesero/modelo_optimized_gnn/local_data/hdf5/proc_483p.hdf5")
    wt = Path("/home/sartesero/modelo_optimized_gnn/local_data/hdf5/wt_companion.hdf5")
    if not (mut.exists() and wt.exists()):
        return
    rows = audit_spatial_hdf5_pairs(mut, wt)
    r101h = [row for row in rows if ":101:" in row["variant_id"] and "->Histidine:" in row["variant_id"]]
    assert len(r101h) == 1
    assert r101h[0]["coverage_union"] < 0.1
