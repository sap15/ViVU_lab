from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gnn_siamese.data.hdf5_loader import (
    HDF5GraphLoadError,
    build_is_mutation_channel,
    load_hdf5_graph_components,
    normalize_edge_index,
)


def _create_graph(
    path: Path,
    graph_key: str,
    *,
    diff_values: list[float],
    edge_index: list[list[int]],
    inject_nan: bool = False,
    include_mask: bool = False,
    edge_index_orientation: str = "E,2",
) -> None:
    num_nodes = len(diff_values)
    edge_index_data = edge_index
    if edge_index_orientation == "2,E":
        edge_index_data = [[row[0] for row in edge_index], [row[1] for row in edge_index]]
    num_edges = len(edge_index)

    with h5py.File(path, "w") as handle:
        graph = handle.create_group(graph_key)
        node_group = graph.create_group("node_features")
        edge_group = graph.create_group("edge_features")
        graph_group = graph.create_group("graph_features")

        bsa = np.linspace(0.1, 0.4, num_nodes)
        if inject_nan:
            bsa[1] = np.nan
        node_group.create_dataset("bsa", data=bsa)
        node_group.create_dataset("res_mass", data=np.linspace(10.0, 40.0, num_nodes))
        node_group.create_dataset("diff_mass", data=diff_values)
        node_group.create_dataset("diff_charge", data=np.zeros(num_nodes))
        node_group.create_dataset("diff_pI", data=np.zeros(num_nodes))
        node_group.create_dataset("diff_size", data=np.zeros(num_nodes))
        node_group.create_dataset("_name", data=[b"GLY"] * num_nodes)
        if include_mask:
            node_group.create_dataset("mask_diff_mass", data=np.ones(num_nodes))

        edge_group.create_dataset("_index", data=edge_index_data)
        edge_group.create_dataset("distance", data=np.linspace(3.0, 6.0, num_edges))
        edge_group.create_dataset("covalent", data=np.array([1.0, 0.0, 0.0, 1.0])[:num_edges])

        graph_group.create_dataset("graph_num_nodes", data=float(num_nodes))
        graph_group.create_dataset("graph_num_edges", data=float(num_edges))
        graph_group.create_dataset("custom_structure_energy", data=42.0)


def test_normalize_edge_index_accepts_e2_and_2e() -> None:
    as_e2 = np.array([[0, 1], [1, 2], [2, 0]])
    as_2e = np.array([[0, 1, 2], [1, 2, 0]])

    normalized_e2 = normalize_edge_index(as_e2, num_nodes=3)
    normalized_2e = normalize_edge_index(as_2e, num_nodes=3)

    assert normalized_e2.shape == (2, 3)
    assert normalized_2e.shape == (2, 3)
    np.testing.assert_array_equal(normalized_e2, normalized_2e)


def test_load_hdf5_graph_components_builds_float32_arrays(tmp_path: Path) -> None:
    path = tmp_path / "proc.hdf5"
    graph_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_graph(path, graph_key, diff_values=[0.0, 1.0, 0.0], edge_index=[[0, 1], [1, 2]])

    components = load_hdf5_graph_components(
        path,
        graph_key,
        node_feature_names=["bsa", "res_mass"],
        edge_feature_names=["distance", "covalent"],
    )

    assert components.x.dtype == np.float32
    assert components.edge_attr.dtype == np.float32
    assert components.edge_index.shape == (2, 2)
    assert components.x.shape == (3, 3)
    assert components.edge_attr.shape == (2, 2)
    assert components.node_feature_names == ("bsa", "res_mass", "is_mutation")


def test_loader_rejects_nan_or_inf(tmp_path: Path) -> None:
    path = tmp_path / "proc_nan.hdf5"
    graph_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_graph(
        path,
        graph_key,
        diff_values=[0.0, 1.0, 0.0],
        edge_index=[[0, 1], [1, 2]],
        inject_nan=True,
    )

    with pytest.raises(HDF5GraphLoadError, match="contains NaN"):
        load_hdf5_graph_components(
            path,
            graph_key,
            node_feature_names=["bsa"],
            edge_feature_names=["distance"],
        )


def test_loader_preserves_metadata_and_excludes_global_energy(tmp_path: Path) -> None:
    path = tmp_path / "proc_energy.hdf5"
    graph_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_graph(path, graph_key, diff_values=[0.0, 1.0, 0.0], edge_index=[[0, 1], [1, 2]])

    components = load_hdf5_graph_components(
        path,
        graph_key,
        node_feature_names=["bsa"],
        edge_feature_names=["distance"],
    )

    assert components.metadata["variant_id"] == graph_key
    assert components.metadata["position"] == 100
    assert components.metadata["wt_aa"] == "G"
    assert components.metadata["mut_aa"] == "D"
    assert components.metadata["source_h5"] == str(path)
    assert components.metadata["graph_key"] == graph_key
    assert "custom_structure_energy" not in components.node_feature_names
    assert "custom_structure_energy" not in components.edge_feature_names
    assert components.x.shape[1] == 2
    assert components.edge_attr.shape[1] == 1


def test_loader_loads_availability_masks_as_auxiliary(tmp_path: Path) -> None:
    path = tmp_path / "proc_mask.hdf5"
    graph_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_graph(
        path,
        graph_key,
        diff_values=[0.0, 1.0, 0.0],
        edge_index=[[0, 1], [1, 2]],
        include_mask=True,
    )

    components = load_hdf5_graph_components(
        path,
        graph_key,
        node_feature_names=["bsa", "diff_mass"],
        edge_feature_names=["distance"],
        node_availability_masks={"diff_mass": "mask_diff_mass"},
    )

    assert "diff_mass" in components.node_availability_masks
    np.testing.assert_array_equal(components.node_availability_masks["diff_mass"], np.ones(3))
    assert "mask_diff_mass" not in components.node_feature_names
    assert components.x.shape[1] == 3


def test_loader_builds_is_mutation_for_missense_and_zero_for_wt(tmp_path: Path) -> None:
    mut_path = tmp_path / "proc_mut.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    mut_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    wt_key = "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    _create_graph(mut_path, mut_key, diff_values=[0.0, 1.0, 0.0], edge_index=[[0, 1], [1, 2]])
    _create_graph(wt_path, wt_key, diff_values=[0.0, 0.0, 0.0], edge_index=[[0, 1], [1, 2]])

    mut_components = load_hdf5_graph_components(
        mut_path,
        mut_key,
        node_feature_names=["bsa"],
        edge_feature_names=["distance"],
    )
    wt_components = load_hdf5_graph_components(
        wt_path,
        wt_key,
        node_feature_names=["bsa"],
        edge_feature_names=["distance"],
    )

    assert mut_components.is_mutation.sum() == 1.0
    assert mut_components.mutation_node_index == 1
    assert wt_components.is_mutation.sum() == 0.0
    assert wt_components.mutation_node_index is None


def test_build_is_mutation_fails_on_ambiguous_missense() -> None:
    node_features = {
        "diff_mass": np.array([1.0, 0.0, 1.0]),
        "diff_charge": np.zeros(3),
        "diff_pI": np.zeros(3),
        "diff_size": np.zeros(3),
    }

    with pytest.raises(HDF5GraphLoadError, match="expects exactly one mutated node"):
        build_is_mutation_channel(
            node_features,
            graph_key="residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
        )
