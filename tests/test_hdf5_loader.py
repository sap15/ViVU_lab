from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gnn_siamese.data.hdf5_loader import (
    HDF5GraphLoadError,
    HDF5GraphComponents,
    NodeFeatureSlice,
    build_is_mutation_channel,
    load_hdf5_graph_components,
    normalize_edge_index,
    validate_graph_components,
    validate_node_feature_slices,
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
        node_group.create_dataset(
            "hse", data=np.arange(num_nodes * 3, dtype=np.float32).reshape(num_nodes, 3)
        )
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
    assert components.node_feature_slices == (
        NodeFeatureSlice("bsa", 0, 1),
        NodeFeatureSlice("res_mass", 1, 2),
        NodeFeatureSlice("is_mutation", 2, 3),
    )


def test_loader_records_hse_runtime_width_as_one_group(tmp_path: Path) -> None:
    path = tmp_path / "proc_hse.hdf5"
    graph_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_graph(path, graph_key, diff_values=[0.0, 1.0, 0.0], edge_index=[[0, 1], [1, 2]])
    components = load_hdf5_graph_components(
        path,
        graph_key,
        node_feature_names=["bsa", "hse", "res_mass"],
        edge_feature_names=["distance"],
    )
    assert components.x.shape == (3, 6)
    assert components.node_feature_slices == (
        NodeFeatureSlice("bsa", 0, 1),
        NodeFeatureSlice("hse", 1, 4),
        NodeFeatureSlice("res_mass", 4, 5),
        NodeFeatureSlice("is_mutation", 5, 6),
    )


@pytest.mark.parametrize(
    ("slices", "width", "message"),
    [
        ((NodeFeatureSlice("", 0, 1),), 1, "non-empty"),
        (
            (NodeFeatureSlice("a", 0, 1), NodeFeatureSlice("a", 1, 2)),
            2,
            "Duplicate",
        ),
        ((NodeFeatureSlice("a", -1, 1),), 1, "expected start 0"),
        ((NodeFeatureSlice("a", 0, 0),), 1, "outside"),
        ((NodeFeatureSlice("a", 0, -1),), 1, "outside"),
        ((NodeFeatureSlice("a", 0, 2),), 1, "outside width"),
        (
            (NodeFeatureSlice("a", 0, 2), NodeFeatureSlice("b", 1, 3)),
            3,
            "overlaps",
        ),
        (
            (NodeFeatureSlice("a", 0, 1), NodeFeatureSlice("b", 2, 3)),
            3,
            "gap",
        ),
        ((NodeFeatureSlice("a", 0, 1),), 2, "incomplete"),
        ((NodeFeatureSlice("a", 1, 2),), 2, "expected start 0"),
        (
            (NodeFeatureSlice("a", 0, 1), NodeFeatureSlice("a", 1, 3)),
            3,
            "Duplicate",
        ),
        ((), 2, "non-empty"),
    ],
    ids=(
        "empty-name",
        "duplicate-name",
        "negative-start",
        "stop-equals-start",
        "stop-before-start",
        "stop-outside-x",
        "overlap",
        "gap",
        "incomplete-tail",
        "first-not-zero",
        "duplicate-different-bounds",
        "empty-metadata-with-columns",
    ),
)
def test_validate_node_feature_slices_rejects_each_invalid_layout(
    slices: tuple[NodeFeatureSlice, ...],
    width: int,
    message: str,
) -> None:
    with pytest.raises(HDF5GraphLoadError, match=message):
        validate_node_feature_slices(slices, width=width)


@pytest.mark.parametrize(
    ("slices", "width"),
    [
        ((NodeFeatureSlice("scalar", 0, 1),), 1),
        (
            (
                NodeFeatureSlice("a", 0, 1),
                NodeFeatureSlice("b", 1, 2),
                NodeFeatureSlice("c", 2, 3),
            ),
            3,
        ),
        ((NodeFeatureSlice("vector", 0, 3),), 3),
        (
            (
                NodeFeatureSlice("left", 0, 1),
                NodeFeatureSlice("vector", 1, 4),
                NodeFeatureSlice("right", 4, 5),
            ),
            5,
        ),
    ],
    ids=("one-scalar", "several-scalars", "one-vector", "scalar-vector-scalar"),
)
def test_validate_node_feature_slices_accepts_exact_complete_layouts(
    slices: tuple[NodeFeatureSlice, ...],
    width: int,
) -> None:
    assert validate_node_feature_slices(slices, width=width) == slices


def _minimal_components(
    *,
    names: tuple[str, ...],
    slices: tuple[NodeFeatureSlice, ...],
) -> HDF5GraphComponents:
    return HDF5GraphComponents(
        x=np.zeros((2, 2), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_attr=np.empty((0, 1), dtype=np.float32),
        metadata={},
        node_feature_names=names,
        node_feature_slices=slices,
        edge_feature_names=("distance",),
        node_availability_masks={},
        mutation_node_index=None,
        is_mutation=np.zeros(2, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (("b", "a"), "exactly match"),
        (("a",), "exactly match"),
        (("a", "b", "extra"), "exactly match"),
    ],
    ids=("incompatible-order", "too-few-names", "too-many-names"),
)
def test_graph_component_validation_rejects_names_incompatible_with_slices(
    names: tuple[str, ...],
    message: str,
) -> None:
    components = _minimal_components(
        names=names,
        slices=(NodeFeatureSlice("a", 0, 1), NodeFeatureSlice("b", 1, 2)),
    )
    with pytest.raises(HDF5GraphLoadError, match=message):
        validate_graph_components(components)


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
