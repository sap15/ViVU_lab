from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gnn_siamese.data.dataset import MutWtPairDataset
from gnn_siamese.data.pairing import MissingWTCompanionError


def _build_config() -> dict:
    return {
        "data": {
            "mutation_node": {
                "source": "diff_features",
                "probes": ["diff_mass", "diff_charge", "diff_pI", "diff_size"],
                "epsilon": 1.0e-12,
                "require_exactly_one_for_missense": True,
                "wt_expected_count": 0,
                "create_is_mutation_channel": True,
            }
        },
        "features": {
            "node_groups": ["structure", "biochemistry", "diff_bioq"],
            "edge_groups": ["distance", "contact"],
            "graph_groups": [],
            "excluded_from_encoder_base": ["diff_polarity"],
            "node_metadata": ["_chain_id", "_name", "_position"],
            "edge_metadata": ["_index", "_name"],
            "confounders": ["custom_structure_energy"],
            "structure": {"enabled": True, "names": ["bsa"]},
            "biochemistry": {"enabled": True, "names": ["res_mass"]},
            "diff_bioq": {
                "enabled": True,
                "names": ["diff_mass", "diff_charge", "diff_pI", "diff_size"],
                "require_masks": True,
                "mask_prefix": "mask_",
            },
            "distance": {"enabled": True, "names": ["distance"]},
            "contact": {"enabled": True, "names": ["covalent"]},
        },
    }


def _build_schema() -> dict:
    return {
        "graph_layout": {
            "node_features": {
                "feature_datasets": {
                    "_chain_id": {},
                    "_name": {},
                    "_position": {},
                    "bsa": {},
                    "res_mass": {},
                    "diff_mass": {},
                    "diff_charge": {},
                    "diff_pI": {},
                    "diff_size": {},
                    "mask_diff_mass": {},
                }
            },
            "edge_features": {
                "feature_datasets": {
                    "_index": {},
                    "distance": {},
                    "covalent": {},
                }
            },
            "graph_features": {
                "feature_datasets": {
                    "custom_structure_energy": {},
                }
            },
        }
    }


def _create_graph(
    path: Path,
    graph_key: str,
    *,
    diff_mass: list[float],
    include_mask: bool = True,
    custom_structure_energy: float = 7.5,
) -> None:
    num_nodes = len(diff_mass)
    position = int(graph_key.split(":")[2])
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    with h5py.File(path, "a") as handle:
        graph = handle.create_group(graph_key)
        node_group = graph.create_group("node_features")
        edge_group = graph.create_group("edge_features")
        graph_group = graph.create_group("graph_features")

        node_group.create_dataset("_chain_id", data=[b"A"] * num_nodes)
        node_group.create_dataset(
            "res_id", data=np.arange(position, position + num_nodes, dtype=np.int64)
        )
        node_group.create_dataset("_name", data=[b"GLY"] * num_nodes)
        node_group.create_dataset(
            "_position",
            data=np.stack(
                [np.arange(num_nodes, dtype=np.float32), np.zeros(num_nodes), np.zeros(num_nodes)],
                axis=1,
            ),
        )
        node_group.create_dataset("bsa", data=np.linspace(0.1, 0.3, num_nodes))
        node_group.create_dataset("res_mass", data=np.linspace(10.0, 30.0, num_nodes))
        node_group.create_dataset("diff_mass", data=np.asarray(diff_mass, dtype=np.float32))
        node_group.create_dataset("diff_charge", data=np.zeros(num_nodes, dtype=np.float32))
        node_group.create_dataset("diff_pI", data=np.zeros(num_nodes, dtype=np.float32))
        node_group.create_dataset("diff_size", data=np.zeros(num_nodes, dtype=np.float32))
        if include_mask:
            node_group.create_dataset("mask_diff_mass", data=np.ones(num_nodes, dtype=np.float32))

        edge_group.create_dataset("_index", data=edge_index)
        edge_group.create_dataset("distance", data=np.array([3.0, 4.0], dtype=np.float32))
        edge_group.create_dataset("covalent", data=np.array([1.0, 0.0], dtype=np.float32))

        graph_group.create_dataset("custom_structure_energy", data=custom_structure_energy)


def test_mut_wt_pair_dataset_loads_single_pair(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    mutant_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    wt_key = "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    _create_graph(mutant_path, mutant_key, diff_mass=[0.0, 1.0, 0.0])
    _create_graph(wt_path, wt_key, diff_mass=[0.0, 0.0, 0.0])

    dataset = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item.graph_mut is not None
    assert item.graph_wt is not None
    assert item.node_pair_alignment is not None
    assert item.mut_aligned_index == (0, 1, 2)
    assert item.wt_aligned_index == (0, 1, 2)
    assert item.exists_MUT == (True, True, True)
    assert item.exists_WT == (True, True, True)


def test_mut_wt_pair_dataset_preserves_pair_metadata(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    mutant_key = "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W"
    wt_key = "residue-srv:A:563:Cysteine->Cysteine:PKP2_WT"
    _create_graph(mutant_path, mutant_key, diff_mass=[1.0, 0.0, 0.0])
    _create_graph(wt_path, wt_key, diff_mass=[0.0, 0.0, 0.0])

    item = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
    )[0]

    assert item.metadata["position"] == 563
    assert item.metadata["wt_aa"] == "C"
    assert item.metadata["mut_aa"] == "W"
    assert item.metadata["mutant_key"] == mutant_key
    assert item.metadata["wt_key"] == wt_key


def test_mut_wt_pair_dataset_excludes_global_energy_and_masks_from_x(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    mutant_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    wt_key = "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    _create_graph(mutant_path, mutant_key, diff_mass=[0.0, 1.0, 0.0], include_mask=True)
    _create_graph(wt_path, wt_key, diff_mass=[0.0, 0.0, 0.0], include_mask=True)

    item = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
    )[0]

    assert "custom_structure_energy" not in item.graph_mut.node_feature_names
    assert "mask_diff_mass" not in item.graph_mut.node_feature_names
    assert "diff_mass" in item.graph_mut.node_availability_masks
    np.testing.assert_array_equal(item.graph_mut.node_availability_masks["diff_mass"], np.ones(3))


def test_mut_wt_pair_dataset_mutation_channels(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    mutant_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    wt_key = "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    _create_graph(mutant_path, mutant_key, diff_mass=[0.0, 1.0, 0.0])
    _create_graph(wt_path, wt_key, diff_mass=[0.0, 0.0, 0.0])

    item = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
    )[0]

    assert item.graph_mut.is_mutation.sum() == 1.0
    assert item.graph_wt.is_mutation.sum() == 0.0


def test_mut_wt_pair_dataset_missing_wt_fails(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    mutant_key = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    _create_graph(mutant_path, mutant_key, diff_mass=[0.0, 1.0, 0.0])
    _create_graph(wt_path, "residue-srv:A:101:Glycine->Glycine:PKP2_WT", diff_mass=[0.0, 0.0, 0.0])

    with pytest.raises(MissingWTCompanionError, match="No WT companion found for mutant"):
        MutWtPairDataset(
            mutant_h5_path=mutant_path,
            wt_h5_path=wt_path,
            config=_build_config(),
            schema=_build_schema(),
        )


def test_mut_wt_pair_dataset_deterministic_order(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    mutant_key_b = "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W"
    mutant_key_a = "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    wt_key_a = "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    wt_key_b = "residue-srv:A:563:Cysteine->Cysteine:PKP2_WT"
    _create_graph(mutant_path, mutant_key_b, diff_mass=[1.0, 0.0, 0.0])
    _create_graph(mutant_path, mutant_key_a, diff_mass=[0.0, 1.0, 0.0])
    _create_graph(wt_path, wt_key_b, diff_mass=[0.0, 0.0, 0.0])
    _create_graph(wt_path, wt_key_a, diff_mass=[0.0, 0.0, 0.0])

    dataset = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
        mutant_graph_keys=[mutant_key_b, mutant_key_a],
        wt_graph_keys=[wt_key_b, wt_key_a],
    )

    assert [dataset[index].variant_id for index in range(len(dataset))] == [
        mutant_key_a,
        mutant_key_b,
    ]
