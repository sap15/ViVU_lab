from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from gnn_siamese.data.collate import collate_mut_wt_pairs
from gnn_siamese.data.dataset import MutWtPairDataset
from gnn_siamese.data.pairing import MissingWTCompanionError

torch_geometric = pytest.importorskip("torch_geometric")


def _build_config() -> dict:
    return {
        "features": {
            "node_groups": ["structure", "biochemistry", "diff_bioq"],
            "edge_groups": ["distance", "contact"],
            "graph_groups": [],
            "excluded_from_encoder_base": ["diff_polarity"],
            "confounders": ["custom_structure_energy"],
            "node_metadata": ["_chain_id", "_name", "_position"],
            "edge_metadata": ["_index"],
            "structure": {"enabled": True, "names": ["bsa"]},
            "biochemistry": {"enabled": True, "names": ["res_mass"]},
            "diff_bioq": {
                "enabled": True,
                "names": ["diff_mass", "diff_charge", "diff_pI", "diff_size", "diff_polarity"],
                "require_masks": True,
                "mask_prefix": "mask_",
            },
            "distance": {"enabled": True, "names": ["distance"]},
            "contact": {"enabled": True, "names": ["covalent"]},
        },
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
                    "diff_polarity": {},
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
                    "graph_num_nodes": {},
                    "graph_num_edges": {},
                }
            },
        }
    }


def _create_graph(
    path: Path,
    graph_key: str,
    *,
    residue_name: bytes,
    diff_mass: list[float],
    diff_charge: list[float],
    diff_pI: list[float],
    diff_size: list[float],
    mask_diff_mass: list[float],
    custom_structure_energy: float,
) -> None:
    num_nodes = len(diff_mass)
    edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    num_edges = edge_index.shape[0]

    with h5py.File(path, "a") as handle:
        graph = handle.create_group(graph_key)
        node_group = graph.create_group("node_features")
        edge_group = graph.create_group("edge_features")
        graph_group = graph.create_group("graph_features")

        node_group.create_dataset("_chain_id", data=[b"A"] * num_nodes)
        node_group.create_dataset("_name", data=[residue_name] * num_nodes)
        node_group.create_dataset(
            "_position",
            data=np.stack(
                [
                    np.arange(num_nodes, dtype=np.float32),
                    np.zeros(num_nodes, dtype=np.float32),
                    np.zeros(num_nodes, dtype=np.float32),
                ],
                axis=1,
            ),
        )
        node_group.create_dataset("bsa", data=np.linspace(0.1, 0.1 * num_nodes, num_nodes))
        node_group.create_dataset("res_mass", data=np.linspace(10.0, 10.0 * num_nodes, num_nodes))
        node_group.create_dataset("diff_mass", data=np.asarray(diff_mass, dtype=np.float32))
        node_group.create_dataset("diff_charge", data=np.asarray(diff_charge, dtype=np.float32))
        node_group.create_dataset("diff_pI", data=np.asarray(diff_pI, dtype=np.float32))
        node_group.create_dataset("diff_size", data=np.asarray(diff_size, dtype=np.float32))
        node_group.create_dataset("diff_polarity", data=[b"same"] * num_nodes)
        node_group.create_dataset("mask_diff_mass", data=np.asarray(mask_diff_mass, dtype=np.float32))

        edge_group.create_dataset("_index", data=edge_index)
        edge_group.create_dataset("distance", data=np.asarray([3.0, 4.0], dtype=np.float32))
        edge_group.create_dataset("covalent", data=np.asarray([1.0, 0.0], dtype=np.float32))

        graph_group.create_dataset("custom_structure_energy", data=custom_structure_energy)
        graph_group.create_dataset("graph_num_nodes", data=float(num_nodes))
        graph_group.create_dataset("graph_num_edges", data=float(num_edges))


def _build_two_pair_hdf5(mutant_path: Path, wt_path: Path) -> None:
    _create_graph(
        mutant_path,
        "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W",
        residue_name=b"CYS",
        diff_mass=[1.0, 0.0, 0.0],
        diff_charge=[1.0, 0.0, 0.0],
        diff_pI=[1.0, 0.0, 0.0],
        diff_size=[1.0, 0.0, 0.0],
        mask_diff_mass=[1.0, 0.0, 1.0],
        custom_structure_energy=7.5,
    )
    _create_graph(
        mutant_path,
        "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
        residue_name=b"GLY",
        diff_mass=[0.0, 1.0, 0.0],
        diff_charge=[0.0, 1.0, 0.0],
        diff_pI=[0.0, 1.0, 0.0],
        diff_size=[0.0, 1.0, 0.0],
        mask_diff_mass=[1.0, 1.0, 0.0],
        custom_structure_energy=8.5,
    )
    _create_graph(
        wt_path,
        "residue-srv:A:563:Cysteine->Cysteine:PKP2_WT",
        residue_name=b"CYS",
        diff_mass=[0.0, 0.0, 0.0],
        diff_charge=[0.0, 0.0, 0.0],
        diff_pI=[0.0, 0.0, 0.0],
        diff_size=[0.0, 0.0, 0.0],
        mask_diff_mass=[1.0, 1.0, 1.0],
        custom_structure_energy=1.5,
    )
    _create_graph(
        wt_path,
        "residue-srv:A:100:Glycine->Glycine:PKP2_WT",
        residue_name=b"GLY",
        diff_mass=[0.0, 0.0, 0.0],
        diff_charge=[0.0, 0.0, 0.0],
        diff_pI=[0.0, 0.0, 0.0],
        diff_size=[0.0, 0.0, 0.0],
        mask_diff_mass=[1.0, 0.0, 1.0],
        custom_structure_energy=2.5,
    )


def test_integrated_dataset_pipeline_builds_valid_mut_wt_batch(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    _build_two_pair_hdf5(mutant_path, wt_path)

    dataset = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
    )

    assert len(dataset) == 2

    sample0 = dataset[0]
    sample1 = dataset[1]

    assert sample0.graph_mut is not None
    assert sample0.graph_wt is not None
    assert sample1.graph_mut is not None
    assert sample1.graph_wt is not None

    assert sample0.metadata["position"] == 100
    assert sample1.metadata["position"] == 563
    assert sample0.variant_id == "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    assert sample1.variant_id == "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W"

    assert sample0.graph_mut.mutation_node_index is not None
    assert sample1.graph_mut.mutation_node_index is not None
    assert sample0.graph_wt.mutation_node_index is None
    assert sample1.graph_wt.mutation_node_index is None

    batch = collate_mut_wt_pairs([sample0, sample1])

    assert batch.batch_size == 2
    assert isinstance(batch.graph_mut, torch_geometric.data.Batch)
    assert isinstance(batch.graph_wt, torch_geometric.data.Batch)
    assert batch.graph_mut.x.dtype == torch.float32
    assert batch.graph_wt.x.dtype == torch.float32
    assert batch.graph_mut.edge_attr.dtype == torch.float32
    assert batch.graph_wt.edge_attr.dtype == torch.float32
    assert batch.graph_mut.edge_index.shape[0] == 2
    assert batch.graph_wt.edge_index.shape[0] == 2
    assert len(batch.graph_mut.ptr) == 3
    assert len(batch.graph_wt.ptr) == 3
    assert len(batch.graph_mut.batch) == batch.graph_mut.x.shape[0]
    assert len(batch.graph_wt.batch) == batch.graph_wt.x.shape[0]
    assert batch.graph_mut.batch is not None
    assert batch.graph_mut.ptr is not None
    assert batch.graph_wt.batch is not None
    assert batch.graph_wt.ptr is not None

    assert "custom_structure_energy" not in batch.graph_mut.node_feature_names
    assert "custom_structure_energy" not in batch.graph_mut.edge_feature_names
    assert "mask_diff_mass" not in batch.graph_mut.node_feature_names
    assert "diff_polarity" not in batch.graph_mut.node_feature_names
    assert "diff_mass" in batch.graph_mut.node_feature_names
    assert "is_mutation" in batch.graph_mut.node_feature_names
    assert "diff_mass" in batch.graph_mut.node_availability_masks
    torch.testing.assert_close(
        batch.graph_mut.node_availability_masks["diff_mass"],
        torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32),
    )

    mut_counts = [
        int(batch.graph_mut.is_mutation[start:end].sum().item())
        for start, end in zip(batch.graph_mut.ptr[:-1], batch.graph_mut.ptr[1:])
    ]
    wt_counts = [
        int(batch.graph_wt.is_mutation[start:end].sum().item())
        for start, end in zip(batch.graph_wt.ptr[:-1], batch.graph_wt.ptr[1:])
    ]
    assert mut_counts == [1, 1]
    assert wt_counts == [0, 0]

    assert batch.variant_ids == [
        "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
        "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W",
    ]
    assert batch.mutant_keys == batch.variant_ids
    assert batch.wt_keys == [
        "residue-srv:A:100:Glycine->Glycine:PKP2_WT",
        "residue-srv:A:563:Cysteine->Cysteine:PKP2_WT",
    ]

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_mut_wt_pairs, shuffle=False)
    loader_batch = next(iter(loader))

    assert isinstance(loader_batch.graph_mut, torch_geometric.data.Batch)
    assert isinstance(loader_batch.graph_wt, torch_geometric.data.Batch)
    assert loader_batch.variant_ids == batch.variant_ids
    assert loader_batch.metadata == batch.metadata
    assert batch.metadata[0]["position"] == 100
    assert batch.metadata[1]["position"] == 563
    assert batch.metadata[0]["wt_aa"] == "G"
    assert batch.metadata[0]["mut_aa"] == "D"
    assert batch.metadata[1]["wt_aa"] == "C"
    assert batch.metadata[1]["mut_aa"] == "W"


def test_integrated_pipeline_rejects_missing_wt(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    _create_graph(
        mutant_path,
        "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
        residue_name=b"GLY",
        diff_mass=[0.0, 1.0, 0.0],
        diff_charge=[0.0, 1.0, 0.0],
        diff_pI=[0.0, 1.0, 0.0],
        diff_size=[0.0, 1.0, 0.0],
        mask_diff_mass=[1.0, 1.0, 1.0],
        custom_structure_energy=9.0,
    )
    _create_graph(
        wt_path,
        "residue-srv:A:101:Glycine->Glycine:PKP2_WT",
        residue_name=b"GLY",
        diff_mass=[0.0, 0.0, 0.0],
        diff_charge=[0.0, 0.0, 0.0],
        diff_pI=[0.0, 0.0, 0.0],
        diff_size=[0.0, 0.0, 0.0],
        mask_diff_mass=[1.0, 1.0, 1.0],
        custom_structure_energy=1.0,
    )

    with pytest.raises(MissingWTCompanionError, match="No WT companion found for mutant"):
        MutWtPairDataset(
            mutant_h5_path=mutant_path,
            wt_h5_path=wt_path,
            config=_build_config(),
            schema=_build_schema(),
        )
