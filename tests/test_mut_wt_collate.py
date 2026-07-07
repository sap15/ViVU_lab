from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from gnn_siamese.data.collate import MutWtPairCollateError, collate_mut_wt_pairs
from gnn_siamese.data.dataset import MutWtPairSample
from gnn_siamese.data.hdf5_loader import HDF5GraphComponents
from gnn_siamese.data.pairing import PairingKey

torch_geometric = pytest.importorskip("torch_geometric")


def _make_graph(
    *,
    variant_id: str,
    x: list[list[float]],
    edge_index: list[list[int]],
    edge_attr: list[list[float]],
    is_mutation: list[float],
    diff_mass_mask: list[float],
) -> HDF5GraphComponents:
    return HDF5GraphComponents(
        x=np.asarray(x, dtype=np.float32),
        edge_index=np.asarray(edge_index, dtype=np.int64),
        edge_attr=np.asarray(edge_attr, dtype=np.float32),
        metadata={"variant_id": variant_id},
        node_feature_names=("bsa", "diff_mass", "is_mutation"),
        edge_feature_names=("distance", "covalent"),
        node_availability_masks={"diff_mass": np.asarray(diff_mass_mask, dtype=np.float32)},
        mutation_node_index=None,
        is_mutation=np.asarray(is_mutation, dtype=np.float32),
    )


def _make_sample(
    *,
    suffix: str,
    pair_key: PairingKey,
    mut_x: list[list[float]],
    wt_x: list[list[float]],
    mut_edge_index: list[list[int]],
    wt_edge_index: list[list[int]],
    mut_is_mutation: list[float],
    wt_is_mutation: list[float],
    mut_mask: list[float],
    wt_mask: list[float],
) -> MutWtPairSample:
    variant_id = f"residue-srv:A:{suffix}:Glycine->Aspartate:pos_{suffix}_G_D"
    wt_key = f"residue-srv:A:{suffix}:Glycine->Glycine:PKP2_WT"
    return MutWtPairSample(
        graph_mut=_make_graph(
            variant_id=variant_id,
            x=mut_x,
            edge_index=mut_edge_index,
            edge_attr=[[3.0, 1.0] for _ in range(len(mut_edge_index[0]))],
            is_mutation=mut_is_mutation,
            diff_mass_mask=mut_mask,
        ),
        graph_wt=_make_graph(
            variant_id=wt_key,
            x=wt_x,
            edge_index=wt_edge_index,
            edge_attr=[[4.0, 0.0] for _ in range(len(wt_edge_index[0]))],
            is_mutation=wt_is_mutation,
            diff_mass_mask=wt_mask,
        ),
        metadata={"variant_id": variant_id, "position": int(suffix)},
        pair_key=pair_key,
        variant_id=variant_id,
        mutant_key=variant_id,
        wt_key=wt_key,
    )


def test_collate_mut_wt_pairs_batches_two_samples() -> None:
    sample_a = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 0.0, 0.0], [0.2, 1.0, 1.0]],
        wt_x=[[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        mut_edge_index=[[0], [1]],
        wt_edge_index=[[0], [1]],
        mut_is_mutation=[0.0, 1.0],
        wt_is_mutation=[0.0, 0.0],
        mut_mask=[1.0, 1.0],
        wt_mask=[1.0, 1.0],
    )
    sample_b = _make_sample(
        suffix="101",
        pair_key=PairingKey(chain_id="A", position=101, wt_aa="G"),
        mut_x=[[0.5, 0.0, 1.0]],
        wt_x=[[0.6, 0.0, 0.0], [0.7, 0.0, 0.0], [0.8, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[0, 1], [1, 2]],
        mut_is_mutation=[1.0],
        wt_is_mutation=[0.0, 0.0, 0.0],
        mut_mask=[0.0],
        wt_mask=[1.0, 1.0, 0.0],
    )

    batch = collate_mut_wt_pairs([sample_a, sample_b])

    assert batch.batch_size == 2
    assert isinstance(batch.graph_mut, torch_geometric.data.Batch)
    assert isinstance(batch.graph_wt, torch_geometric.data.Batch)
    torch.testing.assert_close(
        batch.graph_mut.x,
        torch.tensor([[0.1, 0.0, 0.0], [0.2, 1.0, 1.0], [0.5, 0.0, 1.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch.graph_wt.x,
        torch.tensor(
            [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.7, 0.0, 0.0], [0.8, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        batch.graph_mut.edge_attr,
        torch.tensor([[3.0, 1.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch.graph_wt.edge_attr,
        torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch.graph_mut.edge_index,
        torch.tensor([[0], [1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        batch.graph_wt.edge_index,
        torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=torch.int64),
    )
    assert batch.graph_mut.x.shape == (3, 3)
    assert batch.graph_wt.x.shape == (5, 3)
    assert batch.graph_mut.edge_index.shape == (2, 1)
    assert batch.graph_wt.edge_index.shape == (2, 3)
    assert batch.graph_mut.edge_attr.shape == (1, 2)
    assert batch.graph_wt.edge_attr.shape == (3, 2)
    assert batch.graph_mut.batch is not None
    assert batch.graph_mut.ptr is not None
    assert batch.graph_wt.batch is not None
    assert batch.graph_wt.ptr is not None
    assert batch.graph_mut.x.dtype == torch.float32
    assert batch.graph_mut.edge_index.dtype == torch.int64
    assert batch.graph_mut.edge_attr.dtype == torch.float32
    assert batch.graph_wt.x.dtype == torch.float32
    assert batch.graph_wt.edge_index.dtype == torch.int64
    assert batch.graph_wt.edge_attr.dtype == torch.float32


def test_collate_offsets_edge_index_correctly() -> None:
    sample_a = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 0.0, 1.0], [0.2, 0.0, 0.0]],
        wt_x=[[0.3, 0.0, 0.0]],
        mut_edge_index=[[0, 1], [1, 0]],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0, 0.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0, 1.0],
        wt_mask=[1.0],
    )
    sample_b = _make_sample(
        suffix="101",
        pair_key=PairingKey(chain_id="A", position=101, wt_aa="G"),
        mut_x=[[0.4, 0.0, 0.0], [0.5, 0.0, 1.0], [0.6, 0.0, 0.0]],
        wt_x=[[0.7, 0.0, 0.0]],
        mut_edge_index=[[0, 2], [2, 1]],
        wt_edge_index=[[], []],
        mut_is_mutation=[0.0, 1.0, 0.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0, 1.0, 1.0],
        wt_mask=[1.0],
    )

    batch = collate_mut_wt_pairs([sample_a, sample_b])

    torch.testing.assert_close(
        batch.graph_mut.edge_index,
        torch.tensor([[0, 1, 2, 4], [1, 0, 4, 3]], dtype=torch.int64),
    )
    torch.testing.assert_close(batch.graph_mut.ptr, torch.tensor([0, 2, 5], dtype=torch.int64))
    torch.testing.assert_close(batch.graph_mut.batch, torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64))


def test_collate_preserves_metadata_order() -> None:
    sample_a = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 0.0, 1.0]],
        wt_x=[[0.2, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0],
        wt_mask=[1.0],
    )
    sample_b = _make_sample(
        suffix="563",
        pair_key=PairingKey(chain_id="A", position=563, wt_aa="C"),
        mut_x=[[0.3, 0.0, 1.0]],
        wt_x=[[0.4, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0],
        wt_is_mutation=[0.0],
        mut_mask=[0.0],
        wt_mask=[1.0],
    )
    sample_b = MutWtPairSample(
        graph_mut=sample_b.graph_mut,
        graph_wt=sample_b.graph_wt,
        metadata={"variant_id": sample_b.variant_id, "position": 563, "tag": "second"},
        pair_key=sample_b.pair_key,
        variant_id=sample_b.variant_id,
        mutant_key=sample_b.mutant_key,
        wt_key=sample_b.wt_key,
    )

    batch = collate_mut_wt_pairs([sample_a, sample_b])

    assert batch.variant_ids == [sample_a.variant_id, sample_b.variant_id]
    assert batch.mutant_keys == [sample_a.mutant_key, sample_b.mutant_key]
    assert batch.wt_keys == [sample_a.wt_key, sample_b.wt_key]
    assert batch.pair_keys == [sample_a.pair_key, sample_b.pair_key]
    assert batch.metadata == [sample_a.metadata, sample_b.metadata]


def test_collate_preserves_is_mutation_channels() -> None:
    sample_a = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 0.0, 1.0], [0.2, 0.0, 0.0]],
        wt_x=[[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0, 0.0],
        wt_is_mutation=[0.0, 0.0],
        mut_mask=[1.0, 1.0],
        wt_mask=[1.0, 1.0],
    )
    sample_b = _make_sample(
        suffix="101",
        pair_key=PairingKey(chain_id="A", position=101, wt_aa="G"),
        mut_x=[[0.5, 0.0, 0.0], [0.6, 0.0, 1.0]],
        wt_x=[[0.7, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[0.0, 1.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0, 1.0],
        wt_mask=[1.0],
    )

    batch = collate_mut_wt_pairs([sample_a, sample_b])

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


def test_collate_preserves_availability_masks() -> None:
    sample_a = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 5.0, 1.0]],
        wt_x=[[0.2, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0],
        wt_is_mutation=[0.0],
        mut_mask=[0.0],
        wt_mask=[1.0],
    )
    sample_b = _make_sample(
        suffix="101",
        pair_key=PairingKey(chain_id="A", position=101, wt_aa="G"),
        mut_x=[[0.3, 6.0, 1.0], [0.4, 0.0, 0.0]],
        wt_x=[[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0, 0.0],
        wt_is_mutation=[0.0, 0.0],
        mut_mask=[1.0, 1.0],
        wt_mask=[1.0, 0.0],
    )

    batch = collate_mut_wt_pairs([sample_a, sample_b])

    assert "mask_diff_mass" not in batch.graph_mut.node_feature_names
    torch.testing.assert_close(
        batch.graph_mut.node_availability_masks["diff_mass"],
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
    )
    assert batch.graph_mut.x.shape[1] == 3


def test_collate_works_with_torch_dataloader() -> None:
    sample_a = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 0.0, 1.0]],
        wt_x=[[0.2, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0],
        wt_mask=[1.0],
    )
    sample_b = _make_sample(
        suffix="101",
        pair_key=PairingKey(chain_id="A", position=101, wt_aa="G"),
        mut_x=[[0.3, 0.0, 1.0], [0.4, 0.0, 0.0]],
        wt_x=[[0.5, 0.0, 0.0]],
        mut_edge_index=[[0], [1]],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0, 0.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0, 1.0],
        wt_mask=[1.0],
    )

    loader = DataLoader([sample_a, sample_b], batch_size=2, collate_fn=collate_mut_wt_pairs, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch.graph_mut, torch_geometric.data.Batch)
    assert isinstance(batch.graph_wt, torch_geometric.data.Batch)
    assert batch.variant_ids == [sample_a.variant_id, sample_b.variant_id]
    assert batch.metadata == [sample_a.metadata, sample_b.metadata]


def test_collate_rejects_empty_or_incompatible_samples() -> None:
    with pytest.raises(MutWtPairCollateError, match="non-empty sequence"):
        collate_mut_wt_pairs([])

    valid = _make_sample(
        suffix="100",
        pair_key=PairingKey(chain_id="A", position=100, wt_aa="G"),
        mut_x=[[0.1, 0.0, 1.0]],
        wt_x=[[0.2, 0.0, 0.0]],
        mut_edge_index=[[], []],
        wt_edge_index=[[], []],
        mut_is_mutation=[1.0],
        wt_is_mutation=[0.0],
        mut_mask=[1.0],
        wt_mask=[1.0],
    )
    incompatible_graph = HDF5GraphComponents(
        x=valid.graph_mut.x,
        edge_index=valid.graph_mut.edge_index,
        edge_attr=valid.graph_mut.edge_attr,
        metadata=dict(valid.graph_mut.metadata),
        node_feature_names=("other_feature", "is_mutation"),
        edge_feature_names=valid.graph_mut.edge_feature_names,
        node_availability_masks=valid.graph_mut.node_availability_masks,
        mutation_node_index=valid.graph_mut.mutation_node_index,
        is_mutation=valid.graph_mut.is_mutation,
    )
    incompatible = MutWtPairSample(
        graph_mut=incompatible_graph,
        graph_wt=valid.graph_wt,
        metadata=dict(valid.metadata),
        pair_key=valid.pair_key,
        variant_id=valid.variant_id,
        mutant_key=valid.mutant_key,
        wt_key=valid.wt_key,
    )

    with pytest.raises(MutWtPairCollateError, match="node_feature_names"):
        collate_mut_wt_pairs([valid, incompatible])
