from __future__ import annotations

import numpy as np
import torch

from gnn_siamese.data.collate import collate_mut_wt_pairs
from gnn_siamese.data.dataset import MutWtPairSample
from gnn_siamese.data.hdf5_loader import HDF5GraphComponents, NodeFeatureSlice
from gnn_siamese.data.node_pair_alignment import align_node_pair
from gnn_siamese.data.pairing import PairingKey


def _graph(name: str, size: int, *, mutant: bool) -> HDF5GraphComponents:
    mutation = np.zeros(size, dtype=np.float32)
    if mutant:
        mutation[0] = 1
    return HDF5GraphComponents(
        x=np.column_stack((np.arange(size, dtype=np.float32), mutation)),
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_attr=np.empty((0, 1), dtype=np.float32),
        metadata={"variant_id": name},
        node_feature_names=("value", "is_mutation"),
        node_feature_slices=(
            NodeFeatureSlice("value", 0, 1),
            NodeFeatureSlice("is_mutation", 1, 2),
        ),
        edge_feature_names=("distance",),
        node_availability_masks={},
        mutation_node_index=0 if mutant else None,
        is_mutation=mutation,
    )


def _sample(
    position: int,
    mut_res: tuple[int, ...],
    wt_res: tuple[int, ...],
    *,
    mut_order: tuple[int, ...] | None = None,
    wt_order: tuple[int, ...] | None = None,
    radius: float = 8.0,
) -> MutWtPairSample:
    mut_order = mut_order or tuple(range(len(mut_res)))
    wt_order = wt_order or tuple(range(len(wt_res)))
    ordered_mut = tuple(mut_res[index] for index in mut_order)
    ordered_wt = tuple(wt_res[index] for index in wt_order)
    mut_xyz = np.asarray([[float(value - position), 0, 0] for value in ordered_mut])
    wt_xyz = np.asarray([[float(value - position), 0, 0] for value in ordered_wt])
    alignment = align_node_pair(
        ["A"] * len(ordered_mut),
        ordered_mut,
        mut_xyz,
        ["A"] * len(ordered_wt),
        ordered_wt,
        wt_xyz,
        anchor_key=("A", position),
        radii=(radius,),
        technical_radius_angstrom=radius,
    )
    variant = f"mut-{position}"
    return MutWtPairSample(
        graph_mut=_graph(variant, len(ordered_mut), mutant=True),
        graph_wt=_graph(f"wt-{position}", len(ordered_wt), mutant=False),
        metadata={"position": position},
        pair_key=PairingKey(chain_id="A", position=position, wt_aa="G"),
        variant_id=variant,
        mutant_key=variant,
        wt_key=f"wt-{position}",
        node_pair_alignment=alignment,
    )


def _assert_reconstructable(batch, samples) -> None:
    for pair_index, sample in enumerate(samples):
        mut_offset = int(batch.graph_mut.ptr[pair_index])
        wt_offset = int(batch.graph_wt.ptr[pair_index])
        start, end = batch.alignment_ptr[pair_index : pair_index + 2]
        assert tuple((batch.mut_aligned_index[start:end] - mut_offset).tolist()) == sample.mut_aligned_index
        assert tuple((batch.wt_aligned_index[start:end] - wt_offset).tolist()) == sample.wt_aligned_index
        local_start, local_end = batch.local_alignment_ptr[pair_index : pair_index + 2]
        assert tuple(
            (batch.local_mut_aligned_index[local_start:local_end] - mut_offset).tolist()
        ) == sample.local_mut_aligned_index
        assert tuple(
            (batch.local_wt_aligned_index[local_start:local_end] - wt_offset).tolist()
        ) == sample.local_wt_aligned_index


def test_single_pair_indices_and_pointers_are_local_contract() -> None:
    sample = _sample(10, (10, 11, 13), (10, 11, 14))
    batch = collate_mut_wt_pairs([sample])
    assert tuple(batch.mut_aligned_index.tolist()) == sample.mut_aligned_index
    assert tuple(batch.wt_aligned_index.tolist()) == sample.wt_aligned_index
    assert batch.alignment_ptr.tolist() == [0, 2]
    assert batch.union_ptr.tolist() == [0, 4]
    assert batch.local_alignment_ptr.tolist() == [0, 2]


def test_heterogeneous_pairs_use_independent_offsets_and_reconstruct_segments() -> None:
    samples = [
        _sample(10, (10, 11, 12), (10, 12)),
        _sample(20, (20, 22), (20, 21, 22, 23)),
        _sample(30, (30,), (30, 31, 32)),
    ]
    batch = collate_mut_wt_pairs(samples)
    assert batch.alignment_ptr.tolist() == [0, 2, 4, 5]
    assert batch.union_ptr.tolist() == [0, 3, 7, 10]
    assert batch.local_alignment_ptr.tolist() == [0, 2, 4, 5]
    assert batch.aligned_pair_batch.tolist() == [0, 0, 1, 1, 2]
    assert batch.union_pair_batch.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    _assert_reconstructable(batch, samples)
    assert batch.mut_aligned_index.tolist() == [0, 2, 3, 4, 5]
    assert batch.wt_aligned_index.tolist() == [0, 1, 2, 4, 6]


def test_row_reordering_exclusive_nodes_and_local_offsets_preserve_alignment() -> None:
    samples = [
        _sample(10, (10, 11, 13), (10, 11, 14), mut_order=(2, 0, 1), wt_order=(1, 2, 0)),
        _sample(20, (20, 21), (20, 22, 21), mut_order=(1, 0), wt_order=(2, 0, 1)),
    ]
    batch = collate_mut_wt_pairs(samples)
    _assert_reconstructable(batch, samples)
    assert batch.exists_MUT.tolist() == [True, True, True, False, True, True, False]
    assert batch.exists_WT.tolist() == [True, True, False, True, True, True, True]
    assert not any(
        not mut_exists and not wt_exists
        for mut_exists, wt_exists in zip(batch.exists_MUT, batch.exists_WT)
    )


def test_empty_alignment_segments_and_device_transfer_remain_valid() -> None:
    samples = [
        _sample(10, (10,), (20,)),
        _sample(30, (30, 31), (30, 31)),
    ]
    batch = collate_mut_wt_pairs(samples)
    assert batch.alignment_ptr.tolist() == [0, 0, 2]
    assert batch.local_alignment_ptr.tolist() == [0, 0, 2]
    moved = batch.to("cpu")
    for name in (
        "mut_aligned_index",
        "wt_aligned_index",
        "aligned_pair_batch",
        "alignment_ptr",
        "union_pair_batch",
        "union_ptr",
        "local_mut_aligned_index",
        "local_wt_aligned_index",
        "local_alignment_ptr",
    ):
        assert getattr(moved, name).dtype == torch.long
        assert getattr(moved, name).device.type == "cpu"
    assert moved.exists_MUT.dtype == torch.bool
    assert moved.exists_WT.dtype == torch.bool
