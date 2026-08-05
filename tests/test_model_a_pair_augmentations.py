from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_siamese.data.collate import MutWtPairBatch
from gnn_siamese.data.hdf5_loader import NodeFeatureSlice
from gnn_siamese.data.model_a_pair_augmentations import (
    ModelAPairAugmentationConfig,
    ModelAPairAugmentationError,
    ModelAPairAugmenter,
    stable_seed,
)
from gnn_siamese.data.pairing import PairingKey


FEATURE_NAMES = ("bsa", "hydrophobicity", "diff_mass", "is_mutation")


def _graph(values: list[list[float]], mutation_index: int | None) -> Data:
    count = len(values)
    edges = [[index, index + 1] for index in range(count - 1)]
    edges += [[target, source] for source, target in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    is_mutation = torch.zeros(count)
    if mutation_index is not None:
        is_mutation[mutation_index] = 1.0
    return Data(
        x=torch.tensor(values, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.arange(edge_index.shape[1], dtype=torch.float32).unsqueeze(1),
        is_mutation=is_mutation,
    )


def make_pair_batch(order: tuple[int, ...] = (0, 1)) -> MutWtPairBatch:
    variants = (
        ("v100", PairingKey("A", 100, "G"), "mut100", "wt100"),
        ("v200", PairingKey("A", 200, "C"), "mut200", "wt200"),
    )
    mut_graphs = (
        _graph([[1, 2, 7, 0], [3, 4, -5, 1], [5, 6, 2, 0]], 1),
        _graph([[11, 12, 9, 1], [13, 14, -3, 0]], 0),
    )
    wt_graphs = (
        _graph([[21, 22, 7, 0], [23, 24, -5, 0], [25, 26, 2, 0]], None),
        _graph([[31, 32, 9, 0], [33, 34, -3, 0], [35, 36, 1, 0]], None),
    )
    selected = [variants[index] for index in order]
    graph_mut = Batch.from_data_list([mut_graphs[index] for index in order])
    graph_wt = Batch.from_data_list([wt_graphs[index] for index in order])
    for graph in (graph_mut, graph_wt):
        graph.node_feature_names = FEATURE_NAMES
        graph.node_feature_slices = (
            NodeFeatureSlice("bsa", 0, 1),
            NodeFeatureSlice("hydrophobicity", 1, 2),
            NodeFeatureSlice("diff_mass", 2, 3),
            NodeFeatureSlice("is_mutation", 3, 4),
        )
        graph.edge_feature_names = ("distance",)
        graph.node_availability_masks = {
            "mask_diff_mass": torch.ones(graph.num_nodes)
        }
        graph.graph_metadata = [{"quality": "preserved"} for _ in order]

    alignment_lengths = [mut_graphs[index].num_nodes for index in order]
    mut_offsets = graph_mut.ptr[:-1]
    wt_offsets = graph_wt.ptr[:-1]
    mut_aligned = torch.cat(
        [torch.arange(length) + mut_offsets[i] for i, length in enumerate(alignment_lengths)]
    )
    wt_aligned = torch.cat(
        [torch.arange(length) + wt_offsets[i] for i, length in enumerate(alignment_lengths)]
    )
    ptr = torch.tensor([0, *torch.tensor(alignment_lengths).cumsum(0).tolist()])
    aligned_batch = torch.repeat_interleave(
        torch.arange(len(order)), torch.tensor(alignment_lengths)
    )
    return MutWtPairBatch(
        graph_mut=graph_mut,
        graph_wt=graph_wt,
        metadata=[
            {
                "variant_id": variant_id,
                "position": pair_key.position,
                "wt_aa": pair_key.wt_aa,
                "mut_aa": "D",
                "pair_key": pair_key,
            }
            for variant_id, pair_key, _, _ in selected
        ],
        variant_ids=[item[0] for item in selected],
        mutant_keys=[item[2] for item in selected],
        wt_keys=[item[3] for item in selected],
        pair_keys=[item[1] for item in selected],
        batch_size=len(order),
        mut_aligned_index=mut_aligned,
        wt_aligned_index=wt_aligned,
        aligned_pair_batch=aligned_batch,
        alignment_ptr=ptr,
        exists_MUT=torch.ones(mut_aligned.numel(), dtype=torch.bool),
        exists_WT=torch.ones(mut_aligned.numel(), dtype=torch.bool),
        union_pair_batch=aligned_batch.clone(),
        union_ptr=ptr.clone(),
        local_mut_aligned_index=mut_aligned.clone(),
        local_wt_aligned_index=wt_aligned.clone(),
        local_alignment_ptr=ptr.clone(),
    )


def _augmenter(probability: float = 0.5, *, enabled: bool = True) -> ModelAPairAugmenter:
    return ModelAPairAugmenter(
        ModelAPairAugmentationConfig(
            enabled=enabled,
            feature_mask_probability=probability,
            allowed_feature_names=("bsa", "hydrophobicity"),
            masked_value=-99.0,
        )
    )


def test_deep_clone_original_identity_topology_alignment_and_diff_are_preserved() -> None:
    original = make_pair_batch()
    snapshot = deepcopy(original)
    view1, view2 = _augmenter(0.5).create_two_views(
        original, run_seed=7, epoch=3
    )

    for role in ("graph_mut", "graph_wt"):
        source = getattr(original, role)
        saved = getattr(snapshot, role)
        torch.testing.assert_close(source.x, saved.x)
        assert torch.equal(source.edge_index, saved.edge_index)
        assert torch.equal(source.edge_attr, saved.edge_attr)
        assert torch.equal(source.batch, saved.batch)
        assert torch.equal(source.ptr, saved.ptr)
        assert torch.equal(source.is_mutation, saved.is_mutation)
        assert torch.equal(
            source.node_availability_masks["mask_diff_mass"],
            saved.node_availability_masks["mask_diff_mass"],
        )
        assert getattr(view1.pair_batch, role).x.data_ptr() != source.x.data_ptr()
        assert getattr(view2.pair_batch, role).x.data_ptr() != source.x.data_ptr()

    assert view1.variant_ids == tuple(original.variant_ids)
    assert view1.pair_keys == tuple(original.pair_keys)
    assert view1.pair_batch.metadata == original.metadata
    assert view1.pair_batch.mutant_keys == original.mutant_keys
    assert view1.pair_batch.wt_keys == original.wt_keys
    for field in (
        "mut_aligned_index",
        "wt_aligned_index",
        "aligned_pair_batch",
        "alignment_ptr",
        "local_mut_aligned_index",
        "local_wt_aligned_index",
        "local_alignment_ptr",
        "exists_MUT",
        "exists_WT",
        "union_pair_batch",
        "union_ptr",
    ):
        assert torch.equal(getattr(view1.pair_batch, field), getattr(original, field))

    # diff_mass and is_mutation are bitwise protected.
    assert torch.equal(view1.graph_mut.x[:, 2:], original.graph_mut.x[:, 2:])
    assert torch.equal(view2.graph_mut.x[:, 2:], original.graph_mut.x[:, 2:])
    view1.graph_mut.x[0, 0] = 12345
    assert view2.graph_mut.x[0, 0] != 12345
    assert original.graph_mut.x[0, 0] != 12345


def test_reproducibility_view_diversity_and_separate_paired_masks() -> None:
    batch = make_pair_batch()
    augmenter = _augmenter(0.5)
    first = augmenter.create_view(batch, run_seed=91, epoch=4, view_id=1)
    repeated = augmenter.create_view(batch, run_seed=91, epoch=4, view_id=1)
    second = augmenter.create_view(batch, run_seed=91, epoch=4, view_id=2)
    assert torch.equal(first.graph_mut.x, repeated.graph_mut.x)
    assert torch.equal(first.graph_wt.x, repeated.graph_wt.x)
    assert first.effective_seeds == repeated.effective_seeds
    assert first.effective_seeds != second.effective_seeds
    assert not torch.equal(first.graph_mut.x, second.graph_mut.x)

    for view in (first, second):
        assert hasattr(view.graph_mut, "augmentation_feature_mask")
        for pair_index, item in enumerate(view.augmentation_metadata):
            mut_slice = slice(
                int(view.graph_mut.ptr[pair_index]),
                int(view.graph_mut.ptr[pair_index + 1]),
            )
            wt_slice = slice(
                int(view.graph_wt.ptr[pair_index]),
                int(view.graph_wt.ptr[pair_index + 1]),
            )
            mut_decision = view.graph_mut.augmentation_feature_mask[mut_slice].any(0)
            wt_decision = view.graph_wt.augmentation_feature_mask[wt_slice].any(0)
            assert torch.equal(mut_decision, wt_decision)
            assert (
                tuple(torch.nonzero(mut_decision).flatten().tolist())
                == item.masked_column_indices_MUT
            )
        assert torch.equal(
            view.graph_mut.node_availability_masks["mask_diff_mass"],
            batch.graph_mut.node_availability_masks["mask_diff_mass"],
        )


def test_result_is_independent_of_batch_composition_and_order() -> None:
    augmenter = _augmenter(0.5)
    together = augmenter.create_two_views(
        make_pair_batch((0, 1)), run_seed=8, epoch=2
    )
    reversed_views = augmenter.create_two_views(
        make_pair_batch((1, 0)), run_seed=8, epoch=2
    )
    alone_views = augmenter.create_two_views(
        make_pair_batch((0,)), run_seed=8, epoch=2
    )

    def graph_x(view: object, variant_id: str, role: str) -> torch.Tensor:
        index = view.variant_ids.index(variant_id)
        graph = getattr(view, role)
        return graph.x[int(graph.ptr[index]) : int(graph.ptr[index + 1])]

    def metadata(view: object, variant_id: str) -> object:
        return view.augmentation_metadata[view.variant_ids.index(variant_id)]

    def augmentation_mask(view: object, variant_id: str, role: str) -> torch.Tensor:
        index = view.variant_ids.index(variant_id)
        graph = getattr(view, role)
        return graph.augmentation_feature_mask[
            int(graph.ptr[index]) : int(graph.ptr[index + 1])
        ]

    for view_index in (0, 1):
        views = (
            together[view_index],
            reversed_views[view_index],
            alone_views[view_index],
        )
        for role in ("graph_mut", "graph_wt"):
            expected = graph_x(views[0], "v100", role)
            assert torch.equal(expected, graph_x(views[1], "v100", role))
            assert torch.equal(expected, graph_x(views[2], "v100", role))
            expected_mask = augmentation_mask(views[0], "v100", role)
            assert torch.equal(
                expected_mask, augmentation_mask(views[1], "v100", role)
            )
            assert torch.equal(
                expected_mask, augmentation_mask(views[2], "v100", role)
            )
        expected_metadata = metadata(views[0], "v100")
        assert metadata(views[1], "v100") == expected_metadata
        assert metadata(views[2], "v100") == expected_metadata
        assert expected_metadata.variant_id == "v100"
        assert expected_metadata.pair_key == PairingKey("A", 100, "G")
        assert expected_metadata.masked_value == -99.0


def test_disabled_still_returns_two_independent_clones() -> None:
    batch = make_pair_batch()
    view1, view2 = _augmenter(enabled=False).create_two_views(
        batch, run_seed=1, epoch=0
    )
    assert torch.equal(view1.graph_mut.x, batch.graph_mut.x)
    assert torch.equal(view2.graph_mut.x, batch.graph_mut.x)
    assert view1.graph_mut.x.data_ptr() != view2.graph_mut.x.data_ptr()
    assert view1.transformation_names == ()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"feature_mask_probability": -0.1}, r"\[0, 1\]"),
        ({"feature_mask_probability": 1.1}, r"\[0, 1\]"),
        ({"allowed_feature_names": ()}, "non-empty"),
        ({"allowed_feature_names": ("unknown",)}, "Unknown"),
        ({"allowed_feature_names": ("is_mutation",)}, "Protected"),
        ({"allowed_feature_names": ("diff_mass",)}, "Protected"),
        ({"allowed_feature_names": ("mask_diff_mass",)}, "Protected"),
        ({"allowed_feature_names": ("polarity",)}, "Protected"),
        ({"allowed_feature_names": ("bsa", "bsa")}, "duplicates"),
        ({"masked_value": float("nan")}, "finite"),
    ],
)
def test_config_rejects_invalid_or_protected_features(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelAPairAugmentationError, match=message):
        ModelAPairAugmentationConfig(**kwargs)


@pytest.mark.parametrize(
    "feature_name",
    (
        "diff_charge",
        "diff_hb_acceptors",
        "diff_hb_donors",
        "diff_hbond_count",
        "diff_hydrophobicity",
        "diff_mass",
        "diff_pI",
        "diff_polarity",
        "diff_size",
        "mask_diff",
        "mask_diff_charge",
        "mask_diff_hb_acceptors",
        "mask_diff_hb_donors",
        "mask_diff_hbond_count",
        "mask_diff_hydrophobicity",
        "mask_diff_mass",
        "mask_diff_pI",
        "mask_diff_polarity",
        "mask_diff_size",
        "is_mutation",
        "res_id",
        "res_id_norm",
        "res_type",
        "variant_res",
        "polarity",
        "sec_struct",
        "_chain_id",
        "chain_id",
        "structure_quality",
        "is_truncation_node",
        "global_energy",
    ),
)
def test_current_schema_protected_features_are_never_selectable(
    feature_name: str,
) -> None:
    with pytest.raises(ModelAPairAugmentationError):
        ModelAPairAugmentationConfig(allowed_feature_names=(feature_name,))


def test_only_closed_whitelist_groups_are_selectable_from_realistic_layout() -> None:
    realistic_names = (
        *FEATURE_NAMES,
        "hse",
        "res_charge",
        "res_depth",
        "res_mass",
        "res_pI",
        "rsa",
        "sasa",
        "res_type",
        "mask_diff_mass",
        "is_truncation_node",
    )
    selectable = []
    for name in realistic_names:
        try:
            ModelAPairAugmentationConfig(allowed_feature_names=(name,))
        except ModelAPairAugmentationError:
            continue
        selectable.append(name)
    assert tuple(selectable) == (
        "bsa",
        "hydrophobicity",
        "hse",
        "res_charge",
        "res_depth",
        "res_mass",
        "res_pI",
        "rsa",
        "sasa",
    )


def test_absent_requested_feature_fails_informatively() -> None:
    config = ModelAPairAugmentationConfig(allowed_feature_names=("rsa",))
    with pytest.raises(ModelAPairAugmentationError, match="absent"):
        ModelAPairAugmenter(config).create_two_views(
            make_pair_batch(), run_seed=1, epoch=0
        )


def test_stable_seed_uses_all_context_fields() -> None:
    common = {
        "pair_key": PairingKey("A", 10, "G"),
        "mutant_key": "mut",
        "wt_key": "wt",
    }
    base = stable_seed(
        run_seed=1, epoch=2, variant_id="v", view_id=1, transform_id="mask", **common
    )
    assert base == stable_seed(
        run_seed=1, epoch=2, variant_id="v", view_id=1, transform_id="mask", **common
    )
    assert len(
        {
            base,
            stable_seed(run_seed=2, epoch=2, variant_id="v", view_id=1, transform_id="mask", **common),
            stable_seed(run_seed=1, epoch=3, variant_id="v", view_id=1, transform_id="mask", **common),
            stable_seed(run_seed=1, epoch=2, variant_id="w", view_id=1, transform_id="mask", **common),
            stable_seed(run_seed=1, epoch=2, variant_id="v", view_id=2, transform_id="mask", **common),
            stable_seed(run_seed=1, epoch=2, variant_id="v", view_id=1, transform_id="mask", **{**common, "mutant_key": "other"}),
        }
    ) == 6


def test_default_probability_and_degenerate_statuses() -> None:
    assert ModelAPairAugmentationConfig().feature_mask_probability == 0.10
    batch = make_pair_batch()
    zero1, zero2 = _augmenter(0.0).create_two_views(batch, run_seed=2, epoch=0)
    assert torch.equal(zero1.graph_mut.x, batch.graph_mut.x)
    assert torch.equal(zero2.graph_mut.x, batch.graph_mut.x)
    assert {item.status for item in zero1.augmentation_metadata} == {
        "degenerate_p0_no_op"
    }
    assert not any(item.applied for item in zero1.augmentation_metadata)
    one1, one2 = _augmenter(1.0).create_two_views(batch, run_seed=2, epoch=0)
    assert all(item.status == "degenerate_p1_applied" for item in one1.augmentation_metadata)
    assert all(item.selected_feature_names == ("bsa", "hydrophobicity") for item in one2.augmentation_metadata)
    assert torch.equal(
        one1.graph_mut.augmentation_feature_mask,
        one2.graph_mut.augmentation_feature_mask,
    )


def test_two_view_diversity_policy_is_per_example_and_deterministic() -> None:
    batch = make_pair_batch()
    augmenter = _augmenter(0.10)
    first = augmenter.create_two_views(batch, run_seed=17, epoch=8)
    repeated = augmenter.create_two_views(batch, run_seed=17, epoch=8)
    for index in range(batch.batch_size):
        mask1 = tuple(first[0].augmentation_metadata[index].selected_feature_names)
        mask2 = tuple(first[1].augmentation_metadata[index].selected_feature_names)
        assert mask1 != mask2
        assert mask1 == repeated[0].augmentation_metadata[index].selected_feature_names
        assert mask2 == repeated[1].augmentation_metadata[index].selected_feature_names
    assert any(item.diversity_adjusted for item in first[1].augmentation_metadata)
    for item in first[0].augmentation_metadata:
        assert item.diversity_adjustment_seed is None
    for item in first[1].augmentation_metadata:
        assert item.diversity_adjusted == (
            item.diversity_adjustment_seed is not None
        )
    assert first[1].augmentation_metadata == repeated[1].augmentation_metadata


def test_disabled_metadata_and_identity_have_single_source_of_truth() -> None:
    batch = make_pair_batch()
    view, _ = _augmenter(enabled=False).create_two_views(batch, run_seed=3, epoch=0)
    assert all(item.status == "disabled" and not item.applied for item in view.augmentation_metadata)
    view.pair_batch.variant_ids[0] = "changed-in-own-clone"
    assert view.variant_ids[0] == "changed-in-own-clone"
    assert batch.variant_ids[0] == "v100"


def test_disabled_without_layout_returns_independent_no_op_clones() -> None:
    batch = make_pair_batch()
    original_mut = batch.graph_mut.x.clone()
    original_wt = batch.graph_wt.x.clone()
    del batch.graph_mut.node_feature_slices
    del batch.graph_wt.node_feature_slices
    view1, view2 = _augmenter(enabled=False).create_two_views(
        batch, run_seed=3, epoch=0
    )
    for view in (view1, view2):
        assert torch.equal(view.graph_mut.x, original_mut)
        assert torch.equal(view.graph_wt.x, original_wt)
        assert all(
            item.status == "disabled"
            and not item.applied
            and item.diversity_adjustment_seed is None
            for item in view.augmentation_metadata
        )
    assert view1.graph_mut.x.data_ptr() != view2.graph_mut.x.data_ptr()
    assert view1.graph_mut.x.data_ptr() != batch.graph_mut.x.data_ptr()


def test_nonzero_masked_value_preserves_dtype_and_is_recorded() -> None:
    batch = make_pair_batch((0,))
    view = _augmenter(1.0).create_view(batch, run_seed=1, epoch=0, view_id=1)
    assert view.graph_mut.x.dtype == batch.graph_mut.x.dtype
    assert view.augmentation_metadata[0].masked_value == -99.0
    assert torch.all(view.graph_mut.x[:, :2] == -99.0)


def test_hse_is_one_three_column_group_and_neighbors_are_untouched() -> None:
    batch = make_pair_batch((0,))
    for graph in (batch.graph_mut, batch.graph_wt):
        original = graph.x
        graph.x = torch.cat((original[:, :1], original[:, :1] + 10, original[:, :1] + 20, original[:, :1] + 30, original[:, 1:]), dim=1)
        graph.node_feature_names = ("bsa", "hse", "hydrophobicity", "diff_mass", "is_mutation")
        graph.node_feature_slices = (
            NodeFeatureSlice("bsa", 0, 1),
            NodeFeatureSlice("hse", 1, 4),
            NodeFeatureSlice("hydrophobicity", 4, 5),
            NodeFeatureSlice("diff_mass", 5, 6),
            NodeFeatureSlice("is_mutation", 6, 7),
        )
    config = ModelAPairAugmentationConfig(
        feature_mask_probability=1.0,
        allowed_feature_names=("hse",),
        masked_value=-7.0,
    )
    view = ModelAPairAugmenter(config).create_view(
        batch, run_seed=1, epoch=0, view_id=1
    )
    assert torch.all(view.graph_mut.x[:, 1:4] == -7)
    assert torch.equal(view.graph_mut.x[:, 0], batch.graph_mut.x[:, 0])
    assert torch.equal(view.graph_mut.x[:, 4:], batch.graph_mut.x[:, 4:])
    assert view.augmentation_metadata[0].masked_column_indices_MUT == (1, 2, 3)


def test_layout_missing_or_invalid_is_rejected() -> None:
    batch = make_pair_batch((0,))
    del batch.graph_mut.node_feature_slices
    with pytest.raises(ModelAPairAugmentationError, match="missing"):
        _augmenter().create_two_views(batch, run_seed=1, epoch=0)

    batch = make_pair_batch((0,))
    batch.graph_mut.node_feature_slices = (
        NodeFeatureSlice("bsa", 0, 2),
        NodeFeatureSlice("hydrophobicity", 1, 2),
        NodeFeatureSlice("diff_mass", 2, 3),
        NodeFeatureSlice("is_mutation", 3, 4),
    )
    with pytest.raises(ModelAPairAugmentationError, match="Invalid"):
        _augmenter().create_two_views(batch, run_seed=1, epoch=0)


def test_hse_width_mismatch_between_individually_valid_layouts_is_rejected() -> None:
    batch = make_pair_batch((0,))
    batch.graph_mut.x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    batch.graph_mut.node_feature_names = ("bsa", "hse", "sasa")
    batch.graph_mut.node_feature_slices = (
        NodeFeatureSlice("bsa", 0, 1),
        NodeFeatureSlice("hse", 1, 4),
        NodeFeatureSlice("sasa", 4, 5),
    )
    batch.graph_wt.x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    batch.graph_wt.node_feature_names = ("bsa", "hse", "sasa")
    batch.graph_wt.node_feature_slices = (
        NodeFeatureSlice("bsa", 0, 1),
        NodeFeatureSlice("hse", 1, 3),
        NodeFeatureSlice("sasa", 3, 4),
    )
    config = ModelAPairAugmentationConfig(allowed_feature_names=("hse",))
    with pytest.raises(
        ModelAPairAugmentationError,
        match=r"Feature 'hse' has incompatible MUT/WT widths \(3 != 2\)",
    ):
        ModelAPairAugmenter(config).create_view(
            batch, run_seed=1, epoch=0, view_id=1
        )


def test_hse_numeric_masking_resolves_different_mut_wt_offsets() -> None:
    batch = make_pair_batch((0,))
    batch.graph_mut.x = torch.tensor(
        [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]],
        dtype=torch.float32,
    )
    batch.graph_mut.node_feature_names = ("bsa", "hse", "res_type")
    batch.graph_mut.node_feature_slices = (
        NodeFeatureSlice("bsa", 0, 1),
        NodeFeatureSlice("hse", 1, 4),
        NodeFeatureSlice("res_type", 4, 5),
    )
    batch.graph_wt.x = torch.tensor(
        [
            [100, 101, 102, 103, 104, 105],
            [110, 111, 112, 113, 114, 115],
            [120, 121, 122, 123, 124, 125],
        ],
        dtype=torch.float32,
    )
    batch.graph_wt.node_feature_names = ("res_type", "hse", "bsa")
    batch.graph_wt.node_feature_slices = (
        NodeFeatureSlice("res_type", 0, 2),
        NodeFeatureSlice("hse", 2, 5),
        NodeFeatureSlice("bsa", 5, 6),
    )
    original_mut = batch.graph_mut.x.clone()
    original_wt = batch.graph_wt.x.clone()
    config = ModelAPairAugmentationConfig(
        feature_mask_probability=1.0,
        allowed_feature_names=("hse",),
        masked_value=-7.0,
    )
    view = ModelAPairAugmenter(config).create_view(
        batch, run_seed=1, epoch=0, view_id=1
    )
    assert batch.graph_mut.node_feature_slices[1] == NodeFeatureSlice("hse", 1, 4)
    assert batch.graph_wt.node_feature_slices[1] == NodeFeatureSlice("hse", 2, 5)
    assert torch.equal(view.graph_mut.x[:, 1:4], torch.full((3, 3), -7.0))
    assert torch.equal(view.graph_wt.x[:, 2:5], torch.full((3, 3), -7.0))
    assert torch.equal(view.graph_mut.x[:, 0], original_mut[:, 0])
    assert torch.equal(view.graph_mut.x[:, 4], original_mut[:, 4])
    assert torch.equal(view.graph_wt.x[:, :2], original_wt[:, :2])
    assert torch.equal(view.graph_wt.x[:, 5], original_wt[:, 5])
    item = view.augmentation_metadata[0]
    assert item.masked_column_indices_MUT == (1, 2, 3)
    assert item.masked_column_indices_WT == (2, 3, 4)


DIFF_FEATURE_NAMES = (
    "diff_mass",
    "diff_charge",
    "diff_pI",
    "diff_size",
    "diff_hb_donors",
    "diff_hb_acceptors",
    "diff_polarity",
)
DIFF_MASK_NAMES = tuple(f"mask_{name}" for name in DIFF_FEATURE_NAMES)
OTHER_PROTECTED_NAMES = (
    "is_mutation",
    "res_id",
    "res_id_norm",
    "res_type",
    "variant_res",
    "polarity",
    "sec_struct",
    "chain_id",
    "structure_quality",
    "is_truncation_node",
    "global_energy",
)


def _make_protected_feature_batch() -> MutWtPairBatch:
    batch = make_pair_batch((0,))
    names = ("bsa", *DIFF_FEATURE_NAMES, *OTHER_PROTECTED_NAMES)
    slices = tuple(NodeFeatureSlice(name, index, index + 1) for index, name in enumerate(names))
    signs = torch.tensor([1, -2, 3, -4, 5, -6, 7], dtype=torch.float32)
    for role_index, graph in enumerate((batch.graph_mut, batch.graph_wt), start=1):
        rows = graph.num_nodes
        columns = [torch.arange(rows, dtype=torch.float32) + 10 * role_index]
        columns.extend(
            torch.full((rows,), float(value))
            for value in signs.tolist()
        )
        columns.extend(
            torch.full((rows,), 100.0 * role_index + index)
            for index in range(len(OTHER_PROTECTED_NAMES))
        )
        graph.x = torch.stack(columns, dim=1)
        graph.node_feature_names = names
        graph.node_feature_slices = slices
        graph.node_availability_masks = {
            mask_name: torch.full(
                (rows,), float(1000 * role_index + mask_index)
            )
            for mask_index, mask_name in enumerate(DIFF_MASK_NAMES)
        }
    return batch


def test_all_differential_masks_and_other_protected_inputs_are_bitwise_preserved() -> None:
    batch = _make_protected_feature_batch()
    originals = {
        role: getattr(batch, role).x.clone()
        for role in ("graph_mut", "graph_wt")
    }
    original_masks = {
        role: {
            name: value.clone()
            for name, value in getattr(batch, role).node_availability_masks.items()
        }
        for role in ("graph_mut", "graph_wt")
    }
    config = ModelAPairAugmentationConfig(
        feature_mask_probability=1.0,
        allowed_feature_names=("bsa",),
        masked_value=-99.0,
    )
    views = ModelAPairAugmenter(config).create_two_views(
        batch, run_seed=4, epoch=2
    )
    names = batch.graph_mut.node_feature_names
    protected_names = (*DIFF_FEATURE_NAMES, *OTHER_PROTECTED_NAMES)
    for view in views:
        for role in ("graph_mut", "graph_wt"):
            graph = getattr(view, role)
            original = originals[role]
            for name in protected_names:
                column = names.index(name)
                assert torch.equal(graph.x[:, column], original[:, column]), name
            for name in DIFF_MASK_NAMES:
                assert torch.equal(
                    graph.node_availability_masks[name],
                    original_masks[role][name],
                ), name
            for name in DIFF_FEATURE_NAMES:
                column = names.index(name)
                assert torch.equal(
                    torch.sign(graph.x[:, column]),
                    torch.sign(original[:, column]),
                ), name
        for item in view.augmentation_metadata:
            assert item.selected_feature_names == ("bsa",)
            assert not set(item.selected_feature_names).intersection(DIFF_FEATURE_NAMES)


def test_diversity_adjustment_seed_tracks_context_and_inverted_group() -> None:
    batch = make_pair_batch()
    augmenter = _augmenter(0.10)
    view1, view2 = augmenter.create_two_views(batch, run_seed=17, epoch=8)
    adjusted_index = next(
        index
        for index, item in enumerate(view2.augmentation_metadata)
        if item.diversity_adjusted
    )
    item1 = view1.augmentation_metadata[adjusted_index]
    item2 = view2.augmentation_metadata[adjusted_index]
    expected_seed = stable_seed(
        run_seed=17,
        epoch=8,
        variant_id=batch.variant_ids[adjusted_index],
        pair_key=batch.pair_keys[adjusted_index],
        mutant_key=batch.mutant_keys[adjusted_index],
        wt_key=batch.wt_keys[adjusted_index],
        view_id=2,
        transform_id=augmenter.diversity_transform_id,
    )
    assert item2.diversity_adjustment_seed == expected_seed
    changed_groups = set(item1.selected_feature_names).symmetric_difference(
        item2.selected_feature_names
    )
    expected_group = augmenter.config.allowed_feature_names[
        expected_seed % len(augmenter.config.allowed_feature_names)
    ]
    assert changed_groups == {expected_group}

    repeated = augmenter.create_two_views(batch, run_seed=17, epoch=8)[1]
    assert (
        repeated.augmentation_metadata[adjusted_index].diversity_adjustment_seed
        == expected_seed
    )
    next_epoch = augmenter.create_two_views(batch, run_seed=17, epoch=9)[1]
    next_identity = deepcopy(batch)
    next_identity.mutant_keys[adjusted_index] += "-different"
    changed_identity = augmenter.create_two_views(
        next_identity, run_seed=17, epoch=8
    )[1]
    assert next_epoch.augmentation_metadata[adjusted_index].effective_seed != item2.effective_seed
    assert (
        changed_identity.augmentation_metadata[adjusted_index].effective_seed
        != item2.effective_seed
    )
    epoch_adjustment_seed = stable_seed(
        run_seed=17,
        epoch=9,
        variant_id=batch.variant_ids[adjusted_index],
        pair_key=batch.pair_keys[adjusted_index],
        mutant_key=batch.mutant_keys[adjusted_index],
        wt_key=batch.wt_keys[adjusted_index],
        view_id=2,
        transform_id=augmenter.diversity_transform_id,
    )
    identity_adjustment_seed = stable_seed(
        run_seed=17,
        epoch=8,
        variant_id=next_identity.variant_ids[adjusted_index],
        pair_key=next_identity.pair_keys[adjusted_index],
        mutant_key=next_identity.mutant_keys[adjusted_index],
        wt_key=next_identity.wt_keys[adjusted_index],
        view_id=2,
        transform_id=augmenter.diversity_transform_id,
    )
    assert epoch_adjustment_seed != expected_seed
    assert identity_adjustment_seed != expected_seed
