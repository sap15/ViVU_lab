from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from gnn_siamese.data import GraphViewAugmenter, collate_mut_wt_pairs, resolve_graph_augmentation_config
from gnn_siamese.data.augmentations import AugmentationConfigError
from gnn_siamese.data.dataset import MutWtPairDataset
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def _make_batch(tmp_path: Path):
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    dataset = MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=config,
        schema=json.loads(schema_path.read_text(encoding="utf-8")),
    )
    return collate_mut_wt_pairs([dataset[0], dataset[1]]), config


def test_augmentations_create_independent_views_and_preserve_protected_features(tmp_path: Path) -> None:
    batch, config = _make_batch(tmp_path)
    original_x = batch.graph_mut.x.clone()
    original_edge_index = batch.graph_mut.edge_index.clone()
    augmentation_config = resolve_graph_augmentation_config(config, seed=123)
    augmenter = GraphViewAugmenter(config=augmentation_config, node_feature_names=batch.graph_mut.node_feature_names)

    view1, view2 = augmenter.create_two_views(batch.graph_mut)

    assert view1 is not view2
    assert view1 is not batch.graph_mut
    torch.testing.assert_close(batch.graph_mut.x, original_x)
    torch.testing.assert_close(batch.graph_mut.edge_index, original_edge_index)
    assert not torch.equal(view1.x, view2.x) or not torch.equal(view1.edge_index, view2.edge_index)

    feature_to_index = {name: index for index, name in enumerate(batch.graph_mut.node_feature_names)}
    for protected_name in ("diff_mass", "diff_charge", "diff_pI", "diff_size", "is_mutation"):
        index = feature_to_index[protected_name]
        torch.testing.assert_close(view1.x[:, index], batch.graph_mut.x[:, index])
        torch.testing.assert_close(view2.x[:, index], batch.graph_mut.x[:, index])

    for key, mask in batch.graph_mut.node_availability_masks.items():
        torch.testing.assert_close(view1.node_availability_masks[key], mask)
        torch.testing.assert_close(view2.node_availability_masks[key], mask)
    assert view1.edge_index.shape[1] > 0
    assert view2.edge_index.shape[1] > 0


def test_disabled_augmentations_preserve_exact_graph_data(tmp_path: Path) -> None:
    batch, config = _make_batch(tmp_path)
    config["augmentation"]["enabled"] = False
    augmenter = GraphViewAugmenter(
        config=resolve_graph_augmentation_config(config, seed=123),
        node_feature_names=batch.graph_mut.node_feature_names,
    )

    view1, view2 = augmenter.create_two_views(batch.graph_mut)

    torch.testing.assert_close(view1.x, batch.graph_mut.x)
    torch.testing.assert_close(view2.x, batch.graph_mut.x)
    torch.testing.assert_close(view1.edge_index, batch.graph_mut.edge_index)
    torch.testing.assert_close(view2.edge_index, batch.graph_mut.edge_index)
    torch.testing.assert_close(view1.is_mutation, batch.graph_mut.is_mutation)
    torch.testing.assert_close(view2.is_mutation, batch.graph_mut.is_mutation)


def test_same_seed_reproduces_the_same_two_views(tmp_path: Path) -> None:
    batch, config = _make_batch(tmp_path)
    augmentation_config = resolve_graph_augmentation_config(config, seed=123)
    augmenter_a = GraphViewAugmenter(config=augmentation_config, node_feature_names=batch.graph_mut.node_feature_names)
    augmenter_b = GraphViewAugmenter(config=augmentation_config, node_feature_names=batch.graph_mut.node_feature_names)

    a1, a2 = augmenter_a.create_two_views(batch.graph_mut)
    b1, b2 = augmenter_b.create_two_views(batch.graph_mut)

    torch.testing.assert_close(a1.x, b1.x)
    torch.testing.assert_close(a2.x, b2.x)
    torch.testing.assert_close(a1.edge_index, b1.edge_index)
    torch.testing.assert_close(a2.edge_index, b2.edge_index)


def test_augmentations_raise_for_unknown_node_feature_names(tmp_path: Path) -> None:
    batch, config = _make_batch(tmp_path)
    config["augmentation"]["feature_dropout"]["allowed_feature_names"] = ["missing_feature"]
    augmentation_config = resolve_graph_augmentation_config(config, seed=123)

    with pytest.raises(AugmentationConfigError, match="Augmentation references unknown node features"):
        GraphViewAugmenter(config=augmentation_config, node_feature_names=batch.graph_mut.node_feature_names)
